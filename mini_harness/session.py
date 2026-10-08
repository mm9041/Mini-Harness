"""``ctx.sessions`` —— 追加式会话事件日志,以及从日志投影出模型历史。

对应 dsh 的 ``packages/core/session``。这里承载 dsh 最核心的一条不变量:

    **model-visible means logged** —— 模型看到的上下文,必须能从会话日志重建。

所以本模块只有两件事:① ``append`` 追加事实;② ``derive_messages`` 把日志投影成
模型历史。agent loop 不许自己攒消息列表,它必须落盘再投影。
"""

from __future__ import annotations

import json
import logging
import random
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .kernel import Context, Plugin
from .diagnostics import report_exception
from .llm import Message, ToolCall
from .user_messages import model_user_text

__all__ = [
    "SessionEvent",
    "Session",
    "SessionsService",
    "repair_interrupted_tail",
    "plugin",
]


@dataclass
class SessionEvent:
    """日志里的一条事实。``seq`` 单调递增,顺序即真相。"""

    seq: int
    type: str
    data: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def to_json(self) -> dict[str, Any]:
        return {"seq": self.seq, "ts": self.ts, "type": self.type, "data": self.data}


def repair_interrupted_tail(session: "Session") -> list[str]:
    """补齐中断留下的窟窿,返回做了哪些修复。

    三件事:

    1. **没有结果的工具调用** —— 强杀时 ``tool/call`` 可能已经写进日志但结果还没写。
       这种历史是**协议非法**的:``assistant(tool_calls)`` 后面必须跟对应的 tool 消息,
       否则下一个请求会被 API 直接拒掉。补成"未执行:上一次运行被中断"。
    2. 未闭合的 ``step``;
    3. 未闭合的 ``turn``。

    注意这些都是**补事实,不是编内容**:被补的事件都带 ``repaired=True``,
    谁补的、为什么补,日志里看得见。
    """
    announced: list[tuple[str, str]] = []
    answered: set[str] = set()
    open_steps = 0
    open_turns = 0

    for event in session.events:
        if event.type == "assistant/message":
            for call in event.data.get("tool_calls") or []:
                announced.append((call.get("id") or "", call.get("name") or ""))
        elif event.type == "tool/result":
            answered.add(event.data.get("call_id") or "")
        elif event.type == "step/start":
            open_steps += 1
        elif event.type == "step/end" and open_steps:
            open_steps -= 1
        elif event.type == "turn/start":
            open_turns += 1
        elif event.type == "turn/end" and open_turns:
            open_turns -= 1

    notes: list[str] = []

    missing = [(call_id, name) for call_id, name in announced if call_id and call_id not in answered]
    for call_id, name in missing:
        session.append(
            "tool/result",
            call_id=call_id,
            name=name,
            content="未执行:上一次运行被中断,这条调用的结果没有记录",
            is_error=True,
            repaired=True,
        )
    if missing:
        notes.append(f"补上了 {len(missing)} 条缺失的工具结果(上次运行被中断)")

    for _ in range(open_steps):
        session.append("step/end", cancelled=True, repaired=True)
    if open_steps:
        notes.append(f"补上了 {open_steps} 个未闭合的 step")

    for _ in range(open_turns):
        session.append("turn/end", stopped="interrupted", repaired=True)
    if open_turns:
        notes.append(f"补上了 {open_turns} 个未闭合的 turn(stopped=interrupted)")

    return notes


class Session:
    """一次会话。事件只追加、不修改、不删除。"""

    def __init__(self, session_id: str) -> None:
        self.id = session_id
        self.created_at = time.time()
        self.events: list[SessionEvent] = []
        #: 从哪个文件读回来的(续跑时用来写回同一个文件)。
        self.source_path: Path | None = None
        #: 打开时做过哪些修复/跳过(续跑时打给人看)。
        self.repairs: list[str] = []
        self._recovery_bytes: bytes | None = None
        self._recovery_source: Path | None = None
        self.recovery_backup: Path | None = None
        # 观察者:对应 dsh 的 session/event 广播。日志追加是同步路径,所以这里用
        # 同步回调而不是 await,避免在 append 上引入异步。
        self.observers: list[Callable[[SessionEvent], None]] = []

    # ------------------------------------------------------------ 写:追加事实
    def append(self, type_: str, **data: Any) -> SessionEvent:
        event = SessionEvent(self.events[-1].seq + 1 if self.events else 1, type_, dict(data))
        self.events.append(event)
        for observer in list(self.observers):
            try:
                observer(event)
            except Exception:
                # The event is already committed; a display/subscriber failure
                # must not undo it or prevent other observers from seeing it.
                report_exception(logging.getLogger(__name__), "会话事件观察者失败 (seq=%s, type=%s)", event.seq, event.type, level=logging.ERROR)
        return event

    def observe(self, callback: Callable[[SessionEvent], None]) -> Callable[[], None]:
        self.observers.append(callback)

        def dispose() -> None:
            if callback in self.observers:
                self.observers.remove(callback)

        return dispose

    # ------------------------------------------------------------ 读:投影历史
    def derive_messages(self) -> list[Message]:
        """把日志折叠成模型历史。这是"投影",不是"第二份真相"。

        每条消息都记下 ``source_seq``(来自哪条事件)—— 压缩靠它把边界对齐到事件序号上。
        """
        messages: list[Message] = []
        pending_images: list[dict[str, Any]] = []
        pending_calls: set[str] = set()
        for event in self.events:
            if event.type == "user/message":
                pending_images = []
                pending_calls = set()
                messages.append(
                    Message(
                        role="user",
                        content=model_user_text(event.data),
                        source_seq=event.seq,
                    )
                )
            elif event.type == "assistant/message":
                pending_images = []  # Never carry an incomplete prior tool batch into this one.
                pending_calls = {call["id"] for call in event.data.get("tool_calls") or []}
                messages.append(
                    Message(
                        role="assistant",
                        content=event.data.get("text"),
                        reasoning=event.data.get("reasoning"),
                        tool_calls=[
                            ToolCall(
                                id=raw["id"],
                                name=raw["name"],
                                arguments=raw.get("arguments") or {},
                            )
                            for raw in event.data.get("tool_calls") or []
                        ],
                        source_seq=event.seq,
                    )
                )
            elif event.type == "tool/result":
                messages.append(
                    Message(
                        role="tool",
                        content=event.data.get("content"),
                        tool_call_id=event.data.get("call_id"),
                        source_seq=event.seq,
                    )
                )
                pending_calls.discard(event.data.get("call_id"))
                pending_images.extend(event.data.get("images") or [])
                if pending_images and not pending_calls:
                    messages.append(Message(role="user", content="以下图像来自 read_image 工具的输出，供当前任务参考。", images=pending_images, source_seq=event.seq))
                    pending_images = []
        return messages

    def compactions(self) -> list[dict[str, Any]]:
        """日志里记着的压缩决定,按发生顺序。

        压缩**必须落日志**:它是不可重放的一次性决定(摘要文本是模型生成的),
        只存在内存里的话,续跑之后这一段历史就凭空变回原样了。
        """
        return [
            event.data for event in self.events if event.type == "compaction"
        ]

    def events_of(self, type_: str) -> list[SessionEvent]:
        return [event for event in self.events if event.type == type_]

    # ------------------------------------------------------------------ 持久化
    def to_jsonl(self) -> str:
        return "\n".join(
            json.dumps(event.to_json(), ensure_ascii=False) for event in self.events
        ) + ("\n" if self.events else "")

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        content = self.to_jsonl()
        if self._recovery_bytes is not None and target.resolve() == self._recovery_source:
            if target.read_bytes() != self._recovery_bytes:
                raise OSError("会话文件在读取后已改变，拒绝覆盖，请重新打开")
            if self.recovery_backup is None:
                fd, backup = tempfile.mkstemp(dir=target.parent, prefix=target.name + '.recovery-', suffix='.bak')
                try:
                    with os.fdopen(fd, 'wb') as stream:
                        stream.write(self._recovery_bytes)
                        stream.flush()
                        os.fsync(stream.fileno())
                except BaseException:
                    Path(backup).unlink(missing_ok=True)
                    raise
                self.recovery_backup = Path(backup)
                self.repairs.append(f"原始会话已备份至: {backup}")
        elif target.is_file() and target.read_text(encoding="utf-8-sig") == content:
            return target
        # Replace atomically: a crash must not truncate the previous checkpoint.
        fd, temporary = tempfile.mkstemp(dir=target.parent, prefix=".session-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            # Windows readers (including concurrent history scans) may briefly
            # deny replacement. Keep the previous checkpoint intact and retry
            # only these sharing/access errors for a bounded 310 ms in total.
            for attempt in range(6):
                try:
                    os.replace(temporary, target)
                    break
                except OSError as exc:
                    if os.name != "nt" or getattr(exc, "winerror", None) not in (5, 32, 33) or attempt == 5:
                        raise
                    time.sleep(0.01 * 2 ** attempt)
            if target.resolve() == self._recovery_source:
                self._recovery_bytes = None
                self._recovery_source = None
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return target


class SessionsService:
    """会话工厂。对应 ``ctx.sessions``(dsh 里还有 stat/list/export,这里只留 create/open/save)。"""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root else Path.cwd() / ".mini-harness" / "sessions"
        self._live: dict[str, Session] = {}

    def create(self, session_id: str | None = None) -> Session:
        session = Session(session_id or self.new_id())
        self._live[session.id] = session
        return session

    def release(self, session: Session) -> None:
        """Release a caller-owned, finished session from the in-memory registry."""
        if self._live.get(session.id) is session:
            self._live.pop(session.id)

    def open(self, path: str | Path) -> Session:
        """从 JSONL 读回一个会话 —— 日志是唯一的真相来源,重放即可恢复。

        读取是**容错**的:坏行跳过并记一笔,首次覆盖源文件前备份其原始字节。
        有效事件的序号必须为正整数且严格递增;不重排或改号，以保留压缩边界语义。
        读完还会补齐未闭合的 step/turn(见 ``repair_interrupted_tail``)。
        """
        source = Path(path)
        session = Session(source.stem)
        skipped = 0
        raw_bytes = source.read_bytes()
        for line in raw_bytes.decode("utf-8-sig").splitlines():
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                if not isinstance(raw, dict) or not isinstance(raw.get("type"), str) or not isinstance(raw.get("data", {}), dict):
                    raise ValueError("无效事件结构")
                seq = raw["seq"]
                if isinstance(seq, bool) or not isinstance(seq, int) or seq <= 0:
                    raise ValueError("无效事件序号")
            except (ValueError, KeyError, TypeError):
                skipped += 1
                continue
            if session.events and seq <= session.events[-1].seq:
                raise ValueError(f"会话事件序号重复或倒退 (seq={seq})，已保留原文件，无法安全续跑")
            session.events.append(SessionEvent(seq=seq, type=raw["type"], data=raw.get("data", {}), ts=raw.get("ts", time.time())))
        if skipped:
            session._recovery_bytes = raw_bytes
            session._recovery_source = source.resolve()
            session.repairs.append(f"跳过了 {skipped} 行无法解析的事件；首次写回源文件前会备份原始内容")
        session.repairs.extend(repair_interrupted_tail(session))
        session.source_path = source
        self._live[session.id] = session
        return session

    def save(self, session: Session, path: str | Path | None = None) -> Path:
        """写回 JSONL。

        不指定路径时优先写回**它读回来的那个文件**(续跑的语义),否则写到会话目录。
        写回是同名覆盖 —— 迷你版没有 dsh 那套"版本化代次永不覆盖"的规矩,
        每次落盘都是全量重写。读取时跳过的坏行不进入投影，首次覆盖前会留存原始备份。
        """
        if path is not None:
            target = Path(path)
        elif session.source_path is not None:
            target = session.source_path
        else:
            target = self.root / f"{session.id}.jsonl"
        saved = session.save(target)
        session.source_path = saved
        return saved

    def list_sessions(self, limit: int = 10) -> list[Path]:
        """最近的非零字节会话文件,按修改时间倒序;不修改或删除空文件。"""
        if not self.root.is_dir():
            return []
        found = []
        for path in self.root.glob("*.jsonl"):
            try:
                stat = path.stat()
                if path.is_file() and stat.st_size > 0:
                    found.append((stat.st_mtime, path))
            except FileNotFoundError:
                continue  # Archived/deleted during catalogue enumeration.
        return [path for _, path in sorted(found, reverse=True)[:limit]]

    def latest(self) -> Path | None:
        """最近一次会话 —— ``--continue`` 用它。"""
        found = self.list_sessions(limit=1)
        return found[0] if found else None

    @staticmethod
    def new_id() -> str:
        return time.strftime("%Y%m%d-%H%M%S") + "-" + f"{random.randrange(16**4):04x}"


def plugin(root: str | Path | None = None) -> Plugin:
    def apply(ctx: Context) -> None:
        ctx.provide("sessions", SessionsService(root))

    return Plugin(name="session", apply=apply, description="追加式会话日志")
