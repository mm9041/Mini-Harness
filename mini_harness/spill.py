"""``ctx.spillStore`` + 结果外溢策略 —— 不让"太大"退化成"丢内容"。

对应 dsh 的三个包,这里按同样的三条职责分开写(同一个文件,但边界清楚):

* **存储**(`SpillStore` / `LocalSpillStore`)—— 把超长文本落到**会话级私有目录**,
  返回**不透明 locator + 精确字符/字节数**。对应 `dsh-spill` + `dsh-spill-local`;
* **策略**(`SpillPolicy`)—— 多大算大、头尾各留多少、检索指引怎么写。对应 `dsh-spill-policy`;
* **接线**(`plugin()`)—— 挂在 ``tools/post-execute`` 上,于是**任何**工具的输出都被
  同一套预算管住,工具本身一行都不用改。

为什么"不透明 locator"是关键:模型拿到的是**路径**,不是内容。想要细节就自己
``read`` 去读,或 ``grep``/``sed`` 只看一段 —— 而不是把全文灌进上下文。

两个从 dsh 学来的细节:

1. 文件名**不可预测**(随机后缀)。会话目录是稳定的,但如果文件名可猜,别的进程
   就能预先种一个符号链接把你的输出重定向出去("planted symlink");
2. 存不下时**不回退内容**:这里选择折成头尾硬截断并**明说全文已不可恢复**,
   而不是把 20 万字符原样交回去把上下文撑爆。dsh 把选择权留给调用方,迷你版直接定死。
"""

from __future__ import annotations

import secrets
import os
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .kernel import MODE_WATERFALL, Context, Plugin
from .diagnostics import report_exception
from .tools import ToolResult

__all__ = [
    "SpillRecord",
    "SpillStore",
    "LocalSpillStore",
    "SpillPolicy",
    "DEFAULT_MAX_INLINE_CHARS",
    "plugin",
]

DEFAULT_MAX_INLINE_CHARS = 12000
DEFAULT_HEAD_CHARS = 2000
DEFAULT_TAIL_CHARS = 2000
DEFAULT_RETENTION_DAYS = 7
DEFAULT_MAX_SESSION_FILES = 1000
DEFAULT_MAX_SESSION_BYTES = 256 * 1024 * 1024
_STORAGE_LOCK = threading.RLock()  # Stores in independent conversation contexts can share a root.


def default_root() -> Path:
    """外溢目录的默认位置:用户主目录下的私有目录,不落在工作区里。

    为什么不放进工作区:外溢是**缓存**,不是产物。写进用户的仓库会污染 git status,
    而且多会话共享一个根目录时更容易互相看见。dsh 的 `spill-local` 同样是私有会话目录。
    """
    return Path.home() / ".mini-harness" / "spill"


@dataclass
class SpillRecord:
    """一次外溢的结果。"""

    locator: str
    chars: int
    bytes: int


class SpillStore(Protocol):
    """外溢存储的接口。想换成 S3 / 数据库,实现这一个方法即可。"""

    def save(self, text: str, *, session_id: str, hint: str = "") -> SpillRecord: ...


def _slug(text: str, limit: int = 24) -> str:
    """保留 Unicode 字母/数字（含中文）、下划线和短横线，其余替换为短横线。"""
    kept = [char if char.isalnum() or char in "-_" else "-" for char in text]
    collapsed = "".join(kept).strip("-")
    while "--" in collapsed:
        collapsed = collapsed.replace("--", "-")
    return collapsed[:limit].strip("-")


class LocalSpillStore:
    """落到本地会话级目录。对应 ``dsh-spill-local``。

    目录布局::

        <root>/<session_id>/<HHMMSS>-<随机 8 位十六进制>[-提示词].txt

    会话目录稳定(便于清理与排查),**文件名不可预测**(防符号链接重定向)。
    """

    def __init__(
        self,
        root: Path | str | None = None,
        retention_days: int = DEFAULT_RETENTION_DAYS,
        *, max_session_files: int = DEFAULT_MAX_SESSION_FILES,
        max_session_bytes: int = DEFAULT_MAX_SESSION_BYTES,
    ) -> None:
        for value in (max_session_files, max_session_bytes):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("外溢文件数和字节配额必须为正整数")
        self.root = (Path(root).expanduser() if root else default_root()).resolve()
        self.retention_days = retention_days
        self.max_session_files, self.max_session_bytes = max_session_files, max_session_bytes
        self._next_cleanup = 0.0

    # ------------------------------------------------------------------ 写入
    def session_dir(self, session_id: str) -> Path:
        directory = self.root / (_slug(session_id) or "anonymous")
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        if directory.resolve() != directory:
            raise OSError("外溢会话目录不能是符号链接或目录联接")
        self._harden(directory, 0o700)
        return directory

    def save(self, text: str, *, session_id: str, hint: str = "") -> SpillRecord:
        with _STORAGE_LOCK:
            self.cleanup_if_due()
            return self._save(text, session_id=session_id, hint=hint)

    def _save(self, text: str, *, session_id: str, hint: str) -> SpillRecord:
        directory = self.session_dir(session_id)
        stamp = time.strftime("%H%M%S")
        suffix = _slug(hint)
        name = f"{stamp}-{secrets.token_hex(4)}{('-' + suffix) if suffix else ''}.txt"
        target = directory / name

        data = text.encode("utf-8")
        files = [item for item in directory.iterdir() if item.is_file()]
        if len(files) >= self.max_session_files or sum(item.stat().st_size for item in files) + len(data) > self.max_session_bytes:
            raise OSError("外溢会话存储配额已满，未保存新输出")
        # Restrict access at creation and never follow/overwrite a pre-existing name.
        fd = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0), 0o600)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
        except BaseException:
            target.unlink(missing_ok=True)
            raise
        return SpillRecord(locator=str(target), chars=len(text), bytes=len(data))

    # ------------------------------------------------------------------ 清理
    def cleanup(self, now: float | None = None) -> list[str]:
        """删掉超过保留期的会话目录,返回被删掉的名字。

        按最近新增文件的时间回收；到期后旧 locator 可能失效。
        会话日志通常只保存预览，不保证包含这份完整快照。
        """
        with _STORAGE_LOCK:
            return self._cleanup(now)

    def cleanup_if_due(self) -> None:
        with _STORAGE_LOCK:
            if time.monotonic() < self._next_cleanup:
                return
            self._next_cleanup = time.monotonic() + 3600
            try:
                self.cleanup()
            except OSError:
                report_exception(logging.getLogger(__name__), "外溢缓存清理失败，将在下次检查时重试")

    def _cleanup(self, now: float | None = None) -> list[str]:
        if self.retention_days <= 0 or not self.root.is_dir():
            return []
        deadline = (time.time() if now is None else now) - self.retention_days * 86400
        removed: list[str] = []
        for directory in self.root.iterdir():
            if not directory.is_dir() or directory.resolve() != directory:
                continue
            try:
                stale = directory.stat().st_mtime < deadline
            except OSError:
                continue
            if not stale:
                continue
            try:
                for child in directory.iterdir():
                    child.unlink()
                directory.rmdir()
                removed.append(directory.name)
            except OSError:
                report_exception(logging.getLogger(__name__), "无法清理外溢会话目录 %s", directory)
        return removed

    @staticmethod
    def _harden(path: Path, mode: int) -> None:
        """尽力收紧权限。POSIX 上有效;Windows 上 chmod 基本是空操作,不报错即可。"""
        try:
            path.chmod(mode)
        except OSError:
            pass


@dataclass
class SpillPolicy:
    """多大算大、头尾留多少。对应 ``dsh-spill-policy``。"""

    max_inline_chars: int = DEFAULT_MAX_INLINE_CHARS
    head_chars: int = DEFAULT_HEAD_CHARS
    tail_chars: int = DEFAULT_TAIL_CHARS

    def needs_spill(self, text: str) -> bool:
        return len(text) > self.max_inline_chars

    def window(self, text: str) -> tuple[str, str, int]:
        """按预算切出头尾,返回 ``(head, tail, 省略字符数)``。

        为什么要在这里夹一道:``head_chars + tail_chars`` 有可能超过
        ``max_inline_chars``(三个旋钮是独立的)。真出现这种配置,预览会比原文还长 ——
        那"外溢"就完全没有意义了。所以预算不够时按 6:4 缩头尾。
        """
        budget = max(0, self.max_inline_chars)
        head_chars, tail_chars = self.head_chars, self.tail_chars
        if head_chars + tail_chars > budget:
            head_chars = int(budget * 0.6)
            tail_chars = max(0, budget - head_chars)

        head = text[:head_chars]
        tail = text[len(text) - tail_chars :] if tail_chars else ""
        return head, tail, len(text) - len(head) - len(tail)

    def render(self, text: str, record: SpillRecord, hint: str = "") -> str:
        """有 locator 的预览:头 + 省略标记 + 尾 + **检索指引**。

        指引不是客套话。模型如果不知道"可以再去读",它就会拿这份残缺内容硬答;
        告诉它"用 read 读它 / 用 grep 看一段",它才会去取细节。

        还有一句"这是工具输出快照、行号以快照为准"—— 这是实测踩出来的:
        外溢存的是**工具当次的输出**(read 的输出自带 2 行表头),
        行号和原始文件差几行。模型不被告知就会算错。
        """
        head, tail, elided = self.window(text)
        source = f"{hint} 工具的输出快照" if hint else "该工具的输出快照"
        return (
            f"{head}\n\n"
            f"... [中间省略 {elided} 字符] ...\n\n"
            f"{tail}\n\n"
            f"[{source},全文 {record.chars} 字符已保存到 {record.locator}]\n"
            "需要细节就 read 读它(它的行号以这份快照为准,可能与原始文件有偏移),"
            "或用 shell 的 grep/sed 只看需要的那一段 —— 不要试图一次把全文读回上下文。"
        )

    def render_without_storage(self, text: str, reason: str = "") -> str:
        """存不下时的退路:头尾硬截断,**并明说内容已不可恢复**。"""
        head, tail, elided = self.window(text)
        detail = f"（{' '.join(reason.split())[:160]}）" if reason else ""
        return (
            f"{head}\n\n"
            f"... [中间省略 {elided} 字符;外溢失败{detail},这部分已不可恢复] ...\n\n"
            f"{tail}"
        )

    def apply(
        self,
        result: ToolResult,
        *,
        store: SpillStore,
        session_id: str,
        hint: str = "",
    ) -> tuple[ToolResult, SpillRecord | None]:
        """按预算决定是否外溢。返回 ``(结果, 记录)``;记录为 ``None`` 表示没外溢。"""
        text = result.content or ""
        if not self.needs_spill(text):
            return result, None

        try:
            record = store.save(text, session_id=session_id, hint=hint)
        except OSError as exc:
            return (
                ToolResult(self.render_without_storage(text, str(exc)), result.is_error, result.images),
                None,
            )
        # is_error 必须原样保留:外溢只改"内容怎么给",不改"这次调用成不成功"。
        return ToolResult(self.render(text, record, hint), result.is_error, result.images), record


def plugin(
    root: Path | str | None = None,
    retention_days: int = DEFAULT_RETENTION_DAYS,
    max_inline_chars: int = DEFAULT_MAX_INLINE_CHARS,
    head_chars: int = DEFAULT_HEAD_CHARS,
    tail_chars: int = DEFAULT_TAIL_CHARS,
    max_session_files: int = DEFAULT_MAX_SESSION_FILES,
    max_session_bytes: int = DEFAULT_MAX_SESSION_BYTES,
) -> Plugin:
    """装载外溢存储 + 挂在 ``tools/post-execute`` 上的预算闸门。"""

    def apply(ctx: Context) -> None:
        store = LocalSpillStore(root, retention_days, max_session_files=max_session_files,
                                max_session_bytes=max_session_bytes)
        policy = SpillPolicy(max_inline_chars, head_chars, tail_chars)
        ctx.provide("spillStore", store)
        ctx.provide("spillPolicy", policy)

        # 启动清理:外溢文件是缓存,过保留期即回收(失败不影响启动)。
        store.cleanup_if_due()

        async def spill_oversized(call, result, context, nxt) -> Any:
            session_id = getattr(getattr(context, "session", None), "id", "") or "anonymous"
            spilled, record = policy.apply(
                result, store=store, session_id=session_id, hint=call.name
            )
            if record is not None:
                await ctx.emit("spill/saved", call, record)
            # Also forward hard truncation after failed storage; record=None
            # means either no spill was needed OR saving failed.
            return await nxt(call, spilled, context)

        ctx.effect(ctx.on("tools/post-execute", spill_oversized, mode=MODE_WATERFALL))

    return Plugin(
        name="spill",
        apply=apply,
        inject=("tools",),
        description="工具结果外溢(超预算落盘 + 返回路径)",
    )
