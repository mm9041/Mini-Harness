"""网页版 UI —— 同一套插件树之上的第二个前端。

dsh 的 web 端是一整套 React 生产前端(`apps/web` + `packages/client/*`),这里不照搬实现,
只对齐**它要展示的东西**:对话记录、流式正文、工具调用与结果、审批卡片、状态栏
(模型/目录/上下文/权限)、中断按钮、模型与权限切换。

关键点在于:**这一切都不需要改内核。** 网页版消费的还是那些已经存在的接缝 ——

* 会话事件(`session.observe`)→ 对话记录;
* ``agent/assistant-stream`` → 流式正文;
* ``ctx.approval.set_approver`` → **审批卡片**(和终端里那个 y/N 是同一个接缝,
  换 UI 就是换"问法" —— 这正是当初把策略与问法分开的回报);
* ``ctx.interrupt`` / ``ctx.llm.use_model`` / ``ApprovalService.set_mode`` → 停止 / 换模型 / 换权限。

技术选型上只做两件事:标准库的 ``http.server`` 起服务,**SSE** 往前端推事件
(前端一个 ``EventSource`` 收)。线程与事件循环之间的桥用 ``queue.Queue``
(它本身线程安全)和 ``run_coroutine_threadsafe``(从请求线程调度协程)——
不用任何第三方库,也就没有构建步骤。
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import concurrent.futures
import json
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, parse_qs, quote

from .approval import ACCESS_LABELS, ApprovalDecision, ApprovalRequest
from .console import status_for
from .directory_picker import choose_directory
from .history import ConversationHistory
from .user_messages import user_message_view
from .interaction import preview_type
from .providers import ProviderStore, reasoning_options
from .kernel import MODE_EMIT, Context, Plugin
from .llm import LLMError
from .session import Session, SessionEvent, repair_interrupted_tail
from .token_meter import cache_usage
from .webui_errors import UiConflictError

__all__ = ["WebUi", "ui_message", "transcript_of", "plugin", "STATIC_DIR", "DEFAULT_PORT"]

STATIC_DIR = Path(__file__).resolve().parent / "webui_static"
DEFAULT_PORT = 8770
DEFAULT_APPROVAL_TIMEOUT = 300.0
MAX_ATTACHMENT_BYTES = 10 * 1024 * 1024


def save_attachments(cwd: Path, files: Any) -> list[dict[str, Any]]:
    """Validate the whole batch before writing files into a unique workspace folder."""
    if not isinstance(files, list) or len(files) > 8:
        raise ValueError("每条消息最多上传 8 个文件")
    decoded = []
    total = 0
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("附件格式无效")
        name = item.get("name", "")
        if (not isinstance(name, str) or not name or len(name) > 180
                or name in (".", "..") or name.endswith((".", " "))
                or any(c in name for c in '/\\<>:"|?*')
                or any(ord(c) < 32 for c in name)):
            raise ValueError("附件文件名无效")
        try:
            data = base64.b64decode(item.get("data", ""), validate=True)
        except (ValueError, TypeError, binascii.Error) as exc:
            raise ValueError("附件数据无效") from exc
        total += len(data)
        if total > MAX_ATTACHMENT_BYTES:
            raise ValueError("附件总大小不能超过 10 MB")
        decoded.append((name, data))
    saved = []
    for name, data in decoded:
        folder = cwd / ".mini-harness" / "uploads" / uuid.uuid4().hex
        folder.mkdir(parents=True, exist_ok=False)
        target = folder / name
        target.write_bytes(data)
        saved.append({"name": name, "path": target.resolve().as_posix(), "size": len(data)})
    return saved


# ---------------------------------------------------------------------- 事件映射
def ui_message(event: SessionEvent) -> dict[str, Any] | None:
    """把一条**会话事件**映射成给浏览器的消息。纯函数,所以好测。

    认得出的一律给专门的 kind(前端画得像样一点);认不出的退化成 ``notice`` ——
    以后内核加了新事件类型,前端不用改也能把它显示出来(dsh 的 UI 也是这个思路)。
    """
    kind, data = event.type, event.data
    if kind == "subagent/end":
        return {"kind": "subagent-log", "child_id": data.get("child_session"),
                "title": data.get("title") or "子代理", "stopped": data.get("stopped"),
                "saved": bool(data.get("session_path"))}
    if kind == "command/result":
        return {"kind": "command-result", "text": data.get("text", "")}
    if kind == "compaction":
        before, after = data.get("tokens_before", 0), data.get("tokens_after", 0)
        return {"kind": "command-result", "text": f"上下文已压缩：历史估算 {before:,} → {after:,} tokens。原始事件保留。"}
    if kind in ("interaction/question", "interaction/answer"):
        return dict(data)
    if kind == "artifact/presented":
        return {"kind": "artifact", **data}
    if kind == "user/message":
        return {"kind": "user", **user_message_view(data)}
    if kind == "assistant/message":
        return {
            "kind": "assistant",
            "text": data.get("text") or "",
            "reasoning": data.get("reasoning") or "",
            "model": data.get("model"),
            "completed_at": event.ts,
            "tool_calls": [call.get("name") for call in data.get("tool_calls") or []],
        }
    if kind == "tool/call":
        return {
            "kind": "tool-call",
            "id": data.get("call_id"),
            "name": data.get("name"),
            "arguments": data.get("arguments") or {},
        }
    if kind == "tool/result":
        return {
            "kind": "tool-result",
            "id": data.get("call_id"),
            "name": data.get("name"),
            "content": data.get("content") or "",
            "is_error": bool(data.get("is_error")),
            "images": data.get("images") or [],
        }
    if kind == "turn/start":
        return {"kind": "turn-start", "started_at": event.ts}
    if kind == "turn/end":
        return {
            "kind": "turn-end",
            "stopped": data.get("stopped"),
            "steps": data.get("steps"),
            "step_limit": data.get("step_limit"),
            "duration_ms": data.get("duration_ms"),
            "completed_at": event.ts,
        }
    if kind == "step/start":
        return {
            "kind": "step-start",
            "index": data.get("index"),
            "started_at": event.ts,
            "model": data.get("model"),
            "context_tokens": data.get("context_tokens"),
        }
    if kind == "step/end":
        return {
            "kind": "step-end",
            "index": data.get("index"),
            "duration_ms": data.get("duration_ms"),
            "model_ms": data.get("model_ms"),
            "tools_ms": data.get("tools_ms"),
            "cancelled": bool(data.get("cancelled")),
        }
    # 其余(retry/* / spill 等)统一走 notice:前端当系统提示显示
    return {"kind": "notice", "type": kind, "data": _compact(data)}


def _compact(data: dict[str, Any], limit: int = 300) -> dict[str, Any]:
    """给 notice 用:只截长文本,**保留 JSON 类型**。

    数值和布尔别顺手转成字符串 —— 前端可能拿它们做判断(比如 ``cancelled``),
    ``"true"`` 在 JS 里是真值但不是布尔,迟早出岔子。
    """
    out: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, str):
            out[key] = value if len(value) <= limit else value[:limit] + "…"
        elif value is None or isinstance(value, (int, float, bool)):
            out[key] = value
        else:
            text = json.dumps(value, ensure_ascii=False)
            out[key] = text if len(text) <= limit else text[:limit] + "…"
    return out


def transcript_of(session: Session) -> list[dict[str, Any]]:
    """从日志重建整段对话 —— 新连上的浏览器靠它补齐历史。

    注意它是**投影**:浏览器不持有真相,刷新一下就重新从日志拉一遍。
    """
    messages: list[dict[str, Any]] = []
    for event in session.events:
        message = ui_message(event)
        if message is not None:
            messages.append(message)
    return messages


# ------------------------------------------------------------------------- 服务
@dataclass
class _Client:
    queue: "queue.Queue[dict[str, Any]]"


class ConversationUi:
    """把 harness 接到浏览器上。对应 ``ctx.webui``。"""

    def __init__(
        self,
        ctx: Context,
        config: Any = None,
        host: str = "127.0.0.1",
        port: int = DEFAULT_PORT,
        approval_timeout: float = DEFAULT_APPROVAL_TIMEOUT,
    ) -> None:
        self.ctx = ctx
        self.config = config
        self.host = host
        self.port = port
        self.approval_timeout = approval_timeout

        self.session: Session | None = None
        self.agent: Any = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._clients: list[_Client] = []
        self._pending: dict[str, asyncio.Future] = {}
        self._approval_messages: dict[str, dict[str, Any]] = {}
        self._busy = False
        self._choosing_directory = False
        self._action_lock = threading.RLock()
        self._turn_future = None
        self._turn_task = None
        self.save_path: Path | None = None
        self.autosave = False
        self.history = ConversationHistory(ctx.sessions.root)
        self.providers = ProviderStore(ctx.sessions.root.parent / "providers.json")
        self._provider_disposer = None
        self._active_stream: dict[str, Any] | None = None
        self._disposers: list[Any] = []
        self._session_disposer: Any = None  # 当前会话的观察器(换会话时撤掉重挂)
        self.port_note = ""  # 端口被占自动换端口时,写一句说明给 UI 用

    # ------------------------------------------------------------------ 生命周期
    def attach(self, session: Session | None = None) -> None:
        """挂上事件接缝、审批接缝,并建好会话。"""
        self._bind_session(session)

        # ① 流式增量 → 浏览器(瞬时,不落日志)
        self._disposers.append(
            self.ctx.on("agent/assistant-stream", self._on_stream, mode=MODE_EMIT)
        )
        # ② 审批接缝:换成"在浏览器里问"
        approval = self.ctx.get("approval", None)
        if approval is not None:
            approval.set_approver(self._ask_the_browser)
        questions = self.ctx.get("userQuestions", None)
        if questions is not None:
            questions.browser = True
            self._disposers.append(self.ctx.on("interaction/question", self._relay_question, mode=MODE_EMIT))
            self._disposers.append(self.ctx.on("artifact/presented", self._relay_artifact, mode=MODE_EMIT))

    def _bind_session(self, session: Session | None = None) -> None:
        """绑定(或换绑)会话与 agent,并把观察器接上。

        单独拆出来是因为"新建对话"要换会话 —— 而事件监听器(流式、审批)
        是绑在 ctx 上的,不随会话变,所以只有观察器需要重挂。
        """
        if self._session_disposer is not None:
            self._session_disposer()
        self.session = session or self.ctx.sessions.create()  # type: ignore[attr-defined]
        self._active_stream = None
        meter = self.ctx.get("tokenMeter", None)
        if meter is not None:
            active_model = getattr(self.ctx.llm.active, "model", "")
            meter.set_model(active_model)
            steps = self.session.events_of("step/start")
            matching_step = bool(steps and steps[-1].data.get("model") == active_model)
            meter.note_request(steps[-1].data.get("context_tokens", 0) if matching_step else 0)
            replies = self.session.events_of("assistant/message")
            if (replies and matching_step and replies[-1].data.get("model") == active_model
                    and replies[-1].seq > steps[-1].seq and replies[-1].data.get("usage")):
                meter.note_response(replies[-1].data["usage"])
        self.agent = self.ctx.agents.create(self.session)  # type: ignore[attr-defined]
        self._session_disposer = self.session.observe(
            lambda event: self._on_session_event(event)
        )

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """绑定事件循环 —— 请求线程靠它把协程调度回主循环。"""
        self.loop = loop

    def start(self) -> tuple[str, int]:
        handler = _make_handler(self)
        try:
            self._server = _QuietServer((self.host, self.port), handler)
        except OSError:
            # 端口被占(多半是又起了一个实例):换一个空闲端口,
            # 而不是让整个 harness 起不来 —— 反正真正的地址会打印出来。
            # (前提是 allow_reuse_address=False:开着它,Windows 上第二次绑定会"成功")
            self._server = _QuietServer((self.host, 0), handler)
            self.port_note = f"(默认端口 {self.port} 被占用,已自动换端口)"
        self.port = self._server.server_address[1]  # port=0 时取实际端口
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="mini-harness-web", daemon=True
        )
        self._thread.start()
        return self.host, self.port

    def stop(self) -> None:
        if self._session_disposer is not None:
            self._session_disposer()
            self._session_disposer = None
        questions = self.ctx.get("userQuestions", None)
        if questions is not None:
            questions.browser = False
            if self.loop and self.loop.is_running():
                for _, future in list(questions.pending.values()):
                    self.loop.call_soon_threadsafe(future.cancel)
        for disposer in self._disposers:
            disposer()
        self._disposers.clear()
        for client in list(self._clients):
            try:
                client.queue.put_nowait({"kind": "bye"})
            except queue.Full:
                client.queue.get_nowait()
                client.queue.put_nowait({"kind": "bye"})
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    # ------------------------------------------------------------------ 推送
    def subscribe(self) -> "queue.Queue[dict[str, Any]]":
        client = _Client(queue.Queue(maxsize=1000))
        self._clients.append(client)
        return client.queue

    def unsubscribe(self, target: "queue.Queue[dict[str, Any]]") -> None:
        self._clients = [c for c in self._clients if c.queue is not target]

    def publish(self, message: dict[str, Any]) -> None:
        """推给所有浏览器。

        ``queue.Queue`` 本身线程安全,所以事件循环线程直接 put 就行 ——
        不需要 ``call_soon_threadsafe`` 那一套(和读 SSE 时的桥正好相反)。
        满了就丢最旧的:UI 掉帧可以忍,把 agent 卡住不行。
        """
        for client in list(self._clients):
            try:
                client.queue.put_nowait(message)
            except queue.Full:
                try:
                    client.queue.get_nowait()
                    client.queue.put_nowait(message)
                except (queue.Empty, queue.Full):
                    pass

    # ------------------------------------------------------------------ 投影
    def status(self, *, schemas=None) -> dict[str, Any]:
        """轻量状态:每轮结束推这个就够(不重发整段对话)。"""
        approval = self.ctx.get("approval", None)
        permissions = self.ctx.get("permissions", None)
        permission_state = permissions.snapshot(self.session) if permissions else None
        meter = self.ctx.get("tokenMeter", None)
        usage = meter.snapshot() if meter is not None else None
        model = ""
        service = self.ctx.get("llm", None)
        if service is not None:
            model = getattr(getattr(service, "active", None), "model", "") or ""
        effort = getattr(service.active, "reasoning_effort", None) or "none" if service else "none"
        _, effort_note = reasoning_options(getattr(service.active, "provider_kind", "auto") if service else "auto", model, effort)
        # Read-only projection: do not generate a summary or mutate request metering.
        next_estimated = None
        last_input = None
        last_cached, last_cache_percent = None, None
        last_compaction = None
        if self.session is not None and usage is not None:
            compaction = self.ctx.get("compaction", None)
            messages = compaction.project(self.session) if compaction else self.session.derive_messages()
            raw = self.ctx.agentLoop._estimated_tokens(
                self.ctx.agentLoop.system_for(self.session), messages,
                schemas if schemas is not None else self.ctx.tools.schemas()
            )
            next_estimated = max(0, int(raw * usage.factor))
            # One reverse pass, stopping once the latest reply and latest
            # compaction start are known. Usage/end arrive after their start.
            reply = start = None
            reported_by_id, ended_by_id = {}, {}
            for event in reversed(self.session.events):
                if event.type == "assistant/message" and reply is None:
                    reply = event
                elif event.type == "compaction/start" and start is None:
                    start = event
                elif start is None and event.type == "compaction/usage":
                    reported_by_id.setdefault(event.data.get("operation_id"), event.data.get("usage"))
                elif start is None and event.type == "compaction/end":
                    ended_by_id.setdefault(event.data.get("operation_id"), event.data)
                if reply is not None and start is not None:
                    break
            if reply is not None and reply.data.get("model") == model:
                last_usage = reply.data.get("usage") or {}
                last_cached, last_cache_percent = cache_usage(last_usage)
                value = last_usage.get("prompt_tokens")
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    last_input = value
            if start is not None:
                operation_id = start.data["operation_id"]
                reported = reported_by_id.get(operation_id) or {}
                ended = ended_by_id.get(operation_id) or {}
                count = lambda value: value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None
                cached, ratio = cache_usage(reported)
                last_compaction = {"status": ended.get("status", "incomplete"),
                                   "input_tokens": count(reported.get("prompt_tokens")),
                                   "output_tokens": count(reported.get("completion_tokens")),
                                   "cached_tokens": cached, "cache_hit_percent": ratio}
        return {
            "kind": "status",
            "reasoning": effort,
            "reasoning_note": effort_note,
            "session": self.session.id if self.session else None,
            "busy": self._busy,
            "status": status_for(self.ctx, self.config, self.session) if self.config else "",
            "model": model or (usage.model if usage else ""),
            "cwd": str(self.ctx.agentLoop.cwd),
            "access": permission_state["preset"] if permission_state else (approval.mode_for(self.session) if approval else None),
            "permissions": permission_state,
            "permission_scope": ({"files": permission_state["sandbox"], "shell_sandbox": permission_state["sandbox"] != "danger-full-access"}
                                 if permission_state else {"files": "unrestricted" if getattr(self.config, "allow_outside", False) else "workspace-and-spill-read", "shell_sandbox": False}),
            "access_labels": permission_state["options"] if permission_state else ACCESS_LABELS,
            "context": {
                "used": usage.used if usage else 0,
                "window": usage.window if usage else 0,
                "window_is_guess": usage.window_is_guess if usage else True,
                "window_source": meter.window_source,
                "next_estimated": next_estimated,
                "last_input_measured": last_input,
                "last_cached_tokens": last_cached,
                "last_cache_hit_percent": last_cache_percent,
                "last_compaction": last_compaction,
                "percent": round(usage.percent, 1) if usage else 0.0,
                "source": (
                    "measured"
                    if usage is not None and usage.measured is not None
                    else "estimated"
                ),
                "factor": usage.factor if usage else 1.0,
            }
            if usage
            else None,
        }

    def snapshot(self) -> dict[str, Any]:
        """新连接进来时要的一份全量:状态 + 工具表 + **从日志投影出的整段对话**。"""
        schemas = self.ctx.tools.schemas()
        payload = self.status(schemas=schemas)
        payload["kind"] = "state"
        payload["tools"] = [
            schema.name for schema in schemas
        ]
        payload["transcript"] = transcript_of(self.session) if self.session else []
        payload["approvals"] = list(self._approval_messages.values())
        questions = self.ctx.get("userQuestions", None)
        payload["questions"] = questions.snapshot() if questions else []
        payload["stream"] = dict(self._active_stream) if self._active_stream else None
        payload["history"] = self.history.list(self.session.source_path if self.session else None)
        return payload

    # ------------------------------------------------------------------ 动作
    def send(self, text: str, files: Any = None) -> None:
        """跑一个 turn(在事件循环上)。请求线程调用。"""
        if self.loop is None:
            raise RuntimeError("还没绑定事件循环")
        with self._action_lock:
            if self._busy or self._choosing_directory:
                raise UiConflictError("上一轮还没跑完，或正在选择工作区")
            if text.strip().startswith("/") and files:
                raise ValueError("斜杠命令不能附带文件，请先移除附件")
            attachments = save_attachments(Path(self.ctx.agentLoop.cwd), files if files is not None else [])
            token = self.ctx.get("interrupt", None)
            if token is not None:
                token.reset()
            self._busy = True
            try:
                self._turn_future = asyncio.run_coroutine_threadsafe(self._run_turn(text, reset_interrupt=False, attachments=attachments), self.loop)
            except BaseException:
                self._busy = False
                raise

    async def _run_turn(self, text: str, *, reset_interrupt: bool = True, attachments=None) -> None:
        self._turn_task = asyncio.current_task()
        self._busy = True
        # Reset before accepting the next turn; never reset a running child turn.
        token = self.ctx.get("interrupt", None)
        if reset_interrupt and token is not None:
            token.reset()
        self.publish({"kind": "busy", "value": True})
        try:
            if not self.session.events_of("session/workspace"):
                self.session.append("session/workspace", cwd=str(self.ctx.agentLoop.cwd))
            if text.strip().startswith("/"):
                await self._run_command(text.strip())
            else:
                options = {"attachments": attachments} if attachments else {}
                await self.agent.run(text, **options)
        except LLMError as exc:
            self.publish({"kind": "error", "text": str(exc)})
        except Exception as exc:  # noqa: BLE001 —— 页面不该因为后端炸了就白屏
            self.publish({"kind": "error", "text": f"{type(exc).__name__}: {exc}"})
        finally:
            try:
                self._save_session()
            except Exception as exc:
                self.publish({"kind": "error", "text": f"会话保存失败: {exc}"})
            self._busy = False
            self.publish({"kind": "busy", "value": False})
            self.publish(self.status())
            self.publish({"kind": "history", "items": self.history.list(self.session.source_path)})
            self._turn_task = None

    def _save_session(self) -> None:
        if self.autosave and self.session is not None and self.session.events:
            saved = self.ctx.sessions.save(self.session, self.save_path)
            self.history.remember(saved)

    async def _run_command(self, command: str) -> None:
        """Local commands are UI events, never user messages sent to the model."""
        if command.split(maxsplit=1)[0] == "/permission":
            permissions = self.ctx.get("permissions", None)
            if permissions is None:
                text = "权限预设未启用。"
            else:
                parts = command.split(maxsplit=1)
                if len(parts) == 2:
                    permissions.set(parts[1], self.session)
                state = permissions.snapshot(self.session)
                text = f"权限预设：{state['preset']}；沙箱：{state['sandbox']}；审批：{state['approval']}"
            self.session.append("command/result", text=text)
            return
        if command == "/compact":
            service = self.ctx.get("compaction", None)
            if service is None:
                text = "上下文压缩已禁用，请移除 --no-compaction 后重启服务。"
            else:
                self.publish({"kind": "command-progress", "text": "正在压缩上下文…" + service.manual_retention_hint()})
                try:
                    record = await service.condense_now(self.session)
                except Exception as exc:
                    self.session.append("command/result", text=f"上下文压缩失败：{type(exc).__name__}: {exc}",
                                        code=getattr(exc, "code", "compaction-failed"))
                    return
                if record is not None:
                    return  # compaction event contains the persisted result
                text = "压缩已取消。" if self.ctx.interrupt.cancelled else "当前没有足够可压缩的旧消息，保留最近消息，未修改历史。"
        elif command == "/help":
            text = "/permission [预设] 查看或切换权限\n/compact 压缩上下文\n/status 查看上下文状态\n/tools 列出工具\n/new 新建对话并选择工作区\n/settings 打开模型设置\n/help 查看命令帮助\n压缩会额外请求模型生成摘要；其他命令不调用模型。"
        elif command == "/tools":
            text = "可用工具：\n" + "\n".join(f"{tool.name}：{tool.description}" for tool in self.ctx.tools.schemas())
        elif command == "/status":
            status = self.status()
            c = status.get("context") or {}
            source = {"provider": "服务商报告", "configured": "手动配置", "estimated": "估算"}.get(c.get("window_source"), "未知")
            text = (f"模型：{status['model']}\n工作区：{status['cwd']}\n"
                    f"窗口：{c.get('window', 0):,} tokens（{source}）\n"
                    f"下一次输入估算：{c.get('next_estimated') or 0:,} tokens\n"
                    f"上次输入实测：{c.get('last_input_measured') if c.get('last_input_measured') is not None else '暂无'}\n"
                    "自动压缩默认按窗口 80% 触发（可配置），输出预留、headroom 和额外历史上限可提前触发；输入估算不含草稿、待上传附件或输出预留。")
        else:
            text = "未知命令。输入 /help 查看支持的命令；/new 和 /settings 请在网页输入框使用。"
        self.session.append("command/result", text=text)

    async def shutdown(self) -> None:
        self.interrupt()
        if self._turn_future is not None:
            self._turn_future.cancel()
        tasks = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if self.session is not None:
            repair_interrupted_tail(self.session)
        self._save_session()

    def interrupt(self) -> bool:
        token = self.ctx.get("interrupt", None)
        if token is None or not self._busy:
            return False
        return bool(token.request("浏览器里点了停止"))

    def provider_settings(self) -> dict[str, Any]:
        adapter = self.ctx.llm.active
        return {**self.providers.public(), "current": {
            "name": "当前连接", "base_url": getattr(adapter, "base_url", ""),
            "model": getattr(adapter, "model", ""), "protocol": getattr(adapter, "provider_kind", "auto"),
            "has_key": bool(getattr(adapter, "api_key", "")),
        }}

    def _install_provider(self, profile: dict, key: str) -> None:
        from .adapters.openai_compat import OpenAICompatAdapter
        adapter = OpenAICompatAdapter(key, profile["base_url"], profile["model"], getattr(self.config, "timeout", 120))
        adapter.provider_kind = profile["protocol"]
        adapter.reasoning_effort = self.providers.data["reasoning"]
        if self._provider_disposer is not None:
            self._provider_disposer()
        name = "user-configured"
        self._provider_disposer = self.ctx.llm.register_adapter(name, adapter)
        self.ctx.llm.use(name)
        self.ctx.tokenMeter.clear_capacities()
        self.ctx.tokenMeter.set_model(profile["model"])
        if self.config is not None:
            self.config.model = profile["model"]
            self.config.base_url = profile["base_url"]
        if self.loop is not None:
            asyncio.run_coroutine_threadsafe(self.ctx.emit("llm/model-changed", profile["model"], ""), self.loop)

    def configure_provider(self, payload: dict) -> dict[str, Any]:
        with self._action_lock:
            if self._busy:
                raise UiConflictError("请先停止当前任务再修改厂商")
            if payload.get("id") == "environment" and not payload.get("api_key"):
                adapter = self.ctx.llm.active
                if str(payload.get("base_url", "")).rstrip("/") == getattr(adapter, "base_url", "").rstrip("/"):
                    payload = {**payload, "api_key": getattr(adapter, "api_key", "")}
            profile, key = self.providers.prepare(payload)
            self.providers.save(profile)
            self._install_provider(profile, key)
            self.publish(self.status())
            return {"ok": True, **self.provider_settings()}

    def switch_reasoning(self, effort: str) -> dict[str, Any]:
        with self._action_lock:
            if self._busy:
                raise UiConflictError("请先停止当前任务再修改推理强度")
            self.providers.set_effort(effort)
            self.ctx.llm.active.reasoning_effort = effort
            self.publish(self.status())
            return {"ok": True}

    def _in_bound_loop(self) -> bool:
        try:
            return asyncio.get_running_loop() is self.loop
        except RuntimeError:
            return False

    def _call_async(self, factory, *, timeout=30):
        """HTTP-thread bridge; create the coroutine only after checking ownership."""
        if self.loop is None or not self.loop.is_running():
            raise RuntimeError("事件循环尚未运行")
        if self._in_bound_loop():
            raise RuntimeError("不能在事件循环线程调用同步接口，请使用对应的 async 接口")
        future = asyncio.run_coroutine_threadsafe(factory(), self.loop)
        try:
            return future.result(timeout=timeout)
        except (TimeoutError, concurrent.futures.TimeoutError):
            future.cancel()
            raise

    async def switch_model_async(self, name: str) -> str:
        if not self._in_bound_loop():
            raise RuntimeError("请在绑定的事件循环中调用异步接口")
        if self._busy:
            raise UiConflictError("正在运行，结束后再切换模型")
        previous = await self.ctx.llm.use_model(name)
        self.providers.set_model(name)
        return previous

    def switch_model(self, name: str) -> str:
        return self._call_async(lambda: self.switch_model_async(name))

    def switch_access(self, mode: str) -> str:
        service = self.ctx.get("approval", None)
        if service is None:
            raise RuntimeError("审批服务没装载")
        permissions = self.ctx.get("permissions", None)
        def change():
            if permissions is not None:
                preset = {"ask": "workspace-write", "allow": "danger-full-access", "deny": "read-only"}.get(mode, mode)
                return permissions.set(preset, self.session)
            return service.set_mode(mode, self.session)
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        if self.loop is None or not self.loop.is_running() or current_loop is self.loop:
            previous = change()
            self._save_session()
            return previous
        async def apply():
            previous = change()
            self._save_session()
            return previous
        return asyncio.run_coroutine_threadsafe(apply(), self.loop).result(timeout=30)

    def pick_cwd(self, *, preview: bool = False) -> dict[str, Any]:
        with self._action_lock:
            if (self._busy and not preview) or self._choosing_directory:
                raise UiConflictError("正在运行或已有目录选择窗口，请先完成当前操作")
            if not preview and self.session and self.session.events_of("user/message"):
                raise UiConflictError("已有对话不能更换工作区，请新建对话")
            self._choosing_directory = True
            initial = str(self.ctx.agentLoop.cwd)
        try:
            selected = choose_directory(initial)
            if not selected:
                return {"ok": True, "cancelled": True}
            with self._action_lock:
                if preview:
                    return {"ok": True, "cancelled": False, "cwd": str(Path(selected).resolve())}
                self.set_cwd(selected)
                snapshot = self.status()
                self.publish(snapshot)
                return {"ok": True, "cancelled": False, "cwd": str(self.ctx.agentLoop.cwd)}
        finally:
            self._choosing_directory = False

    def set_cwd(self, path: str, *, record: bool = True) -> str:
        """换工作目录:shell 与文件工具都在新目录下执行。

        只需要改 ``agentLoop.cwd`` 一个地方 —— 工具的 cwd 是从它传下去的,
        文件工具的工作区边界也跟着走。但**不碰**外溢目录(那是缓存,和 cwd 无关)。
        """
        if self._busy:
            raise UiConflictError("正在运行，结束后再切换目录")
        if record and self.session and self.session.events_of("user/message"):
            raise UiConflictError("已有对话不能更换工作区，请新建对话")
        target = Path(path).expanduser().resolve()
        if not target.is_dir():
            raise ValueError(f"目录不存在: {target}")
        loop_service = self.ctx.get("agentLoop", None)
        if loop_service is None:
            raise RuntimeError("agentLoop 没装载")
        previous = str(loop_service.cwd)
        loop_service.cwd = target
        if self.config is not None:
            self.config.task_cwd = target  # 让状态行的显示同步
        if self.loop is not None:
            asyncio.run_coroutine_threadsafe(self.ctx.emit("workspace/changed"), self.loop)
        if record and self.session is not None:
            self.session.append("session/workspace", cwd=str(target))
        return previous

    def open_workspace(self) -> dict[str, Any]:
        """Open only the server's active workspace, never a client-supplied path."""
        target = Path(self.ctx.agentLoop.cwd).resolve()
        if not target.is_dir():
            raise ValueError(f"目录不存在: {target}")
        if os.name == "nt":
            os.startfile(str(target))
        else:
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(target)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return {"ok": True}

    def resolve_approval(self, approval_id: str, approved: bool, note: str = "") -> bool:
        """浏览器点了允许/拒绝。请求线程调用 → 用 call_soon_threadsafe 唤醒协程。"""
        if type(approved) is not bool:
            raise ValueError("approved 必须是布尔值")
        future = self._pending.pop(approval_id, None)
        if future is None or self.loop is None:
            return False
        decision = ApprovalDecision(approved, note or ("浏览器批准" if approved else "浏览器拒绝"))
        self.loop.call_soon_threadsafe(_set_result, future, decision)
        return True

    def resolve_question(self, question_id: str, answer: str, skipped=False) -> bool:
        if self.loop is None or not self.loop.is_running():
            return False
        if self._in_bound_loop():
            return self.ctx.userQuestions.answer(question_id, answer, skipped)
        async def resolve():
            return self.ctx.userQuestions.answer(question_id, answer, skipped)
        return self._call_async(resolve, timeout=10)

    def artifact(self, artifact_id: str) -> tuple[dict, Path]:
        if self.session is None:
            raise ValueError("当前没有会话")
        record = next((e.data for e in self.session.events_of("artifact/presented") if e.data.get("id") == artifact_id), None)
        if record is None:
            raise ValueError("找不到当前对话的交付物")
        path, root = Path(record["path"]), Path(record["root"]).resolve()
        resolved = path.resolve()
        if resolved != path or root not in resolved.parents or not resolved.is_file():
            raise ValueError("交付文件已移动、删除或超出允许目录")
        return record, resolved

    def _relay_question(self, session, message):
        if self.session is not None and session is not self.session:
            self.session.append("interaction/question" if message["kind"] == "question" else "interaction/answer", **message)

    def _relay_artifact(self, session, artifact):
        if self.session is not None and session is not self.session:
            self.session.append("artifact/presented", **artifact)

    async def list_models_async(self) -> list[str]:
        if not self._in_bound_loop():
            raise RuntimeError("请在绑定的事件循环中调用异步接口")
        return await self.ctx.llm.list_models()

    def list_models(self) -> list[str]:
        return self._call_async(self.list_models_async)

    # ------------------------------------------------------------------ 内部
    def _on_session_event(self, event: SessionEvent) -> None:
        if event.type in {"assistant/message", "turn/end"}:
            self._active_stream = None
        # Deliberate durability tradeoff: each checkpoint serializes and compares
        # the entire log (O(history)), even if save() ultimately skips the write.
        # Keep committed messages recoverable after a crash; do not debounce these
        # saves without explicitly deciding which recovery checkpoints may be lost.
        # Streaming deltas are not checkpoints.
        if event.type in {"user/message", "assistant/message", "tool/result", "turn/end", "session/workspace", "artifact/presented", "interaction/question", "interaction/answer"}:
            try:
                self._save_session()
            except OSError as exc:
                self.publish({"kind": "error", "text": f"会话保存失败: {exc}"})
        message = ui_message(event)
        if message is not None:
            self.publish(message)
        if event.type in {"step/start", "step/end"}:
            self.publish(self.status())

    async def _on_stream(self, session: Session, frame: Any) -> None:
        if session is not self.session:
            return
        if frame.phase == "start":
            self._active_stream = {"text": "", "reasoning": "", "started_at": time.time(), "ended": False, "model": getattr(self.ctx.llm.active, "model", None)}
            self.publish({"kind": "stream-start", "started_at": self._active_stream["started_at"], "model": self._active_stream["model"]})
        elif frame.phase == "chunk":
            if self._active_stream is not None:
                self._active_stream["text"] += frame.text or ""
                self._active_stream["reasoning"] += frame.reasoning or ""
            self.publish({"kind": "delta", "text": frame.text or "", "reasoning": frame.reasoning or ""})
        elif frame.phase == "end":
            if self._active_stream is not None:
                self._active_stream["ended"] = True
            self.publish({"kind": "delta-end"})

    async def _ask_the_browser(self, request: ApprovalRequest) -> ApprovalDecision:
        """审批人:把问题推到浏览器,等它回答。

        这就是"策略在插件、问法在 UI"的兑现 —— 终端版问 y/N,网页版发卡片,
        审批策略一个字都没改。
        """
        if self.loop is None:
            return ApprovalDecision(False, "网页版还没准备好")
        approval_id = uuid.uuid4().hex
        future: asyncio.Future = self.loop.create_future()
        self._pending[approval_id] = future
        message = {
                "kind": "approval",
                "id": approval_id,
                "tool": request.call.name,
                "arguments": request.call.arguments,
                "reason": request.reason,
                "timeout": self.approval_timeout,
            }
        self._approval_messages[approval_id] = message
        self.publish(message)
        try:
            token = self.ctx.get("interrupt", None)
            cancel = asyncio.create_task(token.wait()) if token is not None else None
            try:
                done, _ = await asyncio.wait(
                    {future, cancel} if cancel else {future},
                    timeout=self.approval_timeout, return_when=asyncio.FIRST_COMPLETED,
                )
                if cancel is not None and cancel in done:
                    return ApprovalDecision(False, "用户已停止")
                if future in done:
                    return future.result()
                raise asyncio.TimeoutError
            finally:
                if cancel is not None:
                    cancel.cancel()
                    await asyncio.gather(cancel, return_exceptions=True)
                if not future.done():
                    future.cancel()
        except (asyncio.TimeoutError, TimeoutError):
            self._pending.pop(approval_id, None)
            self.publish({"kind": "approval-closed", "id": approval_id, "expired": True})
            return ApprovalDecision(False, "浏览器没有在时限内回应")
        finally:
            self._pending.pop(approval_id, None)
            self._approval_messages.pop(approval_id, None)
            self.publish({"kind": "approval-closed", "id": approval_id})


def _set_result(future: asyncio.Future, value: Any) -> None:
    if not future.done():
        future.set_result(value)


from .webui_sessions import WebUi


# ---------------------------------------------------------------------- HTTP
class _QuietServer(ThreadingHTTPServer):
    """关掉 ``SO_REUSEADDR`` 的本地服务。

    为什么必须关:在 Windows 上 ``SO_REUSEADDR`` 的语义是**允许绑定一个正在使用的端口**
    (和 POSIX 的"只允许 TIME_WAIT 复用"完全不同)。结果是两个实例都能绑定成功、
    谁也不报错,请求却进了先启动的那个 —— 整个测试文件一起"神秘失败"(实测踩过)。
    显式关掉之后,端口被占就会抛 ``OSError``,``start()`` 才能按计划换空闲端口。
    """

    allow_reuse_address = False
    daemon_threads = True


def _make_handler(ui: WebUi) -> type[BaseHTTPRequestHandler]:
    """给这个 UI 实例造一个 handler 类(handler 是按连接实例化的,得靠闭包拿到 ui)。"""

    class Handler(BaseHTTPRequestHandler):
        server_version = "mini-harness-web"
        protocol_version = "HTTP/1.0"  # 关连接表示响应结束,SSE 正好靠这个

        # ------------------------------------------------------------ 工具
        def log_message(self, *args: Any) -> None:  # noqa: ARG002
            pass  # 别把访问日志刷到终端上

        def _json(self, payload: Any, status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _error(self, message: str, status: int = 400) -> None:
            self._json({"error": message}, status=status)

        def _body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if not length:
                return {}
            try:
                return json.loads(self.rfile.read(length).decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                return {}

        # ------------------------------------------------------------ 路由
        def _trusted_request(self) -> bool:
            host = self.headers.get("Host", "")
            allowed = set() if ui.host == "0.0.0.0" else {f"{ui.host}:{ui.port}"}
            if ui.host in ("127.0.0.1", "localhost", "0.0.0.0"):
                allowed.update({f"localhost:{ui.port}", f"127.0.0.1:{ui.port}"})
            if ui.host == "0.0.0.0":
                # Trust the accepted socket's destination, never an arbitrary Host
                # or a DNS lookup of client-controlled input.
                address = self.connection.getsockname()[0]
                allowed.add(f"{address}:{ui.port}")
            if len(self.headers.get_all("Host", [])) != 1:
                return False
            if len(self.headers.get_all("Origin", [])) > 1:
                return False
            if host not in allowed:
                return False
            origin = self.headers.get("Origin")
            return (origin is None or origin == f"http://{host}") and self.headers.get("Sec-Fetch-Site") != "cross-site"

        def do_GET(self) -> None:  # noqa: N802
            if not self._trusted_request():
                return self._error("请求来源不受信任", status=403)
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                return self._static("index.html")
            if path.startswith("/api/artifacts/"):
                return self._artifact(path.rsplit("/", 1)[-1])
            if path == "/api/history":
                return self._json({"items": ui.history_items()})
            if path == "/api/history/archives":
                return self._json(ui.archived_history())
            if path == "/api/state":
                return self._json(ui.snapshot())
            if path == "/api/events":
                return self._sse()
            if path == "/api/providers":
                return self._json(ui.provider_settings())
            if path == "/api/models":
                try:
                    return self._json({"models": ui.list_models()})
                except Exception as exc:  # noqa: BLE001 —— 网关不支持就算了
                    return self._json({"models": [], "error": str(exc)})
            return self._error("没有这个路径", status=404)

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if not self._trusted_request():
                return self._error("请求来源不受信任", status=403)
            if self.headers.get_content_type() != "application/json":
                return self._error("请求必须使用 application/json", status=415)
            try:
                length = int(self.headers.get("Content-Length") or 0)
                limit = 15_000_000 if path == "/api/message" else 1_000_000
                if length < 0 or length > limit:
                    return self._error("请求体过大", status=413)
                body = self._body()
                if not isinstance(body, dict):
                    return self._error("请求体必须是 JSON 对象")
                worker = ui.target(body.get("session"))
                if path == "/api/message":
                    text = str(body.get("text") or "").strip()
                    if not text and not body.get("files"):
                        return self._error("text 不能为空")
                    worker.send(text, body.get("files", []))
                    return self._json({"ok": True})
                if path == "/api/interrupt":
                    return self._json({"ok": worker.interrupt()})
                if path == "/api/approval":
                    approved = body.get("approved")
                    if type(approved) is not bool:
                        return self._error("approved 必须是布尔值")
                    ok = worker.resolve_approval(
                        str(body.get("id") or ""),
                        approved,
                        str(body.get("note") or ""),
                    )
                    return self._json({"ok": ok})
                if path == "/api/question":
                    skipped = body.get("skipped", False)
                    if not isinstance(skipped, bool):
                        return self._error("skipped 必须是布尔值")
                    ok = worker.resolve_question(str(body.get("id") or ""), body.get("answer", ""), skipped)
                    if not ok:
                        return self._error("问题已回答或已关闭", status=409)
                    return self._json({"ok": True})
                if path == "/api/providers":
                    return self._json(worker.configure_provider(body))
                if path == "/api/reasoning":
                    return self._json(worker.switch_reasoning(str(body.get("effort") or "none")))
                if path == "/api/model":
                    name = str(body.get("name") or "").strip()
                    if not name.startswith("/") and " " not in name and name:
                        previous = worker.switch_model(name)
                        worker.publish(worker.status())
                        return self._json({"ok": True, "previous": previous, "model": name})
                    return self._error("模型名不合法")
                if path in ("/api/access", "/api/permission"):
                    mode = str(body.get("mode") or "").strip()
                    previous = worker.switch_access(mode)
                    return self._json({"ok": True, "previous": previous, "mode": mode})
                if path == "/api/history/delete":
                    return self._json(ui.delete_history(str(body.get("id") or "")))
                if path == "/api/history/restore":
                    return self._json(ui.restore_history(str(body.get("token") or "")))
                if path == "/api/history/purge":
                    return self._json(ui.purge_history(str(body.get("token") or "")))
                if path == "/api/history/open":
                    return self._json(ui.open_history(str(body.get("id") or "")))
                if path == "/api/subagent/log":
                    return self._json(ui.subagent_log(str(body.get("child_id") or ""), body.get("session")))
                if path == "/api/new-session":
                    return self._json(ui.new_session(str(body.get("cwd") or "") or None))
                if path == "/api/cwd/pick":
                    return self._json(worker.pick_cwd(preview=body.get("preview") is True))
                if path == "/api/cwd/open":
                    return self._json(worker.open_workspace())
                if path == "/api/cwd":
                    target = str(body.get("path") or "").strip()
                    if not target:
                        return self._error("path 不能为空")
                    with worker._action_lock:
                        if worker._choosing_directory:
                            raise UiConflictError("请先关闭目录选择窗口")
                        previous = worker.set_cwd(target)
                    return self._json({"ok": True, "previous": previous, "cwd": target})
            except UiConflictError as exc:
                return self._error(str(exc), status=409)
            except ValueError as exc:
                return self._error(str(exc), status=400)
            except Exception as exc:  # noqa: BLE001
                return self._error(f"{type(exc).__name__}: {exc}", status=500)
            return self._error("没有这个路径", status=404)

        # ------------------------------------------------------------ 实现
        def _artifact(self, artifact_id: str) -> None:
            try:
                session_id = parse_qs(urlparse(self.path).query).get("session", [None])[0]
                record, path = ui.target(session_id).artifact(artifact_id)
                kind, mime = preview_type(path)
                download = parse_qs(urlparse(self.path).query).get("download") == ["1"] or kind == "download"
                # Read text previews as data. HTML/SVG never execute in the app origin.
                with path.open("rb") as stream:
                    if not download and kind == "text":
                        raw = stream.read(1024 * 1024 + 1)
                        clipped = len(raw) > 1024 * 1024
                        body = raw[:1024 * 1024].decode("utf-8-sig", errors="replace")
                        if clipped:
                            body += "\n\n[预览已截断，请下载完整文件]"
                        body = body.encode("utf-8")
                        size = len(body)
                    else:
                        body = None
                        size = path.stat().st_size
                    self.send_response(200)
                    self.send_header("Content-Type", "application/octet-stream" if download else mime)
                    self.send_header("Content-Length", str(size))
                    self.send_header("Content-Disposition", ("attachment" if download else "inline") + "; filename*=UTF-8''" + quote(record["name"]))
                    self.send_header("X-Content-Type-Options", "nosniff")
                    # Native PDF viewers need their own document context; textual
                    # HTML/SVG previews remain sandboxed and are served as text/plain.
                    self.send_header("Content-Security-Policy", "default-src 'none'; frame-ancestors 'self'" if kind == "pdf" else "sandbox; default-src 'none'; style-src 'unsafe-inline'")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    if body is not None:
                        self.wfile.write(body)
                    else:
                        remaining = size
                        while remaining:
                            chunk = stream.read(min(65536, remaining))
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                            remaining -= len(chunk)
            except (ValueError, OSError) as exc:
                return self._error(str(exc), status=404)

        def _static(self, name: str) -> None:
            target = (STATIC_DIR / name).resolve()
            if not target.is_file() or STATIC_DIR not in target.parents:
                return self._error("找不到静态文件", status=404)
            body = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _sse(self) -> None:
            async def capture():
                # Snapshot and subscription share an event-loop turn so queued deltas
                # cannot already be included in the initial snapshot.
                with ui._action_lock:
                    initial = ui.snapshot()
                    return ui.subscribe(), initial
            if ui.loop is not None and ui.loop.is_running():
                stream, initial = asyncio.run_coroutine_threadsafe(capture(), ui.loop).result(timeout=10)
            else:
                stream, initial = ui.subscribe(), ui.snapshot()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(b": connected\n\n")
            self.wfile.flush()
            try:
                self._emit(initial)
                while True:
                    try:
                        item = stream.get(timeout=15)
                    except queue.Empty:
                        self.wfile.write(b": ping\n\n")  # 心跳:别让中间层把连接掐了
                        self.wfile.flush()
                        continue
                    if item.get("kind") == "bye":
                        break
                    self._emit(item)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass  # 浏览器关了页面,正常
            finally:
                ui.unsubscribe(stream)

        def _emit(self, payload: dict[str, Any]) -> None:
            data = json.dumps(payload, ensure_ascii=False)
            self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
            self.wfile.flush()

    return Handler


def plugin(
    host: str = "127.0.0.1",
    port: int = DEFAULT_PORT,
    approval_timeout: float = DEFAULT_APPROVAL_TIMEOUT,
    config: Any = None,
) -> Plugin:
    """装载网页版 UI(只提供 ``ctx.webui``,不起服务 —— 起服务由 CLI 决定)。

    为什么不在 ``apply`` 里起服务:那样挂载就会有副作用,测试和单次执行都会误起端口。
    ``serve()`` 单独暴露出来,谁需要谁调用。
    """

    def apply(ctx: Context) -> None:
        ui = WebUi(ctx, config, host, port, approval_timeout)
        ctx.provide("webui", ui)
        ctx.effect(ui.stop)

    return Plugin(
        name="webui",
        apply=apply,
        inject=("tools", "sessions", "agents"),
        description="网页版 UI(标准库 HTTP + SSE)",
    )


def serve(
    ctx: Context,
    config: Any,
    host: str = "127.0.0.1",
    port: int = DEFAULT_PORT,
    loop: asyncio.AbstractEventLoop | None = None,
    session: Session | None = None,
    save_path: str | Path | None = None,
) -> WebUi:
    """起服务并挂好会话 —— CLI 的 ``--web`` 走这条路。

    ``loop`` 必须显式传:HTTP 请求线程靠它把协程调度回主循环。不传的话
    ``get_event_loop()`` 在没有运行中的循环时会新建一个,结果就是"起了两个循环、
    消息调度到了没人跑的那个上" —— 页面看着连上了却永远不动。
    """
    ui: WebUi = ctx.get("webui")
    ui.config = config
    ui.host, ui.port = host, port
    ui.attach(session)
    ui.autosave = True
    ui.save_path = Path(save_path).expanduser() if save_path else None
    ui.bind_loop(loop if loop is not None else asyncio.get_event_loop())
    saved_provider = ui.providers.active()
    if saved_provider:
        ui._install_provider(*saved_provider)
    else:
        ui.ctx.llm.active.reasoning_effort = ui.providers.data["reasoning"]
    ui.start()
    return ui
