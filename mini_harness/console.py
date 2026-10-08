"""控制台渲染 —— 会话事件轨迹与流式增量的显示。

单次执行与 REPL 共用这一层,所以它既不认识 CLI 参数,也不认识 REPL 命令。
输出统一走一个 ``write`` 回调(默认 ``print``),这样测试可以把它换成列表收集器,
不必去抓 stdout。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from .approval import ACCESS_LABELS

__all__ = [
    "TracePrinter",
    "format_event",
    "format_status",
    "status_for",
    "human_tokens",
    "shorten_path",
    "preview",
    "indent",
]


def human_tokens(count: int | None) -> str:
    """把 token 数写成好读的形式:1234 → 1.2k。"""
    if count is None:
        return "?"
    if count < 1000:
        return str(count)
    if count < 100_000:
        return f"{count / 1000:.1f}k"
    return f"{round(count / 1000)}k"


def shorten_path(path: str | Path, limit: int = 30) -> str:
    """把路径压成一行:主目录用 ~ 缩写;太长就**保留尾部段落**,中间用 … 省略。

    为什么保留尾部:路径里有信息的是最后几段("哪个项目"),而开头多半是重复的盘符/根。
    所以从尾部往上拼,拼到快超长为止,前面补 ``E:/…/`` 这样的头 ——
    比"掐头去尾留中间"好读得多。
    """
    text = str(path).replace("\\", "/")
    home = str(Path.home()).replace("\\", "/")
    if text.lower().startswith(home.lower()):
        text = "~" + text[len(home) :]
    if len(text) <= limit:
        return text

    segments = [segment for segment in text.split("/") if segment]
    if not segments:
        return text

    kept: list[str] = []
    for segment in reversed(segments):
        candidate = "/".join([segment, *kept])
        if kept and len(candidate) + 2 > limit:
            break
        kept.insert(0, segment)

    root = "//" if text.startswith("//") else "/" if text.startswith("/") else ""
    prefix = root + segments[0]
    if len(kept) == len(segments):
        return root + "/".join(kept)  # 已经从头拼到,没什么可省的
    return f"{prefix}/…/" + "/".join(kept)


def format_status(
    model: str,
    cwd: str | Path,
    usage: Any = None,
    access: str = "",
    width: int = 100,  # noqa: ARG001 —— 预留:将来按终端宽度裁剪
) -> str:
    """一行状态:模型 / 工作目录 / 上下文窗口与用量 / 权限。

    ``usage`` 是 ``UsageSnapshot``。窗口那一项带 ``?`` 表示**这是近似值**
    (网关不返回窗口大小,见 ``token_meter.py``),别让显示假装精确。
    """
    parts = [f"模型 {model or '(未设置)'}", f"目录 {shorten_path(cwd)}"]

    if usage is not None:
        used = human_tokens(usage.used)
        window = human_tokens(usage.window)
        guess = "?" if usage.window_is_guess else ""
        alarm = " ⚠" if usage.percent >= 80 else ""
        if usage.measured is not None:
            source = "实测"
        elif abs(usage.factor - 1.0) > 0.05:
            source = f"估算×{usage.factor:.2f}"  # 已按实测校准过
        else:
            source = "估算"
        parts.append(
            f"上下文 {used}/{window}{guess} ({usage.percent:.0f}%{alarm}, {source})"
        )
        cached = getattr(usage, "cached_tokens", None)
        hit_percent = getattr(usage, "cache_hit_percent", None)
        if cached is not None:
            ratio = f" ({hit_percent:.1f}%)" if hit_percent is not None else ""
            parts.append(f"缓存命中 {human_tokens(cached)} tok{ratio}")

    if access:
        parts.append(f"权限 {ACCESS_LABELS.get(access, access)}")

    return "── " + " │ ".join(parts) + " ──"


def preview(text: str | None, limit: int = 400) -> str:
    body = (text or "").strip()
    if len(body) <= limit:
        return body or "(空)"
    return f"{body[:limit]} ... [截断,共 {len(body)} 字符]"


def indent(text: str, prefix: str = "           ") -> str:
    return text.replace("\n", "\n" + prefix)


def _seconds(ms: Any) -> str:
    return f"{ms / 1000:.1f}s" if isinstance(ms, int) else "?"


def _format_step_end(data: dict) -> str | None:
    """step 收尾那一行:这一步花了多久,以及花在哪。

    时间拆开才有用 —— "这一步 12 秒"和"12 秒里有 10 秒在退避重试"是两回事。
    ``model_ms`` 已在 agent_loop 中扣除 ``retry_ms``,两项独立显示。
    """
    duration = data.get("duration_ms")
    if not isinstance(duration, int):
        return None  # 老日志没有这个字段:宁可不显示,也不编一个数
    parts: list[str] = []
    model = data.get("model_ms")
    retry = data.get("retry_ms") or 0
    if isinstance(model, int):
        parts.append(f"模型 {_seconds(model)}")
        if retry >= 50:
            parts.append(f"退避 {_seconds(retry)}")
    tools = data.get("tools_ms")
    if isinstance(tools, int) and tools >= 50:
        parts.append(f"工具 {_seconds(tools)}")
    detail = f"({' / '.join(parts)})" if parts else ""
    mark = "(被取消)" if data.get("cancelled") else ""
    return f"  ⏱ step {data.get('index')} 用时 {_seconds(duration)}{detail}{mark}"


def format_event(event, streamed_text: bool = False) -> str | None:
    """把一条会话事件渲染成一行人类可读的轨迹。

    ``streamed_text=True`` 表示这段正文已经通过流式逐字打过了,这里就不再重复。
    """
    kind = event.type
    data = event.data

    if kind == "command/result":
        return "  [提示] " + indent(str(data.get("text") or ""))
    if kind == "turn/start":
        return "\n──────── turn 开始 ────────"
    if kind == "turn/end":
        duration = data.get("duration_ms")
        shown = f", 用时 {_seconds(duration)}" if isinstance(duration, int) else ""
        return f"──────── turn 结束({data.get('stopped')}{shown}) ────────"
    if kind == "step/end":
        return _format_step_end(data)
    if kind == "step/start":
        context = data.get("context_tokens")
        suffix = ""
        if isinstance(context, int):
            suffix = f"(上下文 ≈{human_tokens(context)} tok)"
            calibrated = data.get("context_tokens_calibrated")
            # 校准值和原始估算差得多时一起写出来 —— 预算逻辑用的是校准值,
            # 不显示的话"为什么这时候压缩了"就无从解释。
            if isinstance(calibrated, int) and abs(calibrated - context) > context * 0.02:
                suffix = (
                    f"(上下文 ≈{human_tokens(context)} tok,"
                    f"校准后 ≈{human_tokens(calibrated)})"
                )
        return f"\n[step {data.get('index')}] 组装提示与工具表 → 请求模型{suffix}"
    if kind == "assistant/message":
        parts: list[str] = []
        if data.get("tool_calls"):
            rendered = ", ".join(
                f"{call['name']}({json.dumps(call.get('arguments') or {}, ensure_ascii=False)})"
                for call in data["tool_calls"]
            )
            parts.append(f"请求工具 → {rendered}")
        if data.get("text") and not streamed_text:
            parts.append(indent(preview(data["text"])))
        if not parts:
            return None  # 正文已流式打印过,且没有工具调用
        return "  [模型] " + "\n           ".join(parts)
    if kind == "tool/result":
        flag = "失败" if data.get("is_error") else "成功"
        return (
            f"  [工具结果] {data.get('name')} {flag}: "
            + indent(preview(data.get("content")))
        )
    return None


def status_for(ctx: Any, config: Any, session=None) -> str:
    """从 context 与配置拼出状态行(单次执行与 REPL 共用)。

    读的都是**活的状态**:模型名与用量来自 ``ctx.tokenMeter``,权限档位来自
    ``ctx.approval`` —— 不是配置里那份启动快照。否则 ``/model``、``/access``
    切完之后状态行会撒谎。
    """
    from .approval import current_mode

    meter = ctx.get("tokenMeter", None)
    usage = meter.snapshot() if meter is not None else None
    # 模型名依次问:**正在用的适配器** > 用量表 > 配置。
    # 适配器那个才是真的在用的名字(离线适配器就忽略配置里的模型名),
    # 先读配置会显示一个其实没在用的模型 —— 状态行撒谎比不显示更糟。
    service = ctx.get("llm", None)
    adapter = getattr(service, "active", None) if service is not None else None
    model = (
        getattr(adapter, "model", "")
        or (usage.model if usage is not None else "")
        or getattr(config, "model", "")
    )
    cwd = getattr(config, "task_cwd", Path.cwd())
    fallback = getattr(config, "approval", "")
    permissions = ctx.get("permissions", None)
    access = permissions.current(session) if permissions is not None else current_mode(ctx, fallback, session)
    return format_status(model, cwd, usage=usage, access=access)


class TracePrinter:
    """订阅会话事件与流式帧,把 turn/step 的推进过程打到控制台。"""

    def __init__(
        self,
        write: Callable[[str], None] | None = None,
        show_trace: bool = True,
        show_stream: bool = True,
    ) -> None:
        self.write = write or (lambda text: print(text, end="", flush=True))
        self.show_trace = show_trace
        self.show_stream = show_stream
        self._streamed_text = False
        self._stream_open = False
        #: 子代理的缩进层级与观察器 —— 子代理的轨迹要看得见,但**缩进**着看,
        #: 这样"它自己一个会话、父会话只有工具调用与结果"这件事在视觉上一目了然。
        self._child_indent: dict[str, int] = {}
        self._child_streamed: dict[str, bool] = {}
        self._child_disposers: dict[str, Callable[[], None]] = {}

    # ------------------------------------------------------------ 会话事件
    def on_session_event(self, event) -> None:
        if not self.show_trace:
            return
        if event.type == "step/start":
            self._streamed_text = False
        line = format_event(event, streamed_text=self._streamed_text)
        if line:
            self.write(line + "\n")

    # ------------------------------------------------------------ 子代理轨迹
    def _indent(self, text: str, level: int) -> str:
        prefix = "  " * level
        return "\n".join(prefix + line if line.strip() else line for line in text.split("\n"))

    def on_subagent_started(self, session, task: str, depth: int) -> None:
        """子代理开跑:打个头,并把它的会话轨迹**缩进**着接进来。

        dsh 里子代理的中间过程是"留在外面"的 —— 但完全看不见又没法排查,
        所以这里保留可见性,只是明确缩进一级,视觉上区分"父"与"子"。
        """
        level = max(1, depth)
        self._child_indent[session.id] = level
        disposer = session.observe(
            lambda event, sid=session.id: self._on_child_event(event, sid)
        )
        self._child_disposers[session.id] = disposer
        if not self.show_trace:
            return
        self.write(f"{'  ' * level}⤷ 委派子代理(会话 {session.id}):{preview(task, 70)}\n")

    def on_subagent_finished(self, session, result) -> None:
        level = self._child_indent.pop(session.id, 1)
        disposer = self._child_disposers.pop(session.id, None)
        if disposer is not None:
            disposer()  # 子代理结束就摘掉观察器,别越积越多
        self._child_streamed.pop(session.id, None)
        if not self.show_trace:
            return
        status = "完成" if result.ok else f"未完成({result.stopped})"
        if getattr(result, "save_error", ""):
            status = f"结束({result.stopped}，日志保存失败)"
        self.write(f"{'  ' * level}⤶ 子代理{status}:step={result.steps}\n")

    def _on_child_event(self, event, session_id: str) -> None:
        if not self.show_trace:
            return
        line = format_event(
            event, streamed_text=self._child_streamed.get(session_id, False)
        )
        if line:
            self.write(self._indent(line, self._child_indent.get(session_id, 1)) + "\n")

    # -------------------------------------------------------- 流式增量帧
    def on_stream_frame(self, session, frame) -> None:
        """流式帧渲染。

        ``[模型] `` 这个前缀是**懒打印**的:只有真的流到正文才打。
        否则"只请求工具、没有正文"的那一步会留下一个悬空前缀,紧跟着又出现
        ``[模型] 请求工具 → …`` 那一行,读起来莫名其妙。

        子代理的流式正文要**缩进**输出,和它的轨迹对齐。
        """
        if not (self.show_trace and self.show_stream):
            return

        child = self._child_indent.get(getattr(session, "id", ""), 0)
        if child:
            self._on_child_stream(frame, session.id, child)
            return

        if frame.phase == "chunk":
            if frame.reasoning:
                return  # 思维链不混进正文
            if frame.text:
                if not self._stream_open:
                    self.write("  [模型] ")
                    self._stream_open = True
                self._streamed_text = True
                self.write(frame.text)
        elif frame.phase == "end" and self._stream_open:
            self.write("\n")
            self._stream_open = False

    def _on_child_stream(self, frame, session_id: str, level: int) -> None:
        prefix = "  " * level
        if frame.phase == "chunk":
            if frame.reasoning:
                return
            if frame.text:
                if not self._stream_open:
                    self.write(f"{prefix}[模型] ")
                    self._stream_open = True
                self._child_streamed[session_id] = True
                self.write(frame.text)
        elif frame.phase == "end" and self._stream_open:
            self.write("\n")
            self._stream_open = False

    # ---------------------------------------------------- 提前派发的工具调用
    def on_tool_dispatched(self, session, call) -> None:
        """边流边执行真的发生时报一行 —— 这是它唯一的可观测痕迹。"""
        if not self.show_trace:
            return
        arguments = json.dumps(call.arguments or {}, ensure_ascii=False)
        if len(arguments) > 120:
            arguments = arguments[:120] + " …"
        self.write(f"  [提前派发] {call.name}({arguments}) —— 流还没结束,先跑起来\n")

    # ---------------------------------------------------------------- 其他
    @property
    def streamed_last_step(self) -> bool:
        """最后一个 step 的正文是否已经逐字打过(打过了就别再重复整段)。"""
        return self._streamed_text

    def observe(self, session) -> Callable[[], None]:
        """把自己挂到会话的观察者上,返回注销器。"""
        return session.observe(self.on_session_event)

    def attach(self, ctx) -> Callable[[], None]:
        """把流式帧、提前派发、子代理三组实时事件一并挂上,返回注销器。"""
        from .kernel import MODE_EMIT

        undo_stream = ctx.on(
            "agent/assistant-stream", self.on_stream_frame, mode=MODE_EMIT
        )
        undo_dispatch = ctx.on(
            "agent/tool-dispatched", self.on_tool_dispatched, mode=MODE_EMIT
        )
        undo_started = ctx.on(
            "subagent/started", self.on_subagent_started, mode=MODE_EMIT
        )
        undo_finished = ctx.on(
            "subagent/finished", self.on_subagent_finished, mode=MODE_EMIT
        )

        def dispose() -> None:
            undo_stream()
            undo_dispatch()
            undo_started()
            undo_finished()
            for disposer in list(self._child_disposers.values()):
                disposer()
            self._child_disposers.clear()

        return dispose
