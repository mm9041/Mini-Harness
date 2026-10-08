"""``ctx.llm`` —— 消息与工具调用的词汇,外加模型适配器的接缝。

对应 dsh 的 ``packages/llm/llm``。这里能看清"能力接缝(seam)"的三个角色:

  * **Service Definition** —— ``LLMService``(本文件):声明接口;
  * **Service Provider**   —— ``adapters/`` 下的适配器:实现接口;
  * **Consumer**           —— ``agent_loop.py``:使用接口。

换掉 provider 就换掉了整个产品的模型来源,而 consumer 一行都不用改。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Protocol

from .kernel import Context, Plugin

__all__ = [
    "ToolCall",
    "Message",
    "ToolSchema",
    "GenerateRequest",
    "GenerateResult",
    "StreamEvent",
    "LLMAdapter",
    "LLMService",
    "LLMError",
    "plugin",
]


class LLMError(RuntimeError):
    """模型调用失败(网络、鉴权、协议、超时)。

    **错误分类靠结构化字段,不靠正则匹配消息** —— 这是重试策略能不能写对的前提:

    * ``status``:HTTP 状态码(拿得到时)。429/5xx 值得重试,4xx 基本不值得;
    * ``retryable``:显式覆盖(比如连接被重置 —— 没有状态码,但明确可重试);
    * ``kind``:语义分类,目前只有 ``"overflow"``(上下文超了)。这类该**先压缩再试**,
      而不是把同一个请求原样重发一遍。
    """

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        retryable: bool | None = None,
        kind: str | None = None,
        provider_code: str | None = None,
        provider_type: str | None = None,
        provider_message: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.retryable = retryable
        self.kind = kind
        self.provider_code = provider_code
        self.provider_type = provider_type
        self.provider_message = provider_message

    def __str__(self) -> str:
        parts: list[str] = []
        if self.status is not None:
            parts.append(f"status={self.status}")
        if self.kind:
            parts.append(f"kind={self.kind}")
        suffix = f" [{', '.join(parts)}]" if parts else ""
        return f"{super().__str__()}{suffix}"


# ---------------------------------------------------------------------- 词汇


@dataclass
class ToolCall:
    """模型请求的一次工具调用。``arguments`` 是已解析的参数字典。"""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    raw_arguments: str | None = None

    def to_wire(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.raw_arguments
                if self.raw_arguments is not None
                else _dumps(self.arguments),
            },
        }


@dataclass
class Message:
    """一条模型可见的消息。``role`` 取 system / user / assistant / tool。

    ``source_seq`` 是它来自哪条会话事件(投影时填)。它**不进 wire** ——
    只用来让压缩知道"这条消息由哪些日志事件变来",从而把压缩边界对齐到事件序号上。
    没有它,压缩就只能按消息下标记边界,一旦投影规则变了边界就会错位。
    """

    role: str
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    source_seq: int | None = None
    reasoning: str | None = None  # Only adapters that require tool-turn reasoning serialize this.
    images: list[dict[str, Any]] = field(default_factory=list)

    def to_wire(self) -> dict[str, Any]:
        if self.role == "tool":
            return {
                "role": "tool",
                "tool_call_id": self.tool_call_id or "",
                "content": self.content or "",
            }
        out: dict[str, Any] = {"role": self.role, "content": self.content or ""}
        if self.images:
            out["content"] = [{"type": "text", "text": self.content or ""}] + [
                {"type": "image_url", "image_url": {"url": item["data_url"]}} for item in self.images
            ]
        if self.tool_calls:
            out["tool_calls"] = [call.to_wire() for call in self.tool_calls]
        return out


@dataclass
class ToolSchema:
    """面向模型的工具声明,直接进请求体的 ``tools`` 字段。"""

    name: str
    description: str
    parameters: dict[str, Any]

    def to_wire(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class GenerateRequest:
    """一次模型请求的全部输入。工具 schema 走协议字段,不塞进 system 提示。"""

    system: str
    messages: list[Message] = field(default_factory=list)
    tools: list[ToolSchema] = field(default_factory=list)
    model: str | None = None


@dataclass
class GenerateResult:
    """一次模型响应。``text`` 与 ``tool_calls`` 可能同时存在,也可能都为空。

    ``reasoning`` 装的是思考型模型的思维链(``reasoning_content``)，随事件保留。
    支持该字段的适配器在思考模式中回放 assistant 工具消息的 reasoning；
    普通正文回复不额外回放该字段。没有正文时，驱动器也会用它兜底展示。
    """

    text: str | None = None
    reasoning: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None


@dataclass
class StreamEvent:
    """流式过程里的一帧。

    三种 ``kind``:

    * ``"delta"`` —— 一小段增量(``text`` 是正文,``reasoning`` 是思维链);
    * ``"tool_call"`` —— **某个工具调用的参数已经补全,可以立刻执行了**
      (``tool_call`` 字段)。这是"边流边执行"的触发点:模型还在接着说,
      工具可以先跑起来;
    * ``"done"`` —— 流结束,``result`` 是拼装好的完整结果(包含全部工具调用,
      无论是否已经提前派发过)。

    为什么不让增量事件直接携带"半成品 result":因为工具调用(``tool_calls``)在流里
    是**碎片**送达的(名字一次、参数分多次),必须累加完才能解析。所以约定是
    "delta 只报增量,tool_call 报可执行,done 才给完整结果"。
    """

    kind: str  # delta | tool_call | done
    text: str = ""
    reasoning: str = ""
    tool_call: ToolCall | None = None
    result: GenerateResult | None = None


# ------------------------------------------------------------------- 适配器协议


class LLMAdapter(Protocol):
    """Service Provider 必须实现的样子。

    ``generate`` 必需;``stream`` 可选 —— 没实现时 ``LLMService.stream`` 会退化成
    "一次性生成,再拆成 delta 序列",所以**消费方永远只需要按流式写一遍代码**。
    """

    name: str

    async def generate(self, request: GenerateRequest) -> GenerateResult: ...


# ---------------------------------------------------------------------- 服务


class LLMService:
    """适配器注册表 + 请求入口。对应 ``ctx.llm``。"""

    def __init__(self, ctx: Context) -> None:
        self.ctx = ctx
        self._adapters: dict[str, LLMAdapter] = {}
        self._active: str | None = None

    # -- provider 侧 -------------------------------------------------
    def register_adapter(self, name: str, adapter: LLMAdapter) -> Callable[[], None]:
        """注册一个 provider,返回注销器(注册即效果)。"""
        self._adapters[name] = adapter
        if self._active is None:
            self._active = name

        def dispose() -> None:
            if self._adapters.get(name) is adapter:
                self._adapters.pop(name, None)
                if self._active == name:
                    self._active = next(iter(self._adapters), None)

        return dispose

    def use(self, name: str) -> None:
        if name not in self._adapters:
            raise LLMError(
                f"适配器 {name!r} 未注册;已注册: {sorted(self._adapters)}"
            )
        self._active = name

    @property
    def active(self) -> LLMAdapter:
        if self._active is None or self._active not in self._adapters:
            raise LLMError("没有可用的模型适配器,请先装载一个 provider 插件")
        return self._adapters[self._active]

    # -- consumer 侧 -------------------------------------------------
    async def generate(self, request: GenerateRequest) -> GenerateResult:
        """发一次请求。

        请求先经过 ``llm/request`` 瀑布事件 —— 插件可以据此改写提示、注入上下文、
        换模型,或直接短路换成一个假响应(测试里就这么干)。这正是 dsh 里
        ``llm/stream`` 那个瀑布接缝的迷你版。
        """
        request = await self.ctx.waterfall("llm/request", request, default=request)
        if not isinstance(request, GenerateRequest):
            raise LLMError("llm/request 监听器必须返回一个 GenerateRequest")
        result = await self.active.generate(request)
        await self.ctx.emit("llm/result", request, result)
        return result

    async def use_model(self, name: str) -> str:
        """运行时切换模型,返回旧模型名。

        模型名是**连接的属性**,所以改的是 adapter —— 请求侧不再钉死模型
        (驱动器传 ``model=None``),换完下一个 step 就生效。

        换成功会广播 ``llm/model-changed``:系统提示里的"模型:"那一行、用量表里的模型名
        都会跟着刷新。不广播的话提示词里那句话立刻变成假话,而模型会照着假话自我介绍。
        """
        adapter = self.active
        setter = getattr(adapter, "set_model", None)
        if setter is None:
            raise LLMError(f"适配器 {adapter.name} 不支持运行时切换模型")
        previous = setter(name)
        await self.ctx.emit("llm/model-changed", name, previous)
        return previous

    async def list_models(self) -> list[str]:
        """问当前适配器要一份可用模型列表(``/model`` 不带参数时用)。

        适配器不提供就抛 ``LLMError`` —— 由 UI 兜住,退化成"只显示当前模型"。
        """
        lister = getattr(self.active, "list_models", None)
        if lister is None:
            raise LLMError(f"适配器 {self.active.name} 不提供模型列表")
        models = await lister()
        await self.ctx.emit("llm/model-capacities", getattr(self.active, "model_context_windows", {}))
        return models

    async def stream(self, request: GenerateRequest) -> AsyncIterator[StreamEvent]:
        """流式生成。签名与 ``generate`` 同源,只是把结果拆成若干帧。

        适配器没有 ``stream`` 时退化为"先整体生成,再按块吐 delta" —— 这样
        消费方(agent loop)不需要关心 provider 支不支持流式。
        退化路径在全部 delta 后才发 tool_call,工具仍可派发,但不能与模型生成重叠。
        另外每帧 delta 会经 ``llm/delta`` 广播,方便 UI 或审计旁听。
        """
        request = await self.ctx.waterfall("llm/request", request, default=request)
        if not isinstance(request, GenerateRequest):
            raise LLMError("llm/request 监听器必须返回一个 GenerateRequest")

        adapter = self.active
        streamer = getattr(adapter, "stream", None)
        if streamer is None:
            result = await adapter.generate(request)
            await self.ctx.emit("llm/result", request, result)
            for event in _split_result_into_events(result):
                if event.kind == "delta":
                    await self.ctx.emit("llm/delta", request, event)
                elif event.kind == "tool_call":
                    await self.ctx.emit("llm/tool-call", request, event.tool_call)
                yield event
            yield StreamEvent("done", result=result)
            return

        saw_done = False
        async for event in streamer(request):
            if not isinstance(event, StreamEvent):
                raise LLMError(f"适配器 {adapter.name} 返回了无效的流事件", retryable=False)
            if event.kind == "delta":
                await self.ctx.emit("llm/delta", request, event)
            elif event.kind == "tool_call":
                if not isinstance(event.tool_call, ToolCall):
                    raise LLMError(f"适配器 {adapter.name} 的 tool_call 帧缺少有效 ToolCall", retryable=False)
                await self.ctx.emit("llm/tool-call", request, event.tool_call)
            elif event.kind == "done":
                if not isinstance(event.result, GenerateResult):
                    raise LLMError(f"适配器 {adapter.name} 的 done 帧缺少有效 GenerateResult", retryable=False)
                saw_done = True
                await self.ctx.emit("llm/result", request, event.result)
            else:
                raise LLMError(f"适配器 {adapter.name} 返回了未知的流事件类型 {event.kind!r}", retryable=False)
            yield event
        # Only normal exhaustion requires done; cancellation/aclose must keep
        # their original semantics instead of becoming protocol failures.
        if not saw_done:
            raise LLMError(f"适配器 {adapter.name} 的流在结束前没有给出 done 帧", retryable=False)


def _split_result_into_events(
    result: GenerateResult, size: int = 12
) -> list[StreamEvent]:
    """把完整结果切成 delta + tool_call 序列,供不支持流式的适配器复用同一条代码路径。

    虽然这里"求完再切"没有真正的并发收益,但事件形状与真流式一致 ——
    消费方(agent loop)只需要写一遍。
    顺序固定为正文 delta、思考 delta、tool_call;与 mock 为演示提前派发而先发
    tool_call 的顺序不同。此处派发时 generate() 早已结束,不能验证生成与工具并发。
    """
    events: list[StreamEvent] = []
    text = result.text or ""
    for start in range(0, len(text), size):
        events.append(StreamEvent("delta", text=text[start : start + size]))
    reasoning = result.reasoning or ""
    for start in range(0, len(reasoning), size):
        events.append(StreamEvent("delta", reasoning=reasoning[start : start + size]))
    for call in result.tool_calls:
        events.append(StreamEvent("tool_call", tool_call=call))
    return events


def plugin() -> Plugin:
    """把 ``ctx.llm`` 挂上插件树。"""

    def apply(ctx: Context) -> None:
        ctx.provide("llm", LLMService(ctx))

    return Plugin(name="llm", apply=apply, description="模型调用词汇与适配器接缝")


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)
