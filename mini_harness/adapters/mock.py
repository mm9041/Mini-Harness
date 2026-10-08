"""离线适配器 —— 不花 token 也能验证整条闭环。

这两个类是"接缝可替换"最有说服力的证据:agent loop 一行没改,模型来源从真实
API 换成脚本,机器照样跑。dsh 里有对应的 ``dsh-llm-replay``(回放录制的会话),
它的 keyless 快照测试正是靠这个机制。
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

from ..kernel import Context, Plugin
from ..llm import GenerateRequest, GenerateResult, StreamEvent, ToolCall
from .openai_compat import DEFAULT_MODEL

__all__ = [
    "ScriptedAdapter",
    "MockEchoAdapter",
    "scripted_plugin",
    "echo_plugin",
    "stream_from_result",
]

DEFAULT_STREAM_SIZE = 8


async def stream_from_result(
    result: GenerateResult,
    size: int = DEFAULT_STREAM_SIZE,
    delay: float = 0.0,
) -> AsyncIterator[StreamEvent]:
    """把完整结果拆成 delta / tool_call 序列。

    离线适配器靠它复用**同一条流式代码路径** —— 于是"流式渲染"和"边流边执行"
    都能不花 token 验证。顺序刻意放成 思维链 → 工具调用 → 正文(done 收尾):
    真模型常见的行为就是"先决定调用工具,再补一段说明",这样离线也能看到
    工具调用在正文之前被派发出去。

    ``delay`` 是每帧之间等多久(默认 0)。**它对"边流边执行"的验证很关键**:
    delay=0 时生产者从不让出事件循环,提前派发的任务根本没机会跑,看不出并发;
    给一点 delay(哪怕 0.01s),重叠就真实发生了。
    """
    events: list[StreamEvent] = []
    if result.reasoning:
        for start in range(0, len(result.reasoning), size):
            events.append(
                StreamEvent("delta", reasoning=result.reasoning[start : start + size])
            )
    for call in result.tool_calls:
        events.append(StreamEvent("tool_call", tool_call=call))
    if result.text:
        for start in range(0, len(result.text), size):
            events.append(StreamEvent("delta", text=result.text[start : start + size]))
    events.append(StreamEvent("done", result=result))

    for event in events:
        if delay > 0:
            await asyncio.sleep(delay)
        yield event


class ScriptedAdapter:
    """按剧本依次吐响应,方便测试断言每一步。

    剧本条目可以是 ``GenerateResult``,也可以是简洁的 dict::

        {"text": "你好"}
        {"tool_calls": [{"name": "shell", "arguments": {"command": "ls"}}]}
    """

    name = "scripted"

    model = DEFAULT_MODEL

    def __init__(
        self,
        script: list[Any],
        fallback_text: str = "(剧本已用尽)",
        stream_size: int = DEFAULT_STREAM_SIZE,
        stream_delay: float = 0.0,
    ) -> None:
        self.script = [self._coerce(item, index) for index, item in enumerate(script)]
        self.fallback_text = fallback_text
        self.stream_size = stream_size
        self.stream_delay = stream_delay
        self.requests: list[GenerateRequest] = []

    async def generate(self, request: GenerateRequest) -> GenerateResult:
        self.requests.append(request)
        if self.script:
            return self.script.pop(0)
        return GenerateResult(text=self.fallback_text, finish_reason="stop")

    async def stream(self, request: GenerateRequest) -> AsyncIterator[StreamEvent]:
        async for event in stream_from_result(
            await self.generate(request), self.stream_size, self.stream_delay
        ):
            yield event

    def set_model(self, name: str) -> str:
        previous, self.model = self.model, name
        return previous

    async def list_models(self) -> list[str]:
        return ["mock-fast", "mock-default", "mock-strong"]

    @staticmethod
    def _coerce(item: Any, index: int) -> GenerateResult:
        if isinstance(item, GenerateResult):
            return item
        calls = [
            ToolCall(
                id=raw.get("id") or f"call_script_{index + 1}_{position + 1}",
                name=raw["name"],
                arguments=raw.get("arguments") or {},
            )
            for position, raw in enumerate(item.get("tool_calls") or [])
        ]
        return GenerateResult(
            text=item.get("text"),
            reasoning=item.get("reasoning"),
            tool_calls=calls,
            finish_reason=item.get("finish_reason")
            or ("tool_calls" if calls else "stop"),
            usage=item.get("usage"),
        )


class MockEchoAdapter:
    """两步演示:先发一个 shell 调用,再把工具结果原样回述。

    第二步是从**请求里真实投影出来的工具消息**中取内容,所以它顺带验证了
    "工具结果确实被写进日志、又被投影回模型历史"这条链路。
    """

    name = "mock"
    model = DEFAULT_MODEL

    def __init__(
        self,
        command: str = "echo mini-harness-ok",
        stream_size: int = DEFAULT_STREAM_SIZE,
        stream_delay: float = 0.0,
    ) -> None:
        self.command = command
        self.stream_size = stream_size
        self.stream_delay = stream_delay
        self.requests: list[GenerateRequest] = []

    async def generate(self, request: GenerateRequest) -> GenerateResult:
        self.requests.append(request)
        tool_messages = [m for m in request.messages if m.role == "tool"]
        if not tool_messages:
            return GenerateResult(
                text=None,
                tool_calls=[
                    ToolCall(
                        id="call_mock_1",
                        name="pwsh",
                        arguments={"command": self.command},
                    )
                ],
                finish_reason="tool_calls",
            )
        return GenerateResult(
            text=f"工具返回:{tool_messages[-1].content}", finish_reason="stop"
        )

    async def stream(self, request: GenerateRequest) -> AsyncIterator[StreamEvent]:
        async for event in stream_from_result(
            await self.generate(request), self.stream_size, self.stream_delay
        ):
            yield event

    def set_model(self, name: str) -> str:
        previous, self.model = self.model, name
        return previous

    async def list_models(self) -> list[str]:
        return ["mock-fast", "mock-default", "mock-strong"]


def scripted_plugin(
    script: list[Any],
    stream_size: int = DEFAULT_STREAM_SIZE,
    stream_delay: float = 0.0,
) -> Plugin:
    def apply(ctx: Context) -> None:
        adapter = ScriptedAdapter(
            script, stream_size=stream_size, stream_delay=stream_delay
        )
        ctx.effect(ctx.llm.register_adapter(adapter.name, adapter))
        ctx.llm.use(adapter.name)

    return Plugin(name="llm-scripted", apply=apply, inject=("llm",))


def echo_plugin(
    command: str = "echo mini-harness-ok",
    stream_size: int = DEFAULT_STREAM_SIZE,
    stream_delay: float = 0.0,
) -> Plugin:
    def apply(ctx: Context) -> None:
        adapter = MockEchoAdapter(
            command, stream_size=stream_size, stream_delay=stream_delay
        )
        ctx.effect(ctx.llm.register_adapter(adapter.name, adapter))
        ctx.llm.use(adapter.name)

    return Plugin(name="llm-mock", apply=apply, inject=("llm",))
