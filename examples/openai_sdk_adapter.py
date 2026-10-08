"""用**官方 openai 库**实现同一个接缝 —— 反证 "换 transport 不用改别的"。

`mini_harness/adapters/openai_compat.py` 是手写的 HTTP/SSE(为了零第三方依赖)。
但这个仓库的接缝不是那个文件,而是 `LLMAdapter` 协议:**一个名字 + `generate`,
外加可选的 `stream`**。换成官方 SDK 只需要写这一个类,别处一行都不用动 ——
agent loop、工具、审批、会话日志、REPL 全都不知道 provider 换了。

跑法(唯一多出来的一步是装库)::

    pip install openai
    python examples/openai_sdk_adapter.py "用 shell 数一下当前目录有多少 .py 文件"

这个文件**不属于默认路径**,只是把 "为什么不直接用 openai 库" 这个问题的答案
做成可运行的证据。核心复用点:`StreamAccumulator` 是**策略**(把流里的碎片拼成
完整工具调用、并判断参数是否闭合),不是 transport —— 所以它能被两套客户端共用。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, AsyncIterator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mini_harness import llm  # noqa: E402
from mini_harness.adapters.openai_compat import (
    StreamAccumulator,
    parse_completion,
)  # noqa: E402
from mini_harness.app import HarnessConfig, build_context, build_plugins  # noqa: E402
from mini_harness.console import TracePrinter  # noqa: E402
from mini_harness.kernel import Context, Plugin, mount  # noqa: E402
from mini_harness.llm import (  # noqa: E402
    GenerateRequest,
    GenerateResult,
    StreamEvent,
)


class OpenAISDKAdapter:
    """和 ``OpenAICompatAdapter`` 同一个协议,底层换成官方 SDK。

    对比一下就知道哪些代码是"transport"、哪些是"策略":

    ===========================  ==========================  ====================
    做的事                        手写版(SDK 版如何)           SDL 帮不帮忙
    ===========================  ==========================  ====================
    HTTP / SSE / 超时 / 重试       手写(urllib + 子线程)          ✅ 全包
    线程 ↔ 事件循环桥接            40 行 asyncio.Queue 桥         ✅ 原生 async
    消息与工具 schema 组包         手写 dict                      ✅ 顺便帮你校验
    工具调用碎片拼装                StreamAccumulator             ❌ 仍要自己写
    "参数补全了吗"的判定            StreamAccumulator + 深度扫描    ❌ 仍要自己写
    响应 → GenerateResult          parse_completion                        ⚠️ 给你类型化对象了
    base_url 容错解析               resolve_chat_endpoint         ❌ 反而更严格
    ===========================  ==========================  ====================
    """

    name = "openai-sdk"

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float = 120.0,
    ) -> None:
        try:
            from openai import AsyncOpenAI  # 按需导入:没装库时这个文件仍然能被导入
        except ImportError as exc:  # pragma: no cover - 取决于环境
            raise SystemExit(
                "这个例子需要官方 SDK:pip install openai\n"
                "(仓库自带的适配器不需要任何依赖,这个例子只是用来对比)"
            ) from exc

        self.model = model
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    # ------------------------------------------------------------------ 组包
    def _kwargs(self, request: GenerateRequest, *, stream: bool) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": request.model or self.model,
            "messages": [],
            "stream": stream,
        }
        if request.system:
            kwargs["messages"].append({"role": "system", "content": request.system})
        kwargs["messages"].extend(m.to_wire() for m in request.messages)
        if request.tools:
            kwargs["tools"] = [schema.to_wire() for schema in request.tools]
        return kwargs

    # ------------------------------------------------------------------ 非流式
    async def generate(self, request: GenerateRequest) -> GenerateResult:
        response = await self.client.chat.completions.create(
            **self._kwargs(request, stream=False)
        )
        # SDK 给的是类型化对象;`model_dump()` 之后就能复用同一套解析策略。
        # (extra="allow" 让非标准字段如 reasoning_content 也保留下来)
        return parse_completion(response.model_dump())

    # -------------------------------------------------------------------- 流式
    async def stream(self, request: GenerateRequest) -> AsyncIterator[StreamEvent]:
        accumulator = StreamAccumulator()
        # SDK 原生 async,不需要手写线程桥 —— 这就是依赖换来的东西之一
        chunks = await self.client.chat.completions.create(
            **self._kwargs(request, stream=True)
        )
        async for chunk in chunks:
            for event in accumulator.feed(chunk.model_dump()):
                yield event
        yield StreamEvent("done", result=accumulator.finish())


def plugin(adapter: OpenAISDKAdapter) -> Plugin:
    def apply(ctx: Context) -> None:
        ctx.effect(ctx.llm.register_adapter(adapter.name, adapter))
        ctx.llm.use(adapter.name)

    return Plugin(name="llm-openai-sdk", apply=apply, inject=("llm",))


def build_context_with_sdk(config: HarnessConfig) -> Context:
    """默认插件树 + 官方 SDK 适配器(替换掉自带的那个 provider)。

    这一行 `if not plugin.name.startswith("llm-")` 就是**全部**的改动量 ——
    其余每一层(工具、审批、会话、驱动器)都不知道换过。
    """
    adapter = OpenAISDKAdapter(
        api_key=config.api_key,
        base_url=config.base_url,
        model=config.model,
        timeout=config.timeout,
    )
    plugins = [
        item for item in build_plugins(config) if not item.name.startswith("llm-")
    ]
    plugins.insert(0, plugin(adapter))  # 放前面:它 provide 的是 llm 依赖

    ctx = Context(label="sdk-demo")
    mount(ctx, plugins)
    return ctx


async def main(task: str) -> int:
    config = HarnessConfig.from_env(offline=None)
    ctx = build_context_with_sdk(config)

    session = ctx.sessions.create()
    printer = TracePrinter()
    printer.observe(session)
    dispose = printer.attach(ctx)

    print(f"[sdk 版] 会话 {session.id} | 模型 {config.model}")
    print(f"[sdk 版] provider = {ctx.llm.active.name}(官方 openai 库)")
    print(f"[sdk 版] 工具 {[schema.name for schema in ctx.tools.schemas()]}\n")
    try:
        result = await ctx.agents.create(session).run(task)
    finally:
        dispose()

    print(f"\n=== 最终回答 ===\n{result.text or '(空)'}")
    print(f"\n[会话] step={result.steps} 停止原因={result.stopped} "
          f"事件数={len(session.events)}")
    return 0


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "用 shell 执行 echo hi"))
