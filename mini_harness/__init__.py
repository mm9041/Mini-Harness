"""mini-harness:一个可运行的迷你 agent harness。

它把 DeepSeek Harness(dsh)的架构等比例缩小,保留抽象、砍掉工程细节,
用来理解"一个 agent harness 到底由哪些部件构成"。

五个抽象一一对应 dsh:

    kernel.Context / Plugin / effect / emit-waterfall-serial   -> vendor/cordis
    ctx.llm + adapters/                                        -> packages/llm/llm
    ctx.sessions                                                -> packages/core/session
    ctx.systemPrompt                                            -> packages/core/system-prompt
    ctx.tools                                                   -> packages/core/tools
    ctx.agentLoop + ctx.agents                                  -> packages/core/agent-loop, agent

零第三方依赖,只用标准库。
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
