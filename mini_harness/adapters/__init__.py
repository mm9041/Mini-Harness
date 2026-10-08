"""模型适配器(Service Provider)集合。

每个适配器都只做一件事:把 ``GenerateRequest`` 变成 ``GenerateResult``。
它们通过 ``register_adapter`` 挂到 ``ctx.llm`` 上,消费方(agent loop)不知道
自己用的是哪一家模型 —— 这就是接缝的价值。
"""

from .mock import MockEchoAdapter, ScriptedAdapter, echo_plugin, scripted_plugin
from .openai_compat import OpenAICompatAdapter

__all__ = [
    "OpenAICompatAdapter",
    "ScriptedAdapter",
    "MockEchoAdapter",
    "scripted_plugin",
    "echo_plugin",
]
