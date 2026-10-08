"""``ctx.systemPrompt`` —— 系统提示的分区拼装。

对应 dsh 的 ``packages/core/system-prompt``。关键点不在于"拼字符串",而在于
**每个分区都是一条可撤销的注册**:插件卸载时它的提示段自动消失,不会留下孤儿文本。
dsh 里那段"提示只作为 system/message 历史流动、空渲染会清掉全部旧节点"的规则
属于更上层的请求构造,迷你版不涉及。
"""

from __future__ import annotations

from typing import Callable

from .kernel import Context, Plugin

__all__ = ["SystemPromptService", "PERSONA_ORDER", "CONTEXT_ORDER", "DEFAULT_ORDER", "plugin"]

PERSONA_ORDER = 0
CONTEXT_ORDER = 50
DEFAULT_ORDER = 100


class SystemPromptService:
    """命名分区 → 文本。渲染时按 (order, 注册序) 稳定排序。"""

    def __init__(self, persona: str = "") -> None:
        self._sections: dict[str, tuple[int, int, str | Callable[[], str]]] = {}
        self._counter = 0
        if persona:
            self.set_persona(persona)

    def set_persona(self, text: str) -> None:
        """覆盖 persona 配置；不返回插件式撤销入口，空文本渲染时省略。"""
        self.add_section("persona", text, order=PERSONA_ORDER)

    def add_section(
        self, name: str, text: str | Callable[[], str], order: int = DEFAULT_ORDER
    ) -> Callable[[], None]:
        """登记或替换提示分区；同 order 下以本次注册的先后排序。

        动态分区在每次 render 时读取当前状态，不能产生副作用。
        """
        entry = (order, self._counter, text)
        self._counter += 1
        self._sections[name] = entry

        def dispose() -> None:
            if self._sections.get(name) is entry:
                self._sections.pop(name, None)

        return dispose

    def render(self) -> str:
        ordered = sorted(self._sections.values(), key=lambda entry: entry[:2])
        rendered = [text() if callable(text) else text for _, _, text in ordered]
        return "\n\n".join(text for text in rendered if text.strip())

    @property
    def sections(self) -> dict[str, str]:
        return {name: text() if callable(text) else text for name, (_, _, text) in self._sections.items()}


def plugin(persona: str = "") -> Plugin:
    def apply(ctx: Context) -> None:
        ctx.provide("systemPrompt", SystemPromptService(persona))

    return Plugin(name="system-prompt", apply=apply, description="系统提示分区拼装")
