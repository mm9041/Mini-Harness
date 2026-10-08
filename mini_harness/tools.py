"""``ctx.tools`` —— 工具注册表 + 受保护的执行流水线。

对应 dsh 的 ``packages/core/tools``。要点有三个:

1. **工具是注册进来的,不是写死的** —— 于是"面向模型的能力"和"实现"解耦;
2. **``schemas()`` 是模型看到的那份真相** —— dsh 的 ``docs/tool-catalog.md`` 就是
   把每个工具插件真实启动一遍、读 ``ctx.tools.schemas()`` 生成出来的;
3. **执行是一条流水线,不是一个函数调用** —— 审批、参数改写、输出裁剪、审计
   都挂在流水线上,而不是改工具本体。

迷你版把 dsh 的三道瀑布简化为:``tools/pre-execute``(可拒绝)→ 处理器 →
``tools/execute``(观察)→ ``tools/post-execute``(可改写)。
"""

from __future__ import annotations

import inspect
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .kernel import Context, Plugin
from .llm import ToolCall, ToolSchema
from .session import Session

__all__ = [
    "ToolResult",
    "ToolCallContext",
    "Tool",
    "ToolsService",
    "plugin",
]

EMPTY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}


@dataclass
class ToolResult:
    """工具返回给模型的东西。错误也是结果,不能让 agent loop 崩掉。"""

    content: str
    is_error: bool = False
    images: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ToolCallContext:
    """一次工具调用的运行时上下文(工作目录、所属会话、调用 id、取消令牌)。

    ``cancellation`` 是 ``ctx.interrupt``(可选)—— 长跑的工具(如 shell)应该拿它
    和取消赛跑,而不是等超时或等模型把话说完。
    """

    call_id: str
    cwd: Path
    session: Session | None = None
    ctx: Context | None = None
    cancellation: Any = None
    approval_revision: int | None = None
    sandbox_mode: str | None = None


@dataclass
class Tool:
    """一个工具 = 名字 + 描述 + JSON Schema + 处理器。"""

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=lambda: dict(EMPTY_SCHEMA))
    handler: Callable[..., Any] = None  # type: ignore[assignment]

    # Only opt in when repeating an uncommitted call cannot mutate external state.
    safe_to_prefetch: bool = False
    # Trusted plugin metadata, never supplied by model arguments. Unknown is controlled.
    permission: str = "unknown"
    sandboxed: bool = False  # Handler must enforce context.sandbox_mode before effects.

    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )


class ToolsService:
    """工具注册表与执行入口。对应 ``ctx.tools``。"""

    def __init__(self, ctx: Context) -> None:
        self.ctx = ctx
        self._tools: dict[str, Tool] = {}

    def can_prefetch(self, name: str) -> bool:
        tool = self._tools.get(name)
        return bool(tool and tool.safe_to_prefetch)

    def permission_for(self, name: str) -> str:
        tool = self._tools.get(name)
        return tool.permission if tool else "unknown"

    def is_sandboxed(self, name: str) -> bool:
        tool = self._tools.get(name)
        return bool(tool and tool.sandboxed)

    # -------------------------------------------------------------- provider 侧
    def register(self, tool: Tool) -> Callable[[], None]:
        if tool.name in self._tools:
            raise ValueError(f"工具 {tool.name!r} 已注册")
        if tool.handler is None:
            raise ValueError(f"工具 {tool.name!r} 缺少 handler")
        if tool.permission not in {"unknown", "read", "write", "execute", "network", "delegate", "interact", "process-control"}:
            raise ValueError(f"工具 {tool.name!r} 的 permission 无效")
        self._tools[tool.name] = tool
        name = tool.name
        active = True

        def dispose() -> None:
            nonlocal active
            if active and self._tools.get(name) is tool:
                self._tools.pop(name)
            active = False

        return dispose

    # -------------------------------------------------------------- consumer 侧
    def schemas(self) -> list[ToolSchema]:
        """模型看到的那份工具清单,顺序稳定(按注册序),利于 KV 缓存。"""
        schemas = [tool.schema() for tool in self._tools.values()]
        if self.ctx.get("permissions", None) is not None:
            for schema in schemas:
                schema.parameters = deepcopy(schema.parameters)
                schema.parameters.setdefault("properties", {}).update({
                    "sandbox_permissions": {"type": "string", "enum": ["workspace-write", "danger-full-access"], "description": "仅为本次调用申请更宽权限，需要批准"},
                    "justification": {"type": "string", "description": "为何本次调用需要扩大权限"},
                })
        return schemas

    def has(self, name: str) -> bool:
        return name in self._tools

    async def execute(
        self,
        call: ToolCall,
        session: Session | None = None,
        cwd: Path | None = None,
    ) -> ToolResult:
        call = deepcopy(call)
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(
                f"未知工具 {call.name!r};可用工具: {sorted(self._tools)}", is_error=True
            )
        context = ToolCallContext(
            call_id=call.id,
            cwd=Path(cwd or Path.cwd()),
            session=session,
            ctx=self.ctx,
            cancellation=self.ctx.get("interrupt", None),
        )

        # ① 前置瀑布:审批策略、参数校验、危险命令拦截都挂在这里。
        #    监听器调 next() 表示放行;直接 return 一个 ToolResult 即为拒绝。
        allowed = object()
        denial = await self.ctx.waterfall(
            "tools/pre-execute", call, context, default=allowed
        )
        if isinstance(denial, ToolResult):
            return denial
        if denial is not allowed:
            return ToolResult("工具前置检查未完成：监听器必须返回 await next() 或明确的 ToolResult，工具未执行。", is_error=True)

        # Approval may await user input. Never run a removed/replaced handler or
        # silently switch to another tool after this call has been authorized.
        if self._tools.get(call.name) is not tool:
            return ToolResult("工具注册或调用目标在检查期间发生变化，工具未执行，请重新发起调用。", is_error=True)

        approval = self.ctx.get("approval", None)
        if approval is not None and context.approval_revision is not None and context.approval_revision != approval.revision:
            if session is not None:
                session.append("approval/revoked", call_id=call.id, reason="权限已变化，执行前授权失效")
            return ToolResult("权限已变化，执行前授权失效，请重新发起调用", is_error=True)
        if context.cancellation is not None and context.cancellation.cancelled:
            return ToolResult("已取消，工具未执行", is_error=True)

        # ② 真正的执行:任何异常都收敛成结构化错误结果。
        try:
            produced = tool.handler(call.arguments or {}, context)
            if inspect.isawaitable(produced):
                produced = await produced
            result = produced if isinstance(produced, ToolResult) else ToolResult(str(produced))
        except Exception as exc:  # noqa: BLE001 —— 故意兜住一切,交给模型自己处理
            result = ToolResult(f"{type(exc).__name__}: {exc}", is_error=True)

        # ③ 执行后观察(对应 dsh 的 tools/execute)。
        await self.ctx.emit("tools/execute", call, context, result)

        # ④ 后置瀑布:就地改写后 return await nxt()，或替换后
        #    return await nxt(call, replacement, context)。链尾返回最新结果。
        #    这里把 ``context`` 也传下去 —— 结果外溢要按会话分目录落盘,拿不到 context 就只能瞎猜。
        rewritten = await self.ctx.waterfall(
            "tools/post-execute", call, result, context,
            terminal=lambda call, latest, context: latest,
        )
        return rewritten if isinstance(rewritten, ToolResult) else ToolResult("工具后处理未返回有效 ToolResult，原始输出未发送。", is_error=True)


def plugin() -> Plugin:
    def apply(ctx: Context) -> None:
        ctx.provide("tools", ToolsService(ctx))

    return Plugin(name="tools", apply=apply, description="工具注册表与执行流水线")
