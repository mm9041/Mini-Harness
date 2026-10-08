"""``ctx.subagents`` + ``task`` 工具 —— 把活派给一个**隔离子代理**。

对应 dsh 的 `dsh-subagent` / `dsh-subagent-spawn-in-process` / `dsh-tool-subagent`。
迷你版只做最基础的那一种:**进程内 spawn、一次性(one-shot)、父等子**。

为什么需要它 —— 它是"上下文预算"的**第三种花法**:

* 外溢(spill)解决"某一条结果太大";
* 压缩(compaction)解决"历史整体太长";
* 子代理解决"**这段探索过程根本不该进父会话**"。

子代理带自己的会话日志干完一件事,只把**最终结果**交回来。中间的工具输出、失败尝试、
推理过程全留在它自己的日志里 —— 这正是 dsh 反复强调的那句
"intermediate messages and tool traffic stay outside the parent conversation"。

照抄 dsh 的三条:

1. **子代理从空对话开始**,所以 ``prompt`` 必须自包含(它看不到父会话);dsh 的 `fork`
   后端才会把父的**已完成轮次**作为种子,那是另一种取舍,这里不做;
2. **执行失败返回错误,不返回部分成功**(dsh: "failed runs return errors instead of partial
   success")—— 半个结果比没有结果更糟,因为它看起来像成功;
3. **深度上限**:子代理不能无限套娃。深度用 ``ContextVar`` 记 —— 这样将来并行派发时
   每个分支各算各的(普通计数器会被兄弟分支串扰)。
"""

from __future__ import annotations

import asyncio
import logging
from contextvars import ContextVar
from dataclasses import dataclass

from .kernel import Context, Plugin
from .diagnostics import report_exception
from .session import Session, repair_interrupted_tail
from .tools import Tool, ToolCallContext, ToolResult

__all__ = [
    "SubagentPolicy",
    "SubagentResult",
    "SubagentService",
    "TASK_PARAMETERS",
    "plugin",
]

#: 当前嵌套深度。用 ContextVar 而不是实例计数器:并行派发时每个分支各算各的。
_DEPTH: ContextVar[int] = ContextVar("mini_harness_subagent_depth", default=0)

TASK_PARAMETERS = {
    "type": "object",
    "properties": {
        "description": {
            "type": "string",
            "description": "给这次委派起个 3-5 词的名字,仅供显示。",
        },
        "prompt": {
            "type": "string",
            "description": (
                "交给子代理的任务。它看不到当前对话,所以必须自包含:"
                "把背景、目标、验收标准都写清楚。"
            ),
        },
    },
    "required": ["description", "prompt"],
    "additionalProperties": False,
}


@dataclass
class SubagentPolicy:
    #: 允许的最大嵌套深度。1 = 只允许父代理派一层(dsh 的 tool-subagent 也有 depth limit)。
    max_depth: int = 1
    #: 交给父代理的结果上限(再往上还有 spill 兜底,这里只是第一道闸)。
    result_max_chars: int = 8000


@dataclass
class SubagentResult:
    """一次委派的结果。ok 表示执行成功；save_error 单独记录持久化失败。"""

    title: str = ""
    task: str = ""
    text: str | None = None
    session_id: str | None = None
    steps: int = 0
    stopped: str = "final"  # final | cancelled | max-steps | error | refused
    event_count: int = 0
    note: str = ""
    session_path: str | None = None
    save_error: str = ""

    @property
    def ok(self) -> bool:
        return self.stopped == "final" and bool(self.text)

    @property
    def failure_reason(self) -> str:
        if self.ok:
            return ""
        if self.stopped == "refused":
            reason = self.note or "被拒绝"
        elif self.stopped == "cancelled":
            reason = "子代理被取消(父会话的取消令牌是共享的)"
        elif self.stopped == "max-steps":
            reason = f"子代理用完了步数上限({self.steps} 步仍未收尾)"
        elif self.stopped == "error":
            reason = self.note or "子代理运行中出错"
        else:
            reason = "子代理没有给出最终回答"
        return f"{reason}；日志保存也失败：{self.save_error}" if self.save_error else reason


class SubagentService:
    """进程内运行一个子代理。对应 ``ctx.subagents``。

    生命周期刻意做到"一条路走完、一个出口清理"(dsh 称之为 single quiescent disposal
    path):深度用 ``finally`` 复位,父日志的 ``subagent/end`` 一定写上 ——
    无论子代理是成功、失败还是被取消。
    """

    def __init__(self, ctx: Context, policy: SubagentPolicy | None = None) -> None:
        self.ctx = ctx
        self.policy = policy or SubagentPolicy()
        #: 已跑过的子代理(便于观测与测试)。
        self.runs: list[SubagentResult] = []

    @property
    def depth(self) -> int:
        """当前嵌套深度(0 = 父代理自己在跑)。"""
        return _DEPTH.get()

    async def run(
        self,
        task: str,
        *,
        parent_session: Session | None = None,
        title: str = "",
    ) -> SubagentResult:
        depth = _DEPTH.get()
        if depth >= self.policy.max_depth:
            reason = (
                f"已达子代理嵌套上限(深度 {self.policy.max_depth});"
                "请自己完成,不要再往下派"
            )
            result = SubagentResult(
                title=title, task=task, stopped="refused", note=reason
            )
            if parent_session is not None:
                parent_session.append(
                    "subagent/refused", title=title, stopped="refused", reason=reason
                )
            self.runs.append(result)
            return result

        child = self.ctx.sessions.create()
        approval = self.ctx.get("approval", None)
        if parent_session is not None:
            # 父日志里记一条**持久事实**:我把什么派给了谁(可审计的 lineage)
            parent_session.append(
                "subagent/start",
                title=title,
                task=task[:500],
                child_session=child.id,
                depth=depth + 1,
            )
        token = _DEPTH.set(depth + 1)
        result = SubagentResult(title=title, task=task, session_id=child.id,
                                stopped="error", note="子代理未完成初始化")
        try:
            child.append("session/workspace", cwd=str(self.ctx.agentLoop.cwd))
            if parent_session is not None:
                child.append("session/parent", parent_session=parent_session.id, title=title)
            if approval is not None and parent_session is not None:
                child.append("approval/policy", mode=approval.mode_for(parent_session))
                approval.set_parent_session(child, parent_session)
                permissions = self.ctx.get("permissions", None)
                if permissions is not None:
                    permissions.pin(child)
            # Live UI hook (started/finished); durable lineage uses start/end.
            await self.ctx.emit("subagent/started", child, task, depth + 1)
            agent = self.ctx.agents.create(child)  # type: ignore[attr-defined]
            outcome = await agent.run(task)
            result = SubagentResult(
                title=title,
                task=task,
                text=outcome.text,
                session_id=child.id,
                steps=outcome.steps,
                stopped=outcome.stopped,
                event_count=len(child.events),
            )
        except asyncio.CancelledError:
            result.stopped = "cancelled"
            result.note = "子代理任务被取消"
            raise
        except Exception as exc:  # noqa: BLE001 —— 子代理炸了不该带垮父代理
            result = SubagentResult(
                title=title,
                task=task,
                session_id=child.id,
                steps=0,
                stopped="error",
                event_count=len(child.events),
                note=f"{type(exc).__name__}: {exc}"[:400],
            )
        finally:
            _DEPTH.reset(token)
            try:
                if approval is not None and approval.parent_session(child) is not None:
                    # Preserve the final effective policy before dropping live inheritance.
                    child.append("approval/policy", mode=approval.mode_for(child), inherited=True)
                    permissions = self.ctx.get("permissions", None)
                    if permissions is not None:
                        state = permissions.snapshot(child)
                        child.append("permission/preset", preset=state["preset"], sandbox=state["sandbox"],
                                     approval=state["approval"], inherited=True)
                child.repairs.extend(repair_interrupted_tail(child))
                result.steps = len(child.events_of("step/start"))
                result.event_count = len(child.events)
                result.session_path = str(self.ctx.sessions.save(child).resolve())
            except Exception as exc:
                result.save_error = f"{type(exc).__name__}: {exc}"[:400]
                if parent_session is not None:
                    parent_session.append("command/result", text=f"子会话 {child.id} 日志保存失败，仅保留在当前进程内存中：{result.save_error}")
            finally:
                if approval is not None:
                    approval.set_parent_session(child, None)
            if parent_session is not None:
                parent_session.append(
                    "subagent/end",
                    title=title,
                    child_session=child.id,
                    stopped=result.stopped,
                    steps=result.steps,
                    session_path=result.session_path,
                    log_saved=result.session_path is not None,
                    save_error=result.save_error,
                )
            self.runs.append(result)
            try:
                await self.ctx.emit("subagent/finished", child, result)
            except Exception:
                report_exception(logging.getLogger(__name__), "子代理结束通知失败 (session=%s)", child.id, level=logging.ERROR)
            finally:
                if result.session_path is not None:
                    self.ctx.agents.release(child)
                    self.ctx.sessions.release(child)
        return result

    # ------------------------------------------------------------------ 渲染
    def render(self, result: SubagentResult) -> str:
        """把子代理的结果交给父代理 —— **只有结果,没有过程**。"""
        text = result.text or "(没有输出)"
        if len(text) > self.policy.result_max_chars:
            text = (
                f"{text[: self.policy.result_max_chars]}\n"
                f"... [子代理输出被截断,原始 {len(text)} 字符]"
            )
        label = result.title or "子代理"
        warning = (f"\n\n[日志保存失败：{result.save_error}。执行已完成，但未落盘的过程在本进程退出后无法回查。]"
                   if result.save_error else "")
        return (
            f"{label} 完成:step={result.steps},{self.session_reference(result)}\n"
            f"---\n{text}{warning}"
        )

    @staticmethod
    def session_reference(result: SubagentResult) -> str:
        if result.session_id is None:
            return "未创建子会话"
        if result.session_path:
            return f"子会话={result.session_id}（中间过程已保存至 {result.session_path}）"
        return f"子会话={result.session_id}（日志未落盘，仅当前进程内存可用）"


def make_task_tool() -> Tool:
    """构造委派工具。名字取 ``task``(dsh 里这个工具名是可配置的)。"""

    async def handler(args: dict, context: ToolCallContext) -> ToolResult:
        service = context.ctx.get("subagents", None) if context.ctx else None
        if service is None:
            return ToolResult("子代理没有装载:检查 app.build_plugins()", is_error=True)

        prompt = str(args.get("prompt") or "").strip()
        if not prompt:
            return ToolResult("缺少必填参数 prompt", is_error=True)
        title = str(args.get("description") or "").strip()

        result = await service.run(prompt, parent_session=context.session, title=title)
        if not result.ok:
            return ToolResult(
                f"委派失败:{result.failure_reason}\n" +
                service.session_reference(result),
                is_error=True,
            )
        return ToolResult(service.render(result))

    return Tool(
        name="task",
        permission="delegate",
        description=(
            "把一个**自包含**的子任务委派给隔离子代理,等它跑完并只拿回最终结果。"
            "适合:需要多步探索/大量工具输出、而过程不必占用当前对话的任务。"
            "子代理看不到当前对话,所以 prompt 里要交代清楚全部背景;"
            "简单的一步操作不要用它(直接自己做更快更省)。"
        ),
        parameters=TASK_PARAMETERS,
        handler=handler,
    )


def plugin(max_depth: int = 1, result_max_chars: int = 8000) -> Plugin:
    """装载子代理服务与 ``task`` 工具。"""

    def apply(ctx: Context) -> None:
        ctx.provide(
            "subagents",
            SubagentService(
                ctx, SubagentPolicy(max_depth=max_depth, result_max_chars=result_max_chars)
            ),
        )
        ctx.effect(ctx.tools.register(make_task_tool()))

    return Plugin(
        name="subagent",
        apply=apply,
        inject=("tools", "sessions", "agents"),
        description="进程内子代理(委派 + 结果回收)",
    )
