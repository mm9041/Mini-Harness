"""``ctx.approval`` —— 工具调用的审批接缝。

对应 dsh 的 approval 子系统与 ``ctx.userQuestions``。这里把"要不要问"和"怎么问"
拆开,因为这两件事该由不同的人负责:

* **策略(policy)** —— 哪些调用需要审批。属于产品/部署决定,放在插件里;
* **审批人(approver)** —— 怎么问人。属于 UI 决定,由调用方注入
  (CLI 弹 y/N、Web 弹卡片、CI 直接拒绝)。

闸门挂在 ``tools/pre-execute`` 瀑布上 —— 不调 ``next()`` 就等于拒绝,
这正是 dsh 里审批策略的位置。
默认装配由 permissions.authorize 完成沙箱授权与审批;通过后直接委托下游。
ApprovalPolicy 的危险命令正则仅用于未装载 permissions 的旧装配。
"""

from __future__ import annotations

import re
import asyncio
import threading
from copy import deepcopy
from uuid import uuid4
from dataclasses import dataclass
from typing import Awaitable, Callable

from .kernel import MODE_WATERFALL, Context, Plugin
from .llm import ToolCall
from .tools import ToolResult
from .session import Session

__all__ = [
    "ApprovalRequest",
    "ApprovalDecision",
    "ApprovalPolicy",
    "ApprovalService",
    "ACCESS_LABELS",
    "current_mode",
    "DEFAULT_DANGEROUS_PATTERNS",
    "plugin",
]

#: 审批显示名；不宣称文件或命令沙箱边界。
ACCESS_LABELS: dict[str, str] = {
    "ask": "操作前询问",
    "allow": "自动批准",
    "deny": "拒绝受控操作",
}

#: 危险命令模式只补充原因；任何 shell 执行都需要授权。
DEFAULT_DANGEROUS_PATTERNS: tuple[str, ...] = (
    r"\brm\s+-[a-z]*r",          # rm -r / rm -rf(含组合参数)
    r"\brm\s+-[a-z]*f",
    r"\brmdir\b",
    r"\bdel\s+/[a-z]",
    r"\bformat\b",
    r"\bmkfs\b",
    r"\bdd\s+if=",
    r"\bdiskpart\b",
    r"\bshutdown\b",
    r"\breboot\b",
    r":\(\)\s*\{",               # fork bomb
    r"\bgit\s+push\b.*--force",
    r"\bgit\s+reset\s+--hard",
    r"\bchmod\s+-R\s+777",
    r">\s*/dev/sd",
    r"\|\s*(?:ba|z)?sh\b",       # curl ... | sh
    r"\bInvoke-Expression\b",
    r"\bRemove-Item\b.*-Recurse",
)

Approver = Callable[["ApprovalRequest"], "ApprovalDecision | Awaitable[ApprovalDecision]"]


@dataclass
class ApprovalRequest:
    call: ToolCall
    reason: str
    tool: str = ""
    session: Session | None = None
    cancellation: object = None
    policy: str | None = None  # DSH request policy: ask / never; separate from sandbox.

    def __post_init__(self) -> None:
        if not self.tool:
            self.tool = self.call.name


@dataclass
class ApprovalDecision:
    approved: bool
    note: str = ""
    outcome: str = ""


@dataclass
class ApprovalPolicy:
    """哪些调用需要审批。"""

    #: 这些工具总是需要审批(写类工具动的是用户的工作区)。
    always: tuple[str, ...] = ("write_file", "write", "edit", "pwsh")
    #: 命中这些正则的 shell 命令需要审批。
    patterns: tuple[str, ...] = DEFAULT_DANGEROUS_PATTERNS
    #: 这些工具名按 shell 语义检查命令(默认只有 shell)。
    shell_tools: tuple[str, ...] = ("shell", "pwsh")

    def reason_for(self, call: ToolCall, permission: str = "unknown") -> str | None:
        """按可信工具元数据判定；正则只能增加限制，不能证明任意命令安全。"""
        if call.name in self.always:
            return f"{call.name} 属于需要审批的工具"
        if call.name in self.shell_tools:
            command = str(call.arguments.get("command") or "")
            for pattern in self.patterns:
                if re.search(pattern, command, flags=re.IGNORECASE):
                    return f"命令命中危险模式 {pattern}"
            return "命令执行需要审批；当前没有操作系统沙箱"
        if permission in {"read", "network", "delegate", "interact"}:
            return None
        return f"{call.name} 的操作类型为 {permission}，需要审批"


class ApprovalService:
    """审批入口。对应 ``ctx.approval``。"""

    def __init__(self, policy: ApprovalPolicy | None = None, mode: str = "ask") -> None:
        if mode not in ACCESS_LABELS:
            raise ValueError(f"未知审批模式 {mode!r};可选 ask / allow / deny")
        self.policy = policy or ApprovalPolicy()
        self.mode = mode  # Deployment fallback; UI changes are session-scoped.
        self.revision = 0
        self._parents: dict[str, Session] = {}
        self._lock = threading.RLock()
        self._pending: set[asyncio.Future] = set()
        self._approver: Approver | None = None
        self.history: list[tuple[ApprovalRequest, ApprovalDecision]] = []

    def parent_session(self, session: Session) -> Session | None:
        """返回运行中子代理会话的父会话;没有关联时返回 None。"""
        with self._lock:
            return self._parents.get(session.id)

    def set_parent_session(self, session: Session, parent: Session | None) -> None:
        """设置子代理的父会话;传入 None 解除运行时关联。"""
        with self._lock:
            if parent is None:
                self._parents.pop(session.id, None)
            else:
                self._parents[session.id] = parent

    def mode_for(self, session: Session | None = None) -> str:
        if session is not None:
            parent = self.parent_session(session)
            if parent is not None:
                return self.mode_for(parent)
            for event in reversed(session.events):
                if event.type == "approval/policy":
                    mode = event.data.get("mode")
                    if mode not in ACCESS_LABELS:
                        raise ValueError("会话审批策略无效")
                    return mode
        return self.mode

    def set_approver(self, approver: Approver | None) -> None:
        self._approver = approver

    def invalidate(self):
        with self._lock:
            self.revision += 1
            for future in tuple(self._pending):
                future.get_loop().call_soon_threadsafe(self._invalidate, future)

    def set_mode(self, mode: str, session: Session | None = None) -> str:
        """改变策略并撤销所有在途审批；切回原档位也不能复用旧批准。"""
        if mode not in ACCESS_LABELS:
            raise ValueError(f"未知权限档位 {mode!r};可选 {' / '.join(ACCESS_LABELS)}")
        with self._lock:
            previous = self.mode_for(session)
            if previous == mode:
                return previous
            if session is None:
                self.mode = mode
            else:
                session.append("approval/policy", mode=mode, previous=previous)
            self.revision += 1
            for future in tuple(self._pending):
                future.get_loop().call_soon_threadsafe(self._invalidate, future)
        return previous

    @staticmethod
    def _invalidate(future: asyncio.Future) -> None:
        if not future.done():
            future.set_result(None)

    async def decide(self, request: ApprovalRequest) -> ApprovalDecision:
        request_id = uuid4().hex
        session = request.session
        with self._lock:
            revision = self.revision
            mode = request.policy if request.policy is not None else self.mode_for(session)
            changed = asyncio.get_running_loop().create_future()
            self._pending.add(changed)
        if session is not None:
            session.append("approval/asked", id=request_id, call_id=request.call.id,
                           tool=request.call.name, reason=request.reason, mode=mode)
        decision = ApprovalDecision(False, "审批不可用", "unavailable")
        task = cancel = None
        try:
            if request.cancellation is not None and request.cancellation.cancelled:
                decision = ApprovalDecision(False, "已取消", "cancelled")
            elif mode == "allow":
                decision = ApprovalDecision(True, "审批模式 allow，自动批准", "allowed-once")
            elif mode in ("deny", "never"):
                decision = ApprovalDecision(False, "审批模式 deny，拒绝受控操作", "rejected")
            elif self._approver is None:
                decision = ApprovalDecision(False, "当前没有可用的审批人，已拒绝；可配置审批人或显式使用 --approval allow", "unavailable")
            else:
                async def ask():
                    # 审批 UI 得到独立参数快照，不能改写即将执行的调用。
                    shown = ApprovalRequest(deepcopy(request.call), request.reason, request.tool,
                                            session, request.cancellation)
                    value = self._approver(shown)
                    return await value if hasattr(value, "__await__") else value
                task = asyncio.create_task(ask())
                if request.cancellation is not None:
                    cancel = asyncio.create_task(request.cancellation.wait())
                done, _ = await asyncio.wait({task, changed, *([cancel] if cancel else [])}, return_when=asyncio.FIRST_COMPLETED)
                if changed in done or (cancel is not None and cancel in done):
                    decision = ApprovalDecision(False, "权限已变化或操作已取消，旧审批失效", "cancelled")
                else:
                    value = await task
                    if not isinstance(value, ApprovalDecision) or type(value.approved) is not bool:
                        raise ValueError("审批结果必须包含布尔类型 approved")
                    decision = ApprovalDecision(value.approved, value.note,
                                                "allowed-once" if value.approved else "rejected")
            if revision != self.revision or (request.cancellation is not None and request.cancellation.cancelled):
                decision = ApprovalDecision(False, "权限已变化或操作已取消，旧审批失效", "cancelled")
        except asyncio.CancelledError:
            decision = ApprovalDecision(False, "审批已取消", "cancelled")
            raise
        except Exception as exc:
            decision = ApprovalDecision(False, f"审批不可用：{type(exc).__name__}: {exc}", "unavailable")
        finally:
            for pending in (task, cancel):
                if pending is not None and not pending.done():
                    pending.cancel()
            await asyncio.gather(*(p for p in (task, cancel) if p is not None), return_exceptions=True)
            with self._lock:
                self._pending.discard(changed)
            if not changed.done():
                changed.cancel()
            # Cleanup can yield; check policy again before publishing any grant.
            if decision.approved and revision != self.revision:
                decision = ApprovalDecision(False, "权限已变化，旧审批失效", "cancelled")
            self.history.append((request, decision))
            if session is not None:
                session.append("approval/decided", id=request_id, outcome=decision.outcome, note=decision.note)
        return decision


def current_mode(ctx: Context, fallback: str = "ask", session: Session | None = None) -> str:
    """读当前权限档位。服务不在时回退到 ``fallback``。

    为什么从服务读而不是配置读:``/access`` 是**运行时**切换,配置里那个值是启动时的快照。
    显示要是读配置,切完档位状态行会撒谎。
    """
    service = ctx.get("approval", None)
    return service.mode_for(session) if service is not None else fallback


def plugin(
    mode: str = "ask",
    always: tuple[str, ...] | None = None,
    patterns: tuple[str, ...] | None = None,
) -> Plugin:
    """装载审批策略与 ``tools/pre-execute`` 闸门。"""

    def apply(ctx: Context) -> None:
        policy = ApprovalPolicy(
            always=tuple(always) if always is not None else ApprovalPolicy().always,
            patterns=tuple(patterns) if patterns is not None else DEFAULT_DANGEROUS_PATTERNS,
        )
        service = ApprovalService(policy, mode=mode)
        ctx.provide("approval", service)

        async def gate(call: ToolCall, context, nxt):
            context.approval_revision = service.revision
            permissions = ctx.get("permissions", None)
            if permissions is not None:
                denial = await permissions.authorize(call, context)
                return denial if denial is not None else await nxt()
            reason = policy.reason_for(call, ctx.tools.permission_for(call.name))
            if reason is None:
                return await nxt()  # 不需要审批:委托给下游

            decision = await service.decide(ApprovalRequest(call, reason, session=context.session, cancellation=context.cancellation))
            if decision.approved:
                await ctx.emit("approval/granted", call, reason, decision.note)
                return await nxt()

            await ctx.emit("approval/denied", call, reason, decision.note)
            note = f";{decision.note}" if decision.note else ""
            # 不调 next() = 短路,工具不会执行。
            return ToolResult(f"已被审批策略拒绝:{reason}{note}", is_error=True)

        ctx.effect(ctx.on("tools/pre-execute", gate, mode=MODE_WATERFALL))

    return Plugin(
        name="approval",
        apply=apply,
        inject=("tools",),
        description="工具调用审批闸门",
    )
