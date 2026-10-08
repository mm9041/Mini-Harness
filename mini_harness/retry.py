"""``ctx.retry`` —— 在 **step 边界**重试失败的模型调用。

对应 dsh 的 `dsh-llm-retry`。它的几条规矩值得照抄,而且每条都有"为什么":

1. **重试发生在持久的 agent-step 边界**,不是某次 HTTP 调用内部 ——
   这样"重试"对会话日志是可解释的一件事,而不是传输层偷偷摸摸的补丁。
   直接调 ``ctx.llm.stream()`` 的路径因此**保持单次尝试**(重试是驱动器的政策,不是适配器的);
2. **排定的重试在退避之前先落进日志** —— 进程在等待期间被杀,日志也说得清发生过什么。
   这是"日志是唯一真相"的必然推论:决定一旦做出,先记账再行动;
3. **退避期间可取消** —— 取消后停止重试,当前 turn 按取消收尾,历史保持一致;
4. **错误分类靠结构化字段,不靠猜消息** —— 429/5xx 值得退避重试,400/401/403 不值得
   (重试只是白等)。上下文错误优先读取 provider code/type；缺少专用错误码时，
   仅以明确提及模型上下文容量的措辞兜底，不用泛化的参数名猜测溢出。
"""

from __future__ import annotations

import asyncio
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any

from .kernel import Context, Plugin
from .session import Session

__all__ = [
    "RetryPolicy",
    "RetryDecision",
    "RetryService",
    "DEFAULT_RETRY_STATUSES",
    "plugin",
]

#: 值得退避重试的状态码。选的是"临时性"那几类:限流、超时、网关/服务端抖动。
DEFAULT_RETRY_STATUSES: tuple[int, ...] = (408, 409, 425, 429, 500, 502, 503, 504)

#: 没有专用错误码时的保守文本兜底。误判会多发摘要请求、改写前缀并可能损失缓存，
#: 也可能缩短逐字保留的尾部；不能把它当成无代价的普通重试。
DEFAULT_OVERFLOW_PATTERNS: tuple[str, ...] = (
    "maximum context length", "maximum context window",
    "input is too long for this model", "input is too long for the model",
    "上下文长度超过", "上下文超出", "上下文超过",
)
_CONTEXT_CODES = {"context_length_exceeded", "context_window_exceeded", "context_window_overflowed"}
_CONTEXT_WORDING = re.compile(
    r"(?:\bcontext[\s_-](?:length|window)[\s_-](?:exceeded|overflowed|limit[\s_-]exceeded)\b"
    r"|\b(?:request|prompt|input|messages?)\b.{0,40}\b(?:exceeds?|exceeded|overflows?|too (?:large|long) for)\b.{0,40}\b(?:model(?:'s)?\s+)?context(?:\s+(?:length|window))?\b"
    r"|上下文(?:长度|窗口)?.{0,12}(?:超出|超过|超限)"
    # Reverse Chinese order: constrain modifiers, not arbitrary text between
    # 'exceeded' and 'context' (rate limits/retry limits must not trigger compaction).
    r"|(?<!未)(?<!不)(?<!没有)(?<!不能)(?<!不得)(?:超出|超过)(?:了)?\s*(?:(?:当前|本|该|此|模型|允许|支持|所|设定|设置|规定|的|最大|可用)\s*){0,10}"
    r"上下文(?:(?:最大|可用)?(?:长度|窗口|容量|上限|限制)|(?=$|[。！？!?；;])))", re.I,
)


@dataclass
class RetryPolicy:
    """重试策略。分"有界 normal 模式"和"无限 always 模式"(照 dsh 的分法)。"""

    #: 总尝试次数(含第一次)。``0`` 或不重试。
    max_attempts: int = 3
    base_delay: float = 0.5
    max_delay: float = 8.0
    #: 抖动比例:多实例同时被限流时,别让它们在同一毫秒一起重试。
    jitter: float = 0.25
    #: 无限重试,直到成功或取消(对应 dsh 的 always 模式)。
    always: bool = False
    retry_statuses: tuple[int, ...] = DEFAULT_RETRY_STATUSES
    overflow_patterns: tuple[str, ...] = DEFAULT_OVERFLOW_PATTERNS


@dataclass
class RetryDecision:
    """一次失败之后该怎么办。"""

    kind: str  # retry | overflow | give-up
    attempt: int  # 已经用掉的尝试次数
    delay: float
    reason: str

    @property
    def will_retry(self) -> bool:
        return self.kind in ("retry", "overflow")

    @property
    def is_overflow(self) -> bool:
        return self.kind == "overflow"


class RetryService:
    """分类、决定、落日志、退避。对应 ``ctx.retry``。"""

    def __init__(self, policy: RetryPolicy | None = None) -> None:
        self.policy = policy or RetryPolicy()
        #: 本次进程里重试过多少次(便于观测/测试)。
        self.attempts = 0
        self.recovered = 0
        #: 退避累计等待秒数 —— 驱动器把它从"模型耗时"里拆出来,
        #: 否则"这一步 12 秒"看起来像模型慢,其实是退避等了 10 秒。
        self.waited_seconds = 0.0

    # ------------------------------------------------------------------ 分类
    def classify(self, exc: BaseException) -> str:
        """返回 ``retryable`` / ``overflow`` / ``fatal``。

        显式 overflow 优先；鉴权/额度/临时状态不做压缩；其余先看专用错误码，
        仅在请求错误或无状态码时使用保守文本兜底。
        """
        if getattr(exc, "kind", None) == "overflow":
            return "overflow"

        status = getattr(exc, "status", None)
        code = (getattr(exc, "provider_code", None) or "").lower()
        error_type = (getattr(exc, "provider_type", None) or "").lower()
        if status in (401, 402, 403) or {code, error_type} & {"authentication_error", "permission_error", "insufficient_quota", "quota_exceeded"}:
            return "fatal"
        explicit = getattr(exc, "retryable", None)
        if explicit is False:
            return "fatal"
        if status in self.policy.retry_statuses:
            return "retryable"
        if status in (None, 400, 413, 422):
            if {code, error_type} & _CONTEXT_CODES:
                return "overflow"
            message = getattr(exc, "provider_message", None)
            if message is None:
                # Use the actual exception message, not LLMError.__str__'s
                # appended status/kind decorations, when one was supplied.
                message = exc.args[0] if len(exc.args) == 1 and isinstance(exc.args[0], str) else str(exc)
            text = message.lower()
            if _CONTEXT_WORDING.search(text) or any(pattern in text for pattern in self.policy.overflow_patterns):
                return "overflow"
        if status is not None:
            return "fatal"

        if explicit is True:
            return "retryable"
        # 认不出来的错误一律不重试:宁可快失败,也不要在不确定的地方空转。
        return "fatal"

    # ------------------------------------------------------------------ 决策
    def decide(self, exc: BaseException, attempt: int) -> RetryDecision:
        """``attempt`` 是**已经用掉的尝试次数**(从 1 开始)。"""
        kind = self.classify(exc)

        if kind == "fatal":
            return RetryDecision("give-up", attempt, 0.0, "这类错误不该重试(重试只是白等)")
        if kind == "overflow":
            return RetryDecision("overflow", attempt, 0.0, "上下文超出上限:压缩后再试")

        if not self.policy.always and attempt >= max(1, self.policy.max_attempts):
            return RetryDecision(
                "give-up", attempt, 0.0, f"已达重试上限({self.policy.max_attempts} 次尝试)"
            )

        return RetryDecision("retry", attempt, self.delay_for(attempt), "可重试的临时故障")

    def delay_for(self, attempt: int) -> float:
        """指数退避 + 抖动,并封顶。"""
        raw = self.policy.base_delay * (2 ** max(0, attempt - 1))
        capped = min(self.policy.max_delay, raw)
        if capped <= 0 or self.policy.jitter <= 0:
            return max(0.0, capped)
        return max(0.0, capped * (1 + random.uniform(-self.policy.jitter, self.policy.jitter)))

    # ------------------------------------------------------------------ 退避
    async def wait(self, delay: float, cancellation: Any = None) -> bool:
        """退避等待。返回 ``False`` 表示等待期间被取消,调用方应停止重试。"""
        if cancellation is not None and cancellation.cancelled:
            return False
        started = time.perf_counter()
        try:
            return await self._sleep(delay, cancellation)
        finally:
            self.waited_seconds += time.perf_counter() - started

    async def _sleep(self, delay: float, cancellation: Any = None) -> bool:
        if delay <= 0:
            return True
        if cancellation is None:
            await asyncio.sleep(delay)
            return True

        sleep_task = asyncio.ensure_future(asyncio.sleep(delay))
        cancel_task = asyncio.ensure_future(cancellation.wait())
        try:
            done, pending = await asyncio.wait(
                {sleep_task, cancel_task}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            return sleep_task in done
        finally:
            for task in (sleep_task, cancel_task):
                if not task.done():
                    task.cancel()

    # ------------------------------------------------------------------ 记账
    @staticmethod
    def note_scheduled(
        session: Session, step: int, decision: RetryDecision, exc: BaseException
    ) -> None:
        """**退避之前**先落日志 —— 决定一旦做出就先记账。"""
        session.append(
            "retry/scheduled",
            step=step,
            attempt=decision.attempt,
            delay=round(decision.delay, 3),
            kind=decision.kind,
            reason=decision.reason,
            error=f"{type(exc).__name__}: {exc}"[:400],
        )

    @staticmethod
    def note_gave_up(
        session: Session,
        step: int,
        decision: RetryDecision,
        reason: str | None = None,
    ) -> None:
        session.append(
            "retry/gave-up",
            step=step,
            attempt=decision.attempt,
            reason=reason or decision.reason,
        )

    def note_recovered(self, session: Session, step: int, attempt: int) -> None:
        self.recovered += 1
        session.append("retry/recovered", step=step, attempts=attempt)

    def note_attempt(self) -> None:
        self.attempts += 1


def plugin(
    max_attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    always: bool = False,
) -> Plugin:
    """装载重试策略。驱动器会在 step 边界上咨询它。"""

    def apply(ctx: Context) -> None:
        ctx.provide(
            "retry",
            RetryService(
                RetryPolicy(
                    max_attempts=max_attempts,
                    base_delay=base_delay,
                    max_delay=max_delay,
                    always=always,
                )
            ),
        )

    return Plugin(
        name="llm-retry",
        apply=apply,
        description="在 step 边界重试失败的模型调用(先落日志再退避)",
    )
