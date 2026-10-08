"""上下文压缩 —— 把"历史无限增长"变成"预算内工作"。

对应 dsh 的三个包(`dsh-compaction` / `dsh-compaction-tool-result-pruner` /
`dsh-command-compact`)。迷你版把三件事拆开,因为它们的**性质完全不同**:

1. **度量**(`estimate_tokens`)—— 刻意粗糙的 token 估算;
2. **投影**(`project_messages`)—— **纯函数**:把已经记在日志里的压缩决定 + 工具结果裁剪,
   应用到 ``derive_messages()`` 的输出上。不调模型、不改历史、可重放;
3. **触发**(`CompactionService`)—— 压力超预算时,把最老的一段交给模型摘要,
   然后**把这件事写进日志**(``compaction`` 事件)。

由此得到这套设计里最值得记住的一条:

    **投影可算,历史不可悔。**

工具结果裁剪纯属投影 —— 原文仍在日志里,随时能拿回来精确重放;
而摘要一旦生成就**不可重建**(它是模型的输出),所以必须落成日志事件,
否则续跑之后这段历史会凭空变回原样,跟当前对话对不上。
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, replace
from typing import Any, Sequence

from .kernel import MODE_WATERFALL, Context, Plugin
from .llm import GenerateRequest, Message
from .session import Session

__all__ = [
    "estimate_tokens",
    "CompactionPolicy",
    "CompactionRecord",
    "CompactionService",
    "CompactionBusyError",
    "project_messages",
    "plugin",
]

SUMMARY_INSTRUCTION = """把上面的 agent 会话历史压缩成一份交接摘要。

保留:
- 用户的目标、约束、以及对结果的要求;
- 已确认的事实与结论(尤其是工具真正返回过的内容);
- 执行过的关键命令、改过哪些文件、当前的进展;
- 还没做完的事与已知的坑。

丢弃:寒暄、重复试探、已被推翻的中间过程。

按以下 Markdown 标题依次输出，空节写 (none)，每节用简短要点：
## Primary Request and Intent
## Key Technical Concepts
## Files and Code
## Errors and Fixes
## Pending Jobs
## Current Work
## Next Step
## Critical Context

准确保留路径、命令、错误、标识符、数字和用户纠正。不得虚构事实。
已有 <compacted-summary> 是旧检查点：合并仍有效的信息，删除过时结论，不要逐字复制。
只输出摘要正文，不调用工具，不执行任务，不把本次摘要指令写进摘要。"""


def frame_summary(summary: str) -> str:
    return ("[更早的对话已压缩为摘要]\n这是历史背景，请直接继续后续任务，无需复述摘要。\n"
            f"<compacted-summary>\n{summary}\n</compacted-summary>")


def message_tokens(messages: Sequence[Message]) -> int:
    """与驱动器一致，计入工具参数及图像，不能只数正文。"""
    total = _image_tokens(messages)
    for message in messages:
        total += estimate_tokens(message.content or "")
        for call in message.tool_calls:
            total += estimate_tokens(call.raw_arguments or json.dumps(
                call.arguments, ensure_ascii=False, separators=(",", ":")))
    return total


def estimate_tokens(text: str) -> int:
    """粗估 token 数。

    刻意粗糙:ASCII 大约 4 字符/token,中文接近 1–1.5 字符/token。
    这里按 **2 字符/token** 折中,**宁高估不低估** —— 高估只是早一点压缩,
    低估会让请求撞上上游的上下文上限。dsh 也强调字符预算只是近似,
    真正的判据是 token meter。
    """
    return len(text) // 2 + 1


def _image_tokens(messages: Sequence[Message]) -> int:
    return sum(85 + 170 * ((image.get("width", 512) + 511) // 512) * ((image.get("height", 512) + 511) // 512) for m in messages for image in m.images)


#: spill 写的 locator 标记。**这是一份跨模块的格式契约** —— 有测试盯着它,
#: 改了 spill 的写法就会红,免得两边悄悄失联。
_SPILL_LOCATOR = re.compile(r"\[[^\[\]]*已保存到\s*(?P<locator>[^\[\]]+?)\]")


def prune_tool_result(message: Message, policy: "CompactionPolicy") -> Message:
    """把一个超预算的工具结果裁成"有界头 + 中间已裁标记 + 有界尾"。

    **只影响喂给模型的这一份**,日志里的原文一个字都不动 —— 所以重放/续跑拿到的
    仍然是完整事实。dsh 的 pruner 也是这么定位的。

    有一处协同要注意:内容**已经被 spill 外溢过**时不要再叠第二层裁剪。
    外溢自己就是"按预算裁 + 给出取回路径"的策略,再裁一遍只会把外溢选好的头部
    又砍一刀,还多出一个含义重复的省略标记(实测:4216 字符的预览会被再裁成 2036)。
    这种情况只保留头部 + locator —— 全文在哪,比"我又省略了多少"有用得多。
    """
    if message.role != "tool":
        return message
    text = message.content or ""
    if len(text) <= policy.prune_tool_result_chars:
        return message

    match = _SPILL_LOCATOR.search(text)
    if match:
        head = text[: policy.head_chars]
        locator = match.group("locator").strip()
        return replace(
            message,
            content=(
                f"{head}\n\n"
                f"[中间省略;全文见 {locator} —— 用 read 读它,或 grep 取需要的一段]"
            ),
        )

    head = text[: policy.head_chars]
    tail = text[-policy.tail_chars :] if policy.tail_chars else ""
    elided = len(text) - len(head) - len(tail)
    return replace(
        message,
        content=(
            f"{head}\n\n"
            f"... [中间省略 {elided} 字符;原文仍在会话日志里] ...\n\n"
            f"{tail}"
        ),
    )


def project_messages(
    messages: Sequence[Message],
    records: Sequence[dict[str, Any]],
    policy: "CompactionPolicy",
) -> list[Message]:
    """把"日志里记着的压缩决定"应用到消息列表上。纯函数,可反复调用。

    只认**最后一条**压缩记录:后一次压缩是在前一次的投影之上做的,摘要天然覆盖前者,
    所以旧记录不需要再叠加。
    """
    projected = list(messages)

    if records:
        latest = records[-1]
        boundary = int(latest.get("covers_upto_seq") or 0)
        kept = [m for m in projected if (m.source_seq or 0) > boundary]

        # 安全对齐:压缩边界不能落在 assistant(tool_calls) 与它的 tool 结果之间,
        # 否则投影会以"孤儿 tool 消息"开头 —— 请求直接非法。
        # 边界落进这种组里时,把整个组让给摘要,从下一个完整单位重新开始。
        while kept and kept[0].role == "tool":
            kept = kept[1:]

        summary = str(latest.get("summary") or "").strip()
        projected = (
            [Message(role="user", content=frame_summary(summary))]
            if summary
            else []
        ) + kept

    return [prune_tool_result(message, policy) for message in projected]


@dataclass
class CompactionPolicy:
    """什么时候压、压完之后留多少。"""

    # 仅计算 messages（含工具参数/图片），不含 system/schema。
    # 0 关闭这条独立历史上限，不关闭完整请求的窗口压力检查。
    max_history_tokens: int = 0
    keep_recent_messages: int = 0  # 0 = token 保留策略；正数显式使用旧的条数策略
    threshold_ratio: float = 0.8
    retain_ratio: float = 0.16
    headroom_tokens: int = 0
    reserved_completion_tokens: int = 0
    prune_tool_result_chars: int = 4000  # 工具结果超过就裁成头尾
    head_chars: int = 1200
    tail_chars: int = 800
    #: 要压的内容少于这么多字符时**自动驾驶不压** —— 摘要很可能比原文还长,
    #: 白花一次模型调用。手动 ``/compact`` 不受这条限制(用户要压就压)。
    min_cover_chars: int = 2000

    def __post_init__(self) -> None:
        if not 0 < self.retain_ratio < self.threshold_ratio <= 1:
            raise ValueError("require 0 < retain_ratio < threshold_ratio <= 1")
        for name in ("max_history_tokens", "keep_recent_messages", "headroom_tokens",
                     "reserved_completion_tokens"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


@dataclass
class CompactionRecord:
    """一次压缩。会作为 ``compaction`` 事件落进日志。"""

    summary: str
    covers_upto_seq: int
    covered_messages: int
    tokens_before: int
    tokens_after: int

    @property
    def saved_tokens(self) -> int:
        return max(0, self.tokens_before - self.tokens_after)


class CompactionBusyError(RuntimeError):
    """显式压缩遇到同一会话的在途压缩；调用方应显示 busy。"""

    code = "busy"


class CompactionService:
    """压力检查、摘要生成、落日志。对应 ``ctx.compaction``。"""

    def __init__(self, ctx: Context, policy: CompactionPolicy | None = None) -> None:
        self.ctx = ctx
        self.policy = policy or CompactionPolicy()
        self._active: set[int] = set()

    # ------------------------------------------------------------------ 入口
    async def maybe_condense(self, session: Session) -> CompactionRecord | None:
        """驱动器在每个 step 前问一次:当前历史超预算了吗?超了就压。"""
        return await self._condense(session, force=False)

    async def condense_now(self, session: Session) -> CompactionRecord | None:
        """手动压缩(``/compact``)。对应 dsh 的 ``dsh-command-compact``。"""
        return await self._condense(session, force=True)

    def manual_retention_hint(self) -> str:
        """展示有效策略，包括旧的显式条数配置。"""
        if self.policy.keep_recent_messages:
            return f"保留最近至少 {self.policy.keep_recent_messages} 条消息，并保持工具调用与结果完整；更早历史会被摘要替代。"
        return "保留最新一条消息及其完整工具调用组；其余历史会被摘要替代。原始日志保留。"

    # ------------------------------------------------------------------ 实现
    def _calibrate(self, tokens: int) -> int:
        """按用量表的校准系数修正估算值。

        口径说明:系数是拿"整次请求的实测/估算"算出来的(含 system 与 tools),
        而这里只估 messages —— 是个近似。方向是对的(把系统性高估/低估拉回来),
        没打算做到精确:准确值要看 provider 报的 usage。
        """
        meter = self.ctx.get("tokenMeter", None)
        factor = getattr(meter, "factor", None)
        if not isinstance(factor, float) or factor <= 0:
            return tokens
        return int(tokens * factor)

    def project(self, session: Session) -> list[Message]:
        """当前会话在"应用完已记录的压缩"之后的样子。"""
        return project_messages(
            session.derive_messages(), session.compactions(), self.policy
        )

    def window_pressure(self, session: Session) -> bool:
        """Include system prompt and tool schemas, using the same calibrated estimate as UI."""
        meter = self.ctx.get("tokenMeter", None)
        driver = self.ctx.get("agentLoop", None)
        if meter is None or driver is None or meter.window <= 0:
            return False
        tokens = driver._estimated_tokens(
            driver.system_for(session), self.project(session), self.ctx.tools.schemas()
        )
        threshold, _ = self._budgets()
        return self._calibrate(tokens) >= threshold

    def _budgets(self) -> tuple[int, int]:
        """返回 (完整请求输入阈值, 历史尾部保留目标)，均为 token 数。

        threshold 与 window_pressure 中 system + messages + tools 的校准总量
        比较，因此不能再扣 system/schema，否则重复计入这部分开销。
        retain 是 DSH 的尾部目标，不代表摘要或完整输入能够使用的全部空间。
        max_history_tokens 是另一条仅含 messages 的独立上限，不参与本公式。
        """
        meter = self.ctx.get("tokenMeter", None)
        window = getattr(meter, "window", 0) or 8000
        available = window - self.policy.reserved_completion_tokens
        threshold = min(int(window * self.policy.threshold_ratio),
                        available - self.policy.headroom_tokens)
        retain = int(available * self.policy.retain_ratio)
        if threshold <= 0 or retain >= threshold:
            raise ValueError("压缩预算无效：输出预留和 headroom 必须留出足够的输入空间")
        return threshold, retain

    async def _condense(
        self, session: Session, *, force: bool
    ) -> CompactionRecord | None:
        key = id(session)
        if key in self._active:
            if force:
                raise CompactionBusyError("当前会话正在压缩，请等待完成后再试。")
            return None
        self._active.add(key)
        try:
            return await self._condense_once(session, force=force)
        finally:
            self._active.discard(key)

    async def _condense_once(
        self, session: Session, *, force: bool
    ) -> CompactionRecord | None:
        projected = self.project(session)
        # 压力判断用**校准后**的估算:2 字符/token 是折中值,英文场景会高估近一倍,
        # 拿它当预算会让压缩发生得过早。校准系数来自最近几次实测(见 token_meter)。
        before = self._calibrate(message_tokens(projected))

        pressure = self.window_pressure(session)
        history_pressure = self.policy.max_history_tokens > 0 and before > self.policy.max_history_tokens
        if not force and not pressure and not history_pressure:
            return None  # 还没到压力线

        keep = max(1, self.policy.keep_recent_messages)
        if len(projected) <= keep:
            return None  # 没有可压缩的余量,别为了压而压

        # 选边界:从"保留最近 keep 条"的位置**往回退**,退到不在工具结果中间的地方。
        #
        # 为什么是往回退而不是往前推:工具型会话的尾部往往是一长串 tool 消息,
        # 往前推会一直推到列表末尾 → 干脆压不了(实测踩到过:压缩静默失效,
        # 一条 compaction 事件都不产生)。往回退最多多留几条消息,代价小得多。
        # 因此 keep_recent_messages 是**下限**,不是精确值。
        cut = len(projected) - keep
        if not self.policy.keep_recent_messages:
            _, retain = self._budgets()
            # 手动/溢出压缩保留最新完整单位；普通压力保留窗口比例。
            if force:
                retain = 0
            elif self.policy.max_history_tokens:
                retain = min(retain, int(self.policy.max_history_tokens * self.policy.retain_ratio))
            cut = len(projected)
            accumulated = 0
            while cut > 0:
                cut -= 1
                accumulated += self._calibrate(message_tokens([projected[cut]]))
                if accumulated >= retain:
                    break
        while cut > 0 and projected[cut].role == "tool":
            cut -= 1
        if cut <= 0 or cut >= len(projected):
            return None

        covered = projected[:cut]
        boundary = covered[-1].source_seq or 0
        if boundary <= 0:
            return None

        # 要压的东西太少就别压:摘要很可能比原文还长(实测:
        # 3 轮短对话压出来 ≈72 tok → ≈132 tok,白花一次模型调用)。
        # 注意这条只拦**自动驾驶** —— 手动 /compact 是用户的明确意图,照做。
        covered_chars = sum(len(message.content or "") + sum(
            len(call.raw_arguments or json.dumps(call.arguments, ensure_ascii=False))
            for call in message.tool_calls) for message in covered)
        if not force and not pressure and covered_chars < self.policy.min_cover_chars and not _image_tokens(covered):
            return None

        interrupt = self.ctx.get("interrupt", None)
        if interrupt is not None and interrupt.cancelled:
            return None
        operation_id = uuid.uuid4().hex
        session.append("compaction/start", operation_id=operation_id, force=force, covers_upto_seq=boundary)
        started = time.monotonic()
        outcome = "error"
        committed = len(session.compactions())
        try:
            record = await self._compact_covered(session, covered, cut, boundary, operation_id)
            outcome = "completed" if record is not None else "cancelled"
            return record
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        finally:
            # A notification can fail after the compaction was already committed.
            if len(session.compactions()) > committed:
                outcome = "completed"
            session.append("compaction/end", operation_id=operation_id, status=outcome,
                           duration_ms=int((time.monotonic() - started) * 1000))

    async def _compact_covered(self, session, covered, cut, boundary, operation_id):
        interrupt = self.ctx.get("interrupt", None)
        if self.ctx.get("permissions", None) is not None:
            summary_call = self._summarize(covered, system=self.ctx.agentLoop.system_for(session))
        else:
            summary_call = self._summarize(covered)
        task = asyncio.create_task(summary_call)
        cancel = asyncio.create_task(interrupt.wait()) if interrupt is not None else None
        def record_usage(request, result):
            if asyncio.current_task() is task:
                session.append("compaction/usage", operation_id=operation_id,
                               model=request.model or getattr(self.ctx.llm.active, "model", ""),
                               usage=result.usage)
        unsubscribe = self.ctx.on("llm/result", record_usage)
        try:
            done, _ = await asyncio.wait({task, cancel} if cancel else {task}, return_when=asyncio.FIRST_COMPLETED)
            if cancel in done:
                return None
            summary = await task
        finally:
            for pending in (task, cancel):
                if pending is not None and not pending.done():
                    pending.cancel()
            await asyncio.gather(*(p for p in (task, cancel) if p is not None), return_exceptions=True)
            unsubscribe()
        if interrupt is not None and interrupt.cancelled:
            return None
        current = self.project(session)
        if current[:cut] != covered:
            raise RuntimeError("摘要生成期间待压缩历史发生变化，请重试")
        replacement = Message(role="user", content=frame_summary(summary))
        if self._calibrate(message_tokens([replacement])) >= self._calibrate(message_tokens(covered)):
            raise RuntimeError("摘要未缩小上下文，已保留原始历史")
        before = self._calibrate(message_tokens(current))
        tokens_after = self._calibrate(message_tokens([replacement, *current[cut:]]))
        record = CompactionRecord(
            summary=summary,
            covers_upto_seq=boundary,
            covered_messages=len(covered),
            tokens_before=before,
            tokens_after=tokens_after,
        )
        # 关键:摘要不可重建,所以必须落日志 —— 续跑/重放才对得上。
        session.append("compaction", operation_id=operation_id, **asdict(record))
        await self.ctx.emit("compaction/done", session, record)
        return record

    async def _summarize(self, messages: Sequence[Message], *, system: str | None = None) -> str:
        """用**一次额外的模型请求**生成摘要(dsh 也是这个代价模型)。"""
        request = GenerateRequest(
            system=self.ctx.systemPrompt.render() if system is None else system,
            messages=[*messages, Message(role="user", content=SUMMARY_INSTRUCTION)],
            # 复用主请求前缀；本路径不会派发任何返回的工具调用。
            tools=self.ctx.tools.schemas(),
        )
        result = await self.ctx.llm.generate(request)  # type: ignore[attr-defined]
        if result.finish_reason in {"length", "max_tokens", "max-tokens", "error", "aborted", "content_filter"} or result.tool_calls:
            raise RuntimeError("摘要模型未返回完整的纯文本摘要")
        text = (result.text or "").strip()
        if not text:
            raise RuntimeError("摘要模型没有返回内容")
        return text


def plugin(
    max_history_tokens: int = 0,
    keep_recent_messages: int = 0,
    prune_tool_result_chars: int = 4000,
    **policy_options: Any,
) -> Plugin:
    """装载压缩服务 + ``session/messages`` 投影接缝。"""

    def apply(ctx: Context) -> None:
        policy = CompactionPolicy(
            max_history_tokens=max_history_tokens,
            keep_recent_messages=keep_recent_messages,
            prune_tool_result_chars=prune_tool_result_chars,
            **policy_options,
        )
        service = CompactionService(ctx, policy)
        ctx.provide("compaction", service)

        async def project(session: Session, messages: list[Message], nxt) -> Any:
            """投影瀑布:就地改写消息列表再委托,保证多个监听器可叠加。"""
            messages[:] = project_messages(
                messages, session.compactions(), policy
            )
            return await nxt()

        ctx.effect(ctx.on("session/messages", project, mode=MODE_WATERFALL))

    return Plugin(
        name="compaction",
        apply=apply,
        inject=("llm", "tools", "sessions", "systemPrompt"),
        description="上下文压缩(投影裁剪 + 摘要落日志)",
    )
