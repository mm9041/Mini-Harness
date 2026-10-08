"""``ctx.agentLoop`` + ``ctx.agents`` —— turn / step 驱动器。

对应 dsh 的 ``packages/core/agent-loop``(默认驱动器)与 ``packages/core/agent``
(Agent 接口与注册表)。dsh 对这两个词的定义,迷你版照抄:

    **step** = 一次模型请求 + 它调用的工具;
    **turn** = 零到多个 step,从第一次输入被接受开始,直到"没有欠账"为止。

循环里有三处刻意做对的事:

* 模型历史**每次都从日志重新投影**(``session.derive_logs()`` 等价物),而不是自己
  维护一个消息列表 —— 这是 "model-visible means logged" 不变量在代码里的样子;
* 工具还欠一次请求就再走一个 step,这一判断取代了"递归调用自己"的常见写法;
* **取消是协作式的**:每个检查点(开新 step、发请求前后、执行每个工具前、
  每个流式增量)都看一次令牌。取消时必须把**已记录的** ``tool/call`` 全部补齐
  ``tool/result`` —— 否则下一个请求里那条 ``assistant(tool_calls)`` 没有对应结果,
  协议上就是非法的,整个会话就废了。
* **边流边执行 = 并发执行 + 串行落账**:流式里某条工具调用参数一补全就派发出去
  (模型还在说话,工具已经跑起来了),但**写日志仍然严格按调用顺序** ——
  快慢可以乱,账不能乱。
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .compaction import estimate_tokens, message_tokens
from .kernel import Context, Plugin
from .llm import GenerateRequest, GenerateResult, LLMError, Message, ToolCall
from .session import Session
from .tools import ToolResult

__all__ = [
    "RunResult",
    "AssistantStreamFrame",
    "Agent",
    "AgentsService",
    "AgentLoopService",
    "plugin",
]


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _since(started: float) -> int:
    """从 ``started`` 到现在过了多少毫秒。"""
    return int((time.perf_counter() - started) * 1000)


def _ms(seconds: float) -> int:
    return int(seconds * 1000)


@dataclass
class RunResult:
    text: str | None
    steps: int
    session: Session
    stopped: str  # final | max-steps | empty-turn | cancelled
    #: 这一轮从 turn/start 到 turn/end 的墙钟毫秒数。
    duration_ms: int = 0


@dataclass
class AssistantStreamFrame:
    """流式过程中的一帧(process-local,不落日志)。

    对应 dsh 的 ``agent/assistant-stream``:start / chunk / end 三态。它的价值在于
    "增量给 UI,落盘只落拼装好的整条消息" —— 日志不因为流式而变脏。
    """

    phase: str  # start | chunk | end
    text: str = ""
    reasoning: str = ""


class AgentLoopService:
    """默认驱动器。对应 ``ctx.agentLoop``。"""

    def __init__(
        self,
        ctx: Context,
        max_steps: int = 64,
        cwd: Path | None = None,
        model: str | None = None,
        streaming: bool = True,
        early_tools: bool = True,
    ) -> None:
        self.ctx = ctx
        if type(max_steps) is not int or max_steps <= 0:
            raise ValueError("max_steps 必须是正整数；它限制一轮任务的模型决策步数")
        self.max_steps = max_steps
        self.cwd = Path(cwd or Path.cwd())
        self.model = model
        self.streaming = streaming
        self.early_tools = early_tools
        self._step_started = 0.0  # 当前 step 的起点(给 _end_step 算耗时)

    # ----------------------------------------------------------- 取消令牌读取
    @property
    def _interrupt(self) -> Any:
        return self.ctx.get("interrupt", None)

    def _cancel_reason(self) -> str | None:
        token = self._interrupt
        if token is not None and token.cancelled:
            return token.reason or "已取消"
        return None

    # ---------------------------------------------------------------- 主循环
    async def run(self, session: Session, text: str, *, attachments=None) -> RunResult:
        token = self._interrupt
        if token is not None:
            # 让信号处理器能安全唤醒等待方(见 interrupt.py 的 _wake)
            token.bind_loop(asyncio.get_running_loop())

        session.append("turn/start")
        permissions = self.ctx.get("permissions", None)
        if permissions is not None:
            permissions.pin(session)
        started = time.perf_counter()

        # agent/pre-step 是决定"接受哪些输入"的瀑布:监听器可以改写甚至拒绝。
        accepted = await self.ctx.waterfall("agent/pre-step", text, default=text)
        if accepted is None or (not str(accepted).strip() and not attachments):
            duration_ms = _since(started)
            session.append("turn/end", stopped="empty-turn", duration_ms=duration_ms)
            return RunResult(None, 0, session, "empty-turn", duration_ms)

        metadata = {"attachments": [dict(file) for file in attachments]} if attachments else {}
        session.append("user/message", text=str(accepted), **metadata)

        final_text: str | None = None
        steps = 0
        stopped = "max-steps"

        for index in range(1, self.max_steps + 1):
            reason = self._cancel_reason()
            if reason is not None:
                stopped = "cancelled"
                break

            steps = index
            self._step_started = time.perf_counter()
            await self._maybe_condense(session)
            request = await self._build_request(session)
            # 估算一次请求有多大,喂给用量表 —— 这是"发出去之前"唯一能拿到的数。
            # 工具 schema 也算进去,否则会稳定偏低(见 _estimated_tokens)。
            estimated = self._estimated_tokens(
                request.system or "", request.messages, request.tools
            )
            self._note_request_size(estimated)
            session.append(
                "step/start",
                model=getattr(self.ctx.llm.active, "model", None),
                index=index,
                context_tokens=estimated,
                # 校准后的估算才是"预算逻辑实际用的数"(见 token_meter 的自校准),
                # 所以也记进事件里,界面上能拿它和原始估算对照着看。
                context_tokens_calibrated=self._calibrated_tokens(estimated),
            )
            await self.ctx.emit("agent/step", session, index)

            # 边流边执行:流式阶段就派发出去的工具任务,在这里按顺序回收结果。
            prefetched: dict[str, asyncio.Task] = {}
            waited_before = self._retry_waited()
            model_started = time.perf_counter()
            try:
                result = await self._model_call(session, index, prefetched, request)
            except BaseException:
                await self._discard(prefetched)
                raise
            retry_ms = _ms(self._retry_waited() - waited_before)
            model_ms = max(0, _since(model_started) - retry_ms)

            if result is None:  # 流式途中被取消:这一步没有可提交的内容
                await self._discard(prefetched)
                self._end_step(
                    session, index, cancelled=True, model_ms=model_ms, retry_ms=retry_ms
                )
                stopped = "cancelled"
                break

            self._note_usage(result.usage)

            session.append(
                "assistant/message",
                text=result.text,
                reasoning=result.reasoning,
                model=getattr(self.ctx.llm.active, "model", None),
                usage=result.usage,
                tool_calls=[
                    {"id": call.id, "name": call.name, "arguments": call.arguments}
                    for call in result.tool_calls
                ],
                finish_reason=result.finish_reason,
            )

            if not result.tool_calls:
                # 思考型模型(Qwen3 系)可能只把内容放进 reasoning_content,
                # 非流式响应里它已经是完整回答 —— 别让 turn 以空回答收场。
                final_text = result.text or result.reasoning
                self._end_step(session, index, model_ms=model_ms, retry_ms=retry_ms)
                stopped = "final"
                break

            tools_started = time.perf_counter()
            cancelled_during_tools = await self._run_tools(
                session, result.tool_calls, prefetched
            )
            tools_ms = _since(tools_started)
            self._end_step(
                session,
                index,
                cancelled=cancelled_during_tools is not None,
                model_ms=model_ms,
                tools_ms=tools_ms,
                retry_ms=retry_ms,
            )
            if cancelled_during_tools is not None:
                stopped = "cancelled"
                break

        compaction = self.ctx.get("compaction", None)
        if stopped == "final" and not self._cancel_reason() and compaction is not None:
            if compaction.window_pressure(session):
                notices_before = len(session.events_of("command/result"))
                record = await self._maybe_condense(session)
                if record is None and len(session.events_of("command/result")) == notices_before:
                    session.append("command/result", text="上下文已达压缩阈值，但本次未完成压缩（可能没有足够可压缩的旧消息、已取消或摘要失败）。")
        duration_ms = _since(started)
        session.append("turn/end", stopped=stopped, duration_ms=duration_ms,
                       steps=steps, step_limit=self.max_steps)
        return RunResult(final_text, steps, session, stopped, duration_ms)

    # ---------------------------------------------------------------- 组装请求
    def system_for(self, session: Session | None = None) -> str:
        system = self.ctx.systemPrompt.render()
        permissions = self.ctx.get("permissions", None)
        if permissions is not None:
            state = permissions.snapshot(session)
            system += (f"\n\nCurrent sandbox: {state['sandbox']}. Approval policy: {state['approval']}. "
                       "Try operations within the current sandbox normally. If denied, request a strictly wider "
                       "sandbox_permissions value with justification for that single tool call; approval does not "
                       "change the standing session policy. A never policy rejects approval requests. "
                       "Reads and network are not isolated. Never claim a failed sandbox operation succeeded.")
        return system

    async def _build_request(self, session: Session) -> GenerateRequest:
        """按**当前**的日志投影组装请求。

        单独成函数是为了重试:重试时消息可能已经变了(压缩过),必须重建,
        不能把上一次那个请求原样再发一遍。
        """
        return GenerateRequest(
            system=self.system_for(session),
            messages=await self._history(session),
            tools=self.ctx.tools.schemas(),  # type: ignore[attr-defined]
            # 不钉死模型:模型名是**连接的属性**,由 adapter 持有,
            # 这样 /model 换完之后下一个 step 立刻生效。
            model=None,
        )

    async def _model_call(
        self,
        session: Session,
        index: int,
        prefetched: dict[str, asyncio.Task],
        request: GenerateRequest,
    ) -> GenerateResult | None:
        """跑一次模型调用;失败时在 **step 边界**上按策略重试。

        返回 ``None`` 表示"退避期间被取消" —— 交给上面按取消收尾,和流式途中被取消同路。

        每一步都刻意做对:
        * **先落日志再退避**(``note_scheduled``)—— 进程在等待期间被杀也说得清;
        * 重试前把上一次派发出去的工具任务**作废**(``_discard``),避免重复执行;
        * 上下文溢出不是原样重发,而是**先压缩再重建请求**;
        * 压不动(历史太短)就别硬试了,直接放弃并说清原因。
        """
        retry = self.ctx.get("retry", None)
        attempt = 1
        current = request
        overflow_retries = 0

        while True:
            try:
                result = await self._generate(session, current, prefetched)
            except LLMError as exc:
                if retry is None:
                    raise

                decision = retry.decide(exc, attempt)
                if not decision.will_retry:
                    retry.note_gave_up(session, index, decision)
                    raise

                # 上一次尝试可能已经派发过工具任务:作废,否则会重复执行。
                await self._discard(prefetched)

                if decision.is_overflow:
                    # 确认溢出最多恢复一次，避免持续压缩循环（包括无限网络重试模式）。
                    if overflow_retries >= 1:
                        retry.note_gave_up(session, index, decision, reason="上下文溢出压缩重试已达上限")
                        raise
                    overflow_retries += 1
                    record = await self._maybe_condense(session, force=True)
                    if record is None:
                        retry.note_gave_up(
                            session,
                            index,
                            decision,
                            reason="上下文超出上限,但没有可压缩的余量",
                        )
                        raise
                    current = await self._build_request(session)

                retry.note_scheduled(session, index, decision, exc)  # 先记账再退避
                retry.note_attempt()
                if not await retry.wait(decision.delay, self._interrupt):
                    return None  # 退避期间被取消
                attempt += 1
            else:
                if attempt > 1:
                    retry.note_recovered(session, index, attempt)
                return result

    # ---------------------------------------------------------------- 计费时钟
    def _end_step(self, session: Session, index: int, **extra: Any) -> None:
        """给 step 收尾,并统一记上它花了多久。

        时间拆成三段:**模型** / **工具** / **退避**。慢的时候才知道该往哪儿看 ——
        "这一步 12 秒"和"这一步 12 秒里 10 秒是在退避重试"是两件完全不同的事。
        """
        session.append(
            "step/end",
            index=index,
            duration_ms=_since(self._step_started),
            **extra,
        )

    def _retry_waited(self) -> float:
        """重试累计等待了多少秒(没装重试插件就是 0)。"""
        retry = self.ctx.get("retry", None)
        return float(getattr(retry, "waited_seconds", 0.0))

    # ---------------------------------------------------------------- 用量统计
    @property
    def _meter(self) -> Any:
        return self.ctx.get("tokenMeter", None)

    @staticmethod
    def _estimated_tokens(
        system: str, messages: Sequence[Message], tools: Sequence[Any] = ()
    ) -> int:
        """粗估这次请求的 token 数。

        刻意和压缩用**同一套估算**(``estimate_tokens``,2 字符/token,宁高估)——
        口径不一致的话会出现"压缩说没超预算、状态行说已经 90%"这种自相矛盾。

        **必须把工具 schema 算进来**:它是要发给模型的真实内容,而且一点都不小 ——
        实测三个工具的 schema 有 1194 字符(≈598 tok),占整个 prompt 的 45%。
        漏掉它会让估算稳定偏低 3.5 倍,连带把状态行的用量和压缩触发点一起带偏。
        """
        total = estimate_tokens(system or "")
        total += message_tokens(messages)
        if tools:
            total += estimate_tokens(_dump([schema.to_wire() for schema in tools]))
        return total

    def _note_request_size(self, tokens: int) -> None:
        meter = self._meter
        if meter is not None:
            meter.note_request(tokens)

    def _calibrated_tokens(self, tokens: int) -> int:
        """按用量表的校准系数修正估算(没校过就是原值)。"""
        meter = self._meter
        factor = getattr(meter, "factor", 1.0) if meter is not None else 1.0
        if not isinstance(factor, float) or factor <= 0:
            return tokens
        return int(tokens * factor)

    def _note_usage(self, usage: dict[str, Any] | None) -> None:
        """把 provider 报的 usage 记进表里 —— 这才是权威的"使用量"。"""
        meter = self._meter
        if meter is not None and usage:
            meter.note_response(usage)

    # ---------------------------------------------------------------- 取历史
    async def _history(self, session: Session) -> list[Message]:
        """取模型历史 —— 走 ``session/messages`` 投影瀑布。

        压缩就挂在这条接缝上:监听器看到的只是"喂给模型的这一份",**日志一个字不动**。
        对应 dsh 的 `session-projection` 子系统(dsh 有一整组包做这件事)。
        """
        derived = session.derive_messages()
        projected = await self.ctx.waterfall(
            "session/messages", session, derived, default=derived
        )
        return projected if isinstance(projected, list) else derived

    async def _maybe_condense(self, session: Session, *, force: bool = False) -> Any:
        """压力检查(或强制)压缩。没装压缩插件时返回 ``None``。

        为什么把触发点放在驱动器而不是投影里:投影必须是**纯函数**(可重放、可反复调用),
        而压缩要发起一次模型请求、还要往日志里写事件 —— 那是"决定",不是"投影"。
        """
        compaction = self.ctx.get("compaction", None)
        if compaction is None:
            return None
        try:
            if force:
                return await compaction.condense_now(session)
            return await compaction.maybe_condense(session)
        except Exception as exc:  # noqa: BLE001 —— 压不动不该拖垮这一轮
            # 持久化为 UI/终端可见的结果，不进入模型历史；不能只发无人监听的诊断事件。
            session.append("command/result", text=f"上下文压缩失败：{type(exc).__name__}: {exc}",
                           code=getattr(exc, "code", "compaction-failed"))
            await self.ctx.emit("compaction/failed", session, exc)
            return None

    # ---------------------------------------------------------------- 单步取回
    async def _generate(
        self, session: Session, request: GenerateRequest, prefetched: dict[str, asyncio.Task]
    ) -> GenerateResult | None:
        # A stalled stream may not emit another frame to observe cancellation.
        # Race only this conversation's model call against its own token.
        token = self._interrupt
        if token is None:
            return await self._generate_once(session, request, prefetched)
        if token.cancelled:
            return None
        pending = asyncio.create_task(self._generate_once(session, request, prefetched))
        cancelled = asyncio.create_task(token.wait())
        try:
            done, _ = await asyncio.wait({pending, cancelled}, return_when=asyncio.FIRST_COMPLETED)
            if pending in done:
                # Preserve an already completed response so committed tool calls
                # can be paired with their real results even when stop races it.
                return await pending
            return None
        finally:
            for task in (pending, cancelled):
                if not task.done():
                    task.cancel()
            await asyncio.gather(pending, cancelled, return_exceptions=True)

    async def _generate_once(
        self,
        session: Session,
        request: GenerateRequest,
        prefetched: dict[str, asyncio.Task],
    ) -> GenerateResult | None:
        """取一次模型结果。返回 ``None`` 表示流式途中被取消。

        ``prefetched`` 是出参:流式过程中一旦某个工具调用的参数补全,就在这里
        起一个后台任务开始执行 —— 模型接着说的话和工具的执行是**并行**的。
        """
        if not self.streaming:
            return await self.ctx.llm.generate(request)  # type: ignore[attr-defined]

        await self.ctx.emit(
            "agent/assistant-stream", session, AssistantStreamFrame("start")
        )
        stream = self.ctx.llm.stream(request)  # type: ignore[attr-defined]
        result: GenerateResult | None = None
        try:
            async for event in stream:
                if self._cancel_reason() is not None:
                    break
                if event.kind == "delta":
                    await self.ctx.emit(
                        "agent/assistant-stream",
                        session,
                        AssistantStreamFrame("chunk", event.text, event.reasoning),
                    )
                elif event.kind == "tool_call" and event.tool_call is not None:
                    await self.ctx.emit("agent/tool-ready", session, event.tool_call)
                    if self._can_prefetch(event.tool_call):
                        prefetched[event.tool_call.id] = asyncio.create_task(
                            self.ctx.tools.execute(  # type: ignore[attr-defined]
                                event.tool_call, session=session, cwd=self.cwd
                            )
                        )
                        # 只在实际派发时才报这个事件,UI 才不会撒谎
                        await self.ctx.emit(
                            "agent/tool-dispatched", session, event.tool_call
                        )
                elif event.kind == "done":
                    result = event.result
        finally:
            # 提前退出时必须显式关闭异步生成器,否则底层 SSE 连接要等 GC 才断开。
            await stream.aclose()
            await self.ctx.emit(
                "agent/assistant-stream", session, AssistantStreamFrame("end")
            )
        return result

    def _can_prefetch(self, call: ToolCall) -> bool:
        """这个调用能不能提前派发。

        需要审批的调用**不提前**:否则审批提示会和还在滚动的流式输出抢同一行终端,
        读起来一团糟;而且审批本来就要等人,提前问没有收益。
        """
        if self.ctx.get("permissions", None) is not None and call.arguments.get("sandbox_permissions"):
            return False
        if not self.early_tools or not self.ctx.tools.can_prefetch(call.name):
            return False
        service = self.ctx.get("approval", None)
        if service is None:
            return True
        return service.policy.reason_for(call, self.ctx.tools.permission_for(call.name)) is None

    # ---------------------------------------------------------------- 工具阶段
    async def _run_tools(
        self,
        session: Session,
        calls: Sequence[ToolCall],
        prefetched: dict[str, asyncio.Task] | None = None,
    ) -> str | None:
        """执行一批工具调用,返回取消原因(``None`` 表示全部跑完)。

        ``prefetched`` 里是流式阶段就已经开始执行的任务。它们的执行是并行的,
        但**写日志必须按顺序**:``assistant(tool_calls)`` 之后逐条对应结果,
        顺序乱了下一个请求就是非法的。这正是"并发执行、串行落账"。
        """
        prefetched = prefetched if prefetched is not None else {}

        for position, call in enumerate(calls):
            session.append(
                "tool/call",
                call_id=call.id,
                name=call.name,
                arguments=call.arguments,
            )
            reason = self._cancel_reason()
            if reason is not None:
                await self._settle_skipped(session, calls[position:], prefetched, reason)
                return reason

            task = prefetched.pop(call.id, None)
            outcome = (
                await self._await_task(task)
                if task is not None
                else await self.ctx.tools.execute(  # type: ignore[attr-defined]
                    call, session=session, cwd=self.cwd
                )
            )
            session.append(
                "tool/result",
                call_id=call.id,
                name=call.name,
                content=outcome.content,
                is_error=outcome.is_error,
                images=outcome.images,
            )

        if prefetched:  # 正常路径不会剩下;保险起见收掉
            await self._discard(prefetched)
        return None

    async def _settle_skipped(
        self,
        session: Session,
        skipped: Sequence[ToolCall],
        prefetched: dict[str, asyncio.Task],
        reason: str,
    ) -> None:
        """取消时给被跳过的调用补结果。

        **已经提前派发并且跑完了的,如实写它真实的结果** —— 不能写"未执行",
        因为副作用可能真的发生了(文件已经写了、命令已经跑了)。
        只有没跑完的才标成"已取消"。
        """
        for call in skipped:
            task = prefetched.pop(call.id, None)
            if task is not None and task.done() and not task.cancelled():
                outcome = await self._await_task(task)
            else:
                if task is not None:
                    task.cancel()
                outcome = ToolResult(f"已取消,未执行({reason})", is_error=True)
            session.append(
                "tool/result",
                call_id=call.id,
                name=call.name,
                content=outcome.content,
                is_error=outcome.is_error,
                images=outcome.images,
            )

    @staticmethod
    async def _await_task(task: asyncio.Task) -> ToolResult:
        """取回提前派发的任务结果;异常一律收敛成错误结果,不让 turn 崩掉。"""
        try:
            return await task
        except asyncio.CancelledError:
            return ToolResult("已取消", is_error=True)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(f"{type(exc).__name__}: {exc}", is_error=True)

    @staticmethod
    async def _discard(prefetched: dict[str, asyncio.Task]) -> None:
        """收掉已经派发、但本轮不再需要结果的任务(避免 "Task was destroyed" 噪声)。"""
        tasks = list(prefetched.values())
        prefetched.clear()
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


class Agent:
    """一个 agent 句柄:绑定会话,提供 ``run``。对应 dsh 里活的 Agent 对象。"""

    def __init__(self, ctx: Context, session: Session) -> None:
        self.ctx = ctx
        self.session = session
        self.id = session.id

    async def run(self, text: str, *, attachments=None) -> RunResult:
        # dsh 在这里把跑在当前 agent 作用域上的上下文(如 todo、goal)注入请求;
        # 迷你版直接委托给驱动器。
        try:
            options = {"attachments": attachments} if attachments else {}
            return await self.ctx.agentLoop.run(self.session, text, **options)  # type: ignore[attr-defined]
        except Exception as exc:
            # 循环抛出时日志可能停在半途(实测:模型接口 403 时留下了只有 turn/start
            # 没有 turn/end 的 turn)。在这里补一个结束事件 —— "每个 turn/start 都有
            # turn/end" 是日志的基本不变量,不该靠重启后的修复来兜。
            self.session.append(
                "turn/end",
                stopped="error",
                error=f"{type(exc).__name__}: {exc}",
                repaired=True,
            )
            raise


class AgentsService:
    """Agent 注册表。对应 ``ctx.agents``。"""

    def __init__(self, ctx: Context) -> None:
        self.ctx = ctx
        self._agents: dict[str, Agent] = {}

    def create(self, session: Session) -> Agent:
        agent = Agent(self.ctx, session)
        self._agents[agent.id] = agent
        return agent

    def get(self, agent_id: str) -> Agent | None:
        return self._agents.get(agent_id)

    def release(self, session: Session) -> None:
        """Drop a finished agent handle without affecting another session with the same ID."""
        agent = self._agents.get(session.id)
        if agent is not None and agent.session is session:
            self._agents.pop(session.id)


def plugin(
    max_steps: int = 64,
    cwd: Path | None = None,
    model: str | None = None,
    streaming: bool = True,
    early_tools: bool = True,
) -> Plugin:
    """挂上驱动器与 agent 注册表。依赖四个下游服务,顺序由内核推导。"""

    def apply(ctx: Context) -> None:
        ctx.provide(
            "agentLoop",
            AgentLoopService(ctx, max_steps, cwd, model, streaming, early_tools),
        )
        ctx.provide("agents", AgentsService(ctx))

    return Plugin(
        name="agent-loop",
        apply=apply,
        inject=("llm", "tools", "sessions", "systemPrompt"),
        description="turn/step 默认驱动器与 agent 注册表",
    )
