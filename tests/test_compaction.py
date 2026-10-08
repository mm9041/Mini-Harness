"""上下文压缩测试。

三条主线:

1. **投影是纯函数**:`project_messages()` 不改日志、可反复调用,且**边界不能切碎
   assistant(tool_calls) + tool 结果的配对**(否则请求直接非法);
2. **摘要是决定,必须落日志**:压缩会发起一次模型请求并写一条 ``compaction`` 事件 ——
   所以续跑/重放之后压缩仍然生效;
3. **裁剪只动投影**:工具结果的完整原文永远留在日志里。
"""

from __future__ import annotations

import unittest
from unittest import mock

from mini_harness.adapters.mock import ScriptedAdapter, scripted_plugin
from mini_harness.compaction import (
    CompactionPolicy,
    CompactionService,
    CompactionBusyError,
    estimate_tokens,
    project_messages,
    prune_tool_result,
)
from mini_harness.llm import Message, ToolCall
from mini_harness.llm import GenerateResult

from . import support
from .support import adapter_plugin_for as plugin_for
from .support import build_context_with as build_test_context


def msg(role: str, content: str, seq: int, **kwargs) -> Message:
    return Message(role=role, content=content, source_seq=seq, **kwargs)


def long_session(session, turns: int = 6, size: int = 400) -> None:
    """直接往日志里灌若干轮 —— 不需要模型参与。"""
    for index in range(turns):
        session.append("turn/start")
        session.append("user/message", text=f"第 {index} 轮的问题:" + "问" * size)
        session.append("step/start", index=1)
        session.append(
            "assistant/message",
            text=f"第 {index} 轮的回答:" + "答" * size,
            tool_calls=[],
        )
        session.append("step/end", index=1)
        session.append("turn/end", stopped="final")


class EstimateTests(unittest.TestCase):
    def test_rough_but_conservative(self) -> None:
        self.assertGreater(estimate_tokens(""), 0)
        self.assertLess(estimate_tokens("a" * 100), estimate_tokens("a" * 200))
        # 按 2 字符/token 折中 —— 宁高估不低估
        self.assertEqual(estimate_tokens("a" * 100), 51)


class PruneTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = CompactionPolicy(
            prune_tool_result_chars=200, head_chars=50, tail_chars=30
        )

    def test_oversized_tool_result_is_pruned_with_a_marker(self) -> None:
        original = msg("tool", "H" * 50 + "M" * 500 + "T" * 30, seq=1)

        pruned = prune_tool_result(original, self.policy)

        self.assertLess(len(pruned.content), len(original.content))
        self.assertIn("中间省略", pruned.content)
        self.assertIn("原文仍在会话日志里", pruned.content)  # 关键:告诉模型原文没丢
        self.assertEqual(pruned.tool_call_id, original.tool_call_id)
        self.assertEqual(pruned.source_seq, original.source_seq)
        self.assertEqual(original.content, "H" * 50 + "M" * 500 + "T" * 30)  # 没被改

    def test_small_result_and_non_tool_messages_are_untouched(self) -> None:
        small = msg("tool", "小", seq=1)
        self.assertIs(prune_tool_result(small, self.policy), small)
        assistant = msg("assistant", "x" * 5000, seq=2)
        self.assertIs(prune_tool_result(assistant, self.policy), assistant)


class ProjectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = CompactionPolicy(prune_tool_result_chars=10**9)

    def test_without_records_only_pruning_happens(self) -> None:
        messages = [msg("user", "问题", 1), msg("assistant", "回答", 2)]
        self.assertEqual(project_messages(messages, [], self.policy), messages)

    def test_record_replaces_the_covered_prefix_with_one_summary(self) -> None:
        messages = [
            msg("user", "老问题", 1),
            msg("assistant", "老回答", 2),
            msg("user", "新问题", 3),
            msg("assistant", "新回答", 4),
        ]
        records = [{"summary": "之前聊过老问题", "covers_upto_seq": 2}]

        projected = project_messages(messages, records, self.policy)

        self.assertEqual(len(projected), 3)
        self.assertIn("更早的对话已压缩", projected[0].content)
        self.assertIn("之前聊过老问题", projected[0].content)
        self.assertEqual([m.content for m in projected[1:]], ["新问题", "新回答"])

    def test_boundary_never_leaves_an_orphan_tool_message(self) -> None:
        """边界落进 assistant(tool_calls) 与它的工具结果之间时,整个组让给摘要。

        否则投影会以 tool 消息开头,而它对应的 assistant 调用已经被压进摘要 ——
        上游会直接以"tool 消息没有对应的 tool_calls"拒绝这个请求。
        """
        messages = [
            msg("user", "早期问题", 1),
            msg(
                "assistant",
                None,
                2,
                tool_calls=[ToolCall(id="c1", name="shell", arguments={"command": "ls"})],
            ),
            msg("tool", "工具输出", 3, tool_call_id="c1"),
            msg("user", "后来又问", 4),
        ]
        records = [{"summary": "摘要", "covers_upto_seq": 2}]  # 边界刻意落在组中间

        projected = project_messages(messages, records, self.policy)

        self.assertNotEqual(projected[0].role, "tool")
        self.assertEqual([m.content for m in projected[1:]], ["后来又问"])

    def test_spilled_result_is_not_pruned_twice(self) -> None:
        """已经外溢过的内容不要再叠一层裁剪。

        spill 与 pruner 是两个策略,各按自己的阈值裁同一份内容会互相打架:
        外溢选好的头部被再砍一刀,还多出一个含义重复的省略标记
        (实测:4216 字符的预览被再裁成 2036)。这种情况只保留头部 + locator。
        这条测试同时是**跨模块格式契约**的看门人:spill 的 locator 写法一改,
        这里就认不出来,断言会红。
        """
        from mini_harness.spill import LocalSpillStore, SpillPolicy
        from mini_harness.tools import ToolResult

        store = LocalSpillStore(support.make_temp_dir())
        self.addCleanup(
            lambda: __import__("shutil").rmtree(store.root, ignore_errors=True)
        )
        spilled, record = SpillPolicy().apply(
            ToolResult("第 1 行:内容\n" * 3000),
            store=store,
            session_id="s",
            hint="read_file",
        )
        self.assertGreater(len(spilled.content), 4000)  # 确实超过 pruner 阈值

        pruned = prune_tool_result(
            msg("tool", spilled.content, seq=3, tool_call_id="c1"),
            CompactionPolicy(prune_tool_result_chars=100, head_chars=200, tail_chars=100),
        )

        self.assertIn(record.locator, pruned.content)
        self.assertEqual(pruned.content.count("省略"), 1, "不该出现两层省略标记")
        self.assertNotIn("原文仍在会话日志里", pruned.content)

    def test_plain_result_still_gets_the_old_treatment(self) -> None:
        """没有 locator 的内容照旧裁成头尾(别把协同修成例外)。"""
        policy = CompactionPolicy(
            prune_tool_result_chars=100, head_chars=20, tail_chars=10
        )
        pruned = prune_tool_result(msg("tool", "H" * 500 + "T" * 50, seq=1), policy)

        self.assertIn("原文仍在会话日志里", pruned.content)

    def test_recent_tool_results_keep_their_pairing(self) -> None:
        """裁剪必须保住配对:被裁的 tool 消息还是那条消息,不能变成别的角色或消失。"""
        policy = CompactionPolicy(prune_tool_result_chars=50, head_chars=10, tail_chars=10)
        messages = [
            msg("user", "跑个命令", 1),
            msg(
                "assistant",
                None,
                2,
                tool_calls=[ToolCall(id="c1", name="shell", arguments={"command": "ls"})],
            ),
            msg("tool", "x" * 500, 3, tool_call_id="c1"),
            msg("assistant", "看完了", 4),
        ]

        projected = project_messages(messages, [], policy)

        roles = [m.role for m in projected]
        self.assertEqual(roles, ["user", "assistant", "tool", "assistant"])
        self.assertEqual(projected[2].tool_call_id, "c1")
        self.assertEqual(
            [call.id for call in projected[1].tool_calls], ["c1"]
        )


class ServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.cwd = support.make_temp_dir()

    async def asyncTearDown(self) -> None:
        await support.remove_tree(self.cwd)

    def _ctx(self, script: list[dict], **overrides):
        self.adapter = ScriptedAdapter(script)
        # 默认值与 overrides 必须**合并**再展开:写成 `f(x=1, **overrides)` 时,
        # overrides 里再出现 x 就是 "got multiple values"(踩过两次)。
        settings = {
            "max_history_tokens": 2000,
            "keep_recent_messages": 4,
            **overrides,
        }
        return build_test_context(plugin_for(self.adapter), self.cwd, **settings)

    async def test_default_uses_window_not_8000_token_limit(self):
        ctx = self._ctx([], max_history_tokens=0, keep_recent_messages=0,
                        context_window=100000)
        session = ctx.sessions.create()
        long_session(session, turns=10, size=1000)
        self.assertIsNone(await ctx.compaction.maybe_condense(session))
        self.assertEqual(self.adapter.requests, [])

    async def test_retention_uses_tokens_not_message_count(self):
        ctx = self._ctx([{"text": "摘要"}], max_history_tokens=0,
                        keep_recent_messages=0, context_window=10000)
        session = ctx.sessions.create()
        long_session(session, turns=12, size=1000)
        record = await ctx.compaction.maybe_condense(session)
        self.assertIsNotNone(record)
        # 16% of 10000 = 1600 tokens: four roughly 500-token messages.
        self.assertEqual(len(ctx.compaction.project(session)), 5)

    async def test_manual_default_keeps_latest_unit(self):
        ctx = self._ctx([{"text": "摘要"}], max_history_tokens=0, keep_recent_messages=0)
        session = ctx.sessions.create()
        session.append("user/message", text="旧事实" * 300)
        session.append("user/message", text="继续")
        self.assertIsNotNone(await ctx.compaction.condense_now(session))
        self.assertEqual(ctx.compaction.project(session)[-1].content, "继续")

    async def test_summary_replays_full_content_and_tool_arguments(self):
        ctx = self._ctx([{"text": "摘要"}])
        messages = [msg("user", "x" * 9000 + "重要尾部", 1),
                    msg("assistant", None, 2, tool_calls=[ToolCall("c", "shell", {"command": "pwd"})]),
                    msg("tool", "result", 3, tool_call_id="c")]
        await ctx.compaction._summarize(messages)
        request = self.adapter.requests[-1]
        self.assertEqual(request.messages[:-1], messages)
        self.assertIn("重要尾部", request.messages[0].content)
        self.assertIn("## Pending Jobs", request.messages[-1].content)

    async def test_bad_summary_never_replaces_history(self):
        ctx = self._ctx([])
        session = ctx.sessions.create()
        long_session(session, turns=6, size=400)
        original = ctx.compaction.project(session)
        for result in (GenerateResult(text="x" * 20000),
                       GenerateResult(text="partial", finish_reason="length"),
                       GenerateResult(reasoning="only reasoning"),
                       GenerateResult(text="summary", tool_calls=[ToolCall("c", "shell")])):
            with self.subTest(result=result), mock.patch.object(self.adapter, "generate", return_value=result):
                with self.assertRaises(RuntimeError):
                    await ctx.compaction.condense_now(session)
            self.assertEqual(session.compactions(), [])
            self.assertEqual(ctx.compaction.project(session), original)

    async def test_compaction_lock_and_injected_tail(self):
        import asyncio
        ctx = self._ctx([])
        session = ctx.sessions.create()
        long_session(session, turns=6, size=400)
        started, finish = asyncio.Event(), asyncio.Event()
        async def summarize(messages):
            started.set()
            await finish.wait()
            return "摘要"
        with mock.patch.object(ctx.compaction, "_summarize", side_effect=summarize):
            first = asyncio.create_task(ctx.compaction.condense_now(session))
            await started.wait()
            with self.assertRaises(CompactionBusyError):
                await ctx.compaction.condense_now(session)
            self.assertIsNone(await ctx.compaction.maybe_condense(session))
            session.append("user/message", text="新注入的要求")
            finish.set()
            self.assertIsNotNone(await first)
        self.assertEqual(len(session.compactions()), 1)
        self.assertEqual(ctx.compaction.project(session)[-1].content, "新注入的要求")

    async def test_output_reservation_and_headroom_cap_threshold(self):
        ctx = self._ctx([], context_window=10000,
                        compaction_reserved_completion_tokens=2000,
                        compaction_headroom_tokens=1000)
        self.assertEqual(ctx.compaction._budgets(), (7000, 1280))

    async def test_full_request_budget_counts_envelope_exactly_once(self):
        ctx = self._ctx([], context_window=10000,
                        compaction_reserved_completion_tokens=2000,
                        compaction_headroom_tokens=1000)
        # 使用真实工具表和估算器，不 mock token 总数。headroom 分支决定阈值。
        with mock.patch.object(ctx.systemPrompt, "render", return_value="system instructions"):
            overhead = ctx.agentLoop._estimated_tokens(ctx.systemPrompt.render(), [], ctx.tools.schemas())
            threshold, _ = ctx.compaction._budgets()
            self.assertEqual(threshold, 7000)
            self.assertGreater(overhead, 0)
            self.assertLess(overhead, threshold)
            for expected_total in (6999, 7000):
                session = ctx.sessions.create()
                session.append("user/message", text="x" * (2 * (expected_total - overhead - 1)))
                actual = ctx.agentLoop._estimated_tokens(ctx.systemPrompt.render(),
                    ctx.compaction.project(session), ctx.tools.schemas())
                self.assertEqual(actual, expected_total)
                self.assertEqual(ctx.compaction.window_pressure(session), expected_total >= threshold)

    async def test_auto_failure_is_visible_without_polluting_model_history(self):
        from mini_harness.console import format_event
        from mini_harness.webui import ui_message
        ctx = self._ctx([{"text": "x" * 20000}, {"text": "正常回答"}])
        session = ctx.sessions.create()
        long_session(session, turns=6, size=400)
        result = await ctx.agents.create(session).run("继续")
        self.assertEqual(result.text, "正常回答")
        notices = session.events_of("command/result")
        self.assertEqual(len(notices), 1)
        self.assertIn("摘要未缩小", notices[0].data["text"])
        self.assertIn("摘要未缩小", format_event(notices[0]))
        self.assertEqual(ui_message(notices[0])["kind"], "command-result")
        self.assertFalse(any("摘要未缩小" in (m.content or "") for m in session.derive_messages()))
        self.assertEqual(session.compactions(), [])

    async def test_tools_only_arguments_are_counted(self):
        ctx = self._ctx([{"text": "摘要"}], max_history_tokens=1000, keep_recent_messages=1)
        session = ctx.sessions.create()
        session.append("assistant/message", text=None, tool_calls=[
            {"id": "c", "name": "write_file", "arguments": {"content": "x" * 10000}}])
        session.append("tool/result", call_id="c", name="write_file", content="ok")
        session.append("user/message", text="继续")
        self.assertIsNotNone(await ctx.compaction.maybe_condense(session))

    async def test_tool_heavy_tail_still_compacts(self) -> None:
        """尾部是一长串工具结果时也必须能压 —— 这是工具型会话最常见的形状。

        回归测试:早先的边界规则是"往前推直到不在工具结果中间",尾部一长串 tool 消息时
        会一直推到列表末尾,于是**压缩静默失效**(实测在真实会话里一条 compaction 都没有)。
        """
        ctx = self._ctx([{"text": "摘要"}])
        session = ctx.sessions.create()
        for index in range(4):
            session.append("turn/start")
            session.append("user/message", text=f"第 {index} 轮" + "问" * 300)
            session.append("step/start", index=1)
            session.append(
                "assistant/message",
                text=None,
                tool_calls=[
                    {"id": f"c{index}a", "name": "shell", "arguments": {"command": "ls"}},
                    {"id": f"c{index}b", "name": "shell", "arguments": {"command": "pwd"}},
                ],
            )
            session.append("tool/result", call_id=f"c{index}a", name="shell", content="输出" * 200, is_error=False)
            session.append("tool/result", call_id=f"c{index}b", name="shell", content="输出" * 200, is_error=False)
            session.append("step/end", index=1)
            session.append("turn/end", stopped="final")

        record = await ctx.compaction.maybe_condense(session)

        self.assertIsNotNone(record, "尾部全是工具结果时不该压不了")
        projected = ctx.compaction.project(session)
        # 投影仍然合法:开头的消息不是孤儿 tool 消息
        self.assertNotEqual(projected[0].role, "tool")
        # 配对不破:被保留的 assistant(tool_calls) 都有对应结果
        announced = [
            call.id for message in projected if message.role == "assistant" for call in message.tool_calls
        ]
        answered = [m.tool_call_id for m in projected if m.role == "tool"]
        self.assertEqual(sorted(announced), sorted(answered))

    async def test_pressure_check_uses_the_calibrated_estimate(self) -> None:
        """预算判断要用**校准后**的估算。

        2 字符/token 是折中值,英文场景会高估近一倍 —— 不加校准,压缩会发生得过早。
        这条测试刻意造一个"原始估算超预算、校准之后不超"的局面。
        """
        text = "问" * 2400
        raw = estimate_tokens(text) * 8  # 8 条同样长的消息
        budget = int(raw * 0.85)  # 卡在"校准后(×0.75)"与"原始(×1.0)"之间

        def make(**overrides):
            ctx = self._ctx([{"text": "不该被调用"}], max_history_tokens=budget, **overrides)
            session = ctx.sessions.create()
            for _ in range(8):
                session.append("user/message", text=text)
            return ctx, session

        # ① 没校准(系数 1.0):原始估算超预算 → 触发
        plain_ctx, plain_session = make()
        self.assertIsNotNone(await plain_ctx.compaction.maybe_condense(plain_session))

        # ② 校准:样本 0.5 → 系数 = 0.5×1.0 + 0.5×0.5 = 0.75 → 压回预算之内 → 不触发
        calibrated_ctx, calibrated_session = make()
        calibrated_ctx.tokenMeter.note_request(1000)
        calibrated_ctx.tokenMeter.note_response({"prompt_tokens": 500})
        self.assertEqual(calibrated_ctx.tokenMeter.factor, 0.75)
        self.assertIsNone(
            await calibrated_ctx.compaction.maybe_condense(calibrated_session)
        )

    async def test_auto_skips_a_pointless_summary(self) -> None:
        """要压的内容太少时,自动驾驶不压 —— 摘要很可能比原文长,白花一次模型调用。

        (这个数是实测出来的:3 轮短对话压出来 ≈72 tok → ≈132 tok,反而更大。)
        """
        ctx = self._ctx([{"text": "不该被调用"}])
        session = ctx.sessions.create()
        long_session(session, turns=8, size=2)  # 条数够多,但内容极短

        record = await ctx.compaction.maybe_condense(session)

        self.assertIsNone(record)
        self.assertEqual(self.adapter.requests, [])

    async def test_manual_compact_ignores_the_size_guard(self) -> None:
        """手动 /compact 是用户的明确意图,照做。"""
        ctx = self._ctx([{"text": "手动摘要"}])
        session = ctx.sessions.create()
        long_session(session, turns=8, size=2)

        record = await ctx.compaction.condense_now(session)

        self.assertIsNotNone(record)

    async def test_no_op_below_pressure(self) -> None:
        ctx = self._ctx([{"text": "不该被调用"}])
        session = ctx.sessions.create()
        session.append("user/message", text="很短")

        record = await ctx.compaction.maybe_condense(session)

        self.assertIsNone(record)
        self.assertEqual(session.compactions(), [])
        self.assertEqual(self.adapter.requests, [])  # 没有多花一次模型调用

    async def test_pressure_triggers_one_summary_that_is_logged(self) -> None:
        ctx = self._ctx([{"text": "这是摘要"}])
        session = ctx.sessions.create()
        long_session(session, turns=6, size=400)
        before = len(session.events)

        record = await ctx.compaction.maybe_condense(session)

        self.assertIsNotNone(record)
        self.assertGreater(record.saved_tokens, 0)
        self.assertEqual([event.type for event in session.events[before:]],
                         ["compaction/start", "compaction/usage", "compaction", "compaction/end"])
        self.assertEqual(session.events[-1].data['status'], 'completed')
        self.assertEqual(len({event.data['operation_id'] for event in session.events[before:]}), 1)
        self.assertEqual(session.compactions()[-1]["summary"], "这是摘要")

        # 摘要复用主请求前缀，并把压缩指令放在最后。
        summarizer = self.adapter.requests[-1]
        self.assertEqual(summarizer.system, ctx.systemPrompt.render())
        self.assertEqual(summarizer.tools, ctx.tools.schemas())
        self.assertIn("压缩成一份交接摘要", summarizer.messages[-1].content)

    async def test_condense_now_ignores_the_budget(self) -> None:
        ctx = self._ctx([{"text": "手动摘要"}])
        session = ctx.sessions.create()
        long_session(session, turns=3, size=50)

        record = await ctx.compaction.condense_now(session)

        self.assertIsNotNone(record)
        self.assertEqual(session.compactions()[-1]["summary"], "手动摘要")

    async def test_compaction_shrinks_what_the_model_sees(self) -> None:
        """最终判据:压完之后,请求里带的是摘要 + 近期消息,不是全部历史。"""
        ctx = self._ctx([{"text": "摘要正文"}, {"text": "答"}])
        session = ctx.sessions.create()
        long_session(session, turns=6, size=400)

        await ctx.agents.create(session).run("再来一轮")

        sent = self.adapter.requests[-1].messages
        joined = "\n".join(message.content or "" for message in sent)
        self.assertIn("更早的对话已压缩", joined)
        self.assertIn("摘要正文", joined)
        # 第一轮的原话已经不在请求里了(它进了摘要)
        self.assertNotIn("第 0 轮的问题", joined)
        self.assertLess(len(sent), len(session.derive_messages()))

    async def test_summary_survives_resume(self) -> None:
        """压缩是"不可重建的决定",所以必须靠日志跨进程生效。"""
        ctx = self._ctx([{"text": "持久化的摘要"}])
        session = ctx.sessions.create()
        long_session(session, turns=6, size=400)
        await ctx.compaction.condense_now(session)
        path = ctx.sessions.save(session, self.cwd / "compacted.jsonl")

        reopened = build_test_context(scripted_plugin([{"text": "x"}]), self.cwd)
        resumed = reopened.sessions.open(path)

        self.assertEqual(len(resumed.compactions()), 1)
        projected = reopened.compaction.project(resumed)
        self.assertIn("持久化的摘要", projected[0].content)
        self.assertLess(len(projected), len(resumed.derive_messages()))

    async def test_second_compaction_covers_the_first(self) -> None:
        """再压一次不会把旧摘要叠成两条 —— 后一次是在前一次的投影之上做的。"""
        ctx = self._ctx([{"text": "第一次摘要"}, {"text": "第二次摘要"}])
        session = ctx.sessions.create()
        long_session(session, turns=6, size=400)

        await ctx.compaction.condense_now(session)
        long_session(session, turns=6, size=400)
        await ctx.compaction.condense_now(session)

        projected = ctx.compaction.project(session)
        summary_messages = [
            message for message in projected if "更早的对话已压缩" in (message.content or "")
        ]
        self.assertEqual(len(summary_messages), 1)
        # 第二次摘要把它自己看到的内容(含第一次摘要)都收进去了
        self.assertIn("第二次摘要", summary_messages[0].content)

    async def test_failure_does_not_break_the_turn(self) -> None:
        """摘不出内容(模型返回空)时,这一轮不该挂掉,只发一个 compaction/failed。"""
        ctx = self._ctx([{"text": ""}, {"text": "正常回答"}])
        session = ctx.sessions.create()
        long_session(session, turns=6, size=400)

        result = await ctx.agents.create(session).run("再来一轮")

        self.assertEqual(result.stopped, "final")
        self.assertEqual(result.text, "正常回答")
        self.assertEqual(session.compactions(), [])

    async def test_disabled_compaction_sends_everything(self) -> None:
        ctx = self._ctx([{"text": "答"}], compaction=False)
        session = ctx.sessions.create()
        long_session(session, turns=6, size=400)

        await ctx.agents.create(session).run("再来一轮")

        sent = self.adapter.requests[-1].messages
        self.assertNotIn(
            "更早的对话已压缩",
            "\n".join(message.content or "" for message in sent),
        )

    async def test_service_is_optional(self) -> None:
        """没装压缩插件时,驱动器不该因为找不到服务而报错。"""
        ctx = build_test_context(scripted_plugin([{"text": "答"}]), self.cwd, compaction=False)
        self.assertIsNone(ctx.get("compaction", None))

        result = await ctx.agents.create(ctx.sessions.create()).run("问一句")

        self.assertEqual(result.stopped, "final")


if __name__ == "__main__":
    unittest.main()
