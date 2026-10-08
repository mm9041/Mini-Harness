"""子代理测试。

四件事要证明,每件都对应 dsh 里明说过的一条:

1. **隔离** —— 子代理的中间过程(prompt、工具输出、尝试)不进父会话的日志;
   父会话只多出"我派了个活给谁"和"结果是什么"。
   (dsh: "intermediate messages and tool traffic stay outside the parent conversation")
2. **拿回来的是结果,不是过程** —— 父代理看到的 tool 结果里只有子代理的最终回答和它的会话 id;
3. **失败不给部分成功** —— 子代理超步数/被取消时,父代理拿到的是**错误**;
4. **不越权** —— 子代理的工具调用照样过审批闸门,不能绕开父代理的权限档位。
"""

from __future__ import annotations

import unittest

from mini_harness.adapters.mock import ScriptedAdapter
from mini_harness.console import TracePrinter
from mini_harness.llm import ToolCall
from mini_harness.subagent import SubagentPolicy, SubagentService

from . import support
from .support import adapter_plugin_for as plugin_for
from .support import build_context_with as build_test_context
from functools import partial

# These execution tests explicitly allow the PowerShell tool.
build_test_context = partial(build_test_context, approval="allow")


def delegate(prompt: str = "数一下 .py 文件", title: str = "交给子代理") -> dict:
    return {
        "tool_calls": [
            {"name": "task", "arguments": {"description": title, "prompt": prompt}}
        ]
    }


class IsolationTests(unittest.IsolatedAsyncioTestCase):
    """子代理干活的过程,父会话里看不到。"""

    async def asyncSetUp(self) -> None:
        self.cwd = support.make_temp_dir()

    async def asyncTearDown(self) -> None:
        await support.remove_tree(self.cwd)

    def _ctx(self, script: list[dict], **overrides):
        self.adapter = ScriptedAdapter(script)
        return build_test_context(plugin_for(self.adapter), self.cwd, **overrides)

    async def test_child_traffic_stays_out_of_the_parent_log(self) -> None:
        ctx = self._ctx(
            [
                delegate(),  # 父 step 1:派活
                {"tool_calls": [{"name": "pwsh", "arguments": {"command": "echo 子代理在干活"}}]},
                {"text": "子代理的结论:20 个"},
                {"text": "父代理汇总:子代理说 20 个"},
            ]
        )
        parent = ctx.sessions.create()

        result = await ctx.agents.create(parent).run("统计一下")

        # ① 父会话里**只有**委派这件事与它的结果,没有子代理的任何内层事件
        types = [event.type for event in parent.events]
        self.assertIn("subagent/start", types)
        self.assertIn("subagent/end", types)
        self.assertEqual(types.count("tool/call"), 1)  # 只有那一次委派
        self.assertEqual(types.count("tool/result"), 1)
        for event in parent.events:
            if event.type == "tool/result":
                self.assertEqual(event.data["name"], "task")

        # ② 父代理的历史里没有子代理的工具输出
        joined = "\n".join(
            message.content or "" for message in parent.derive_messages()
        )
        self.assertNotIn("子代理在干活", joined)
        self.assertIn("20 个", joined)  # 只有结果进来了

        # ③ 父代理的第二次请求也没带上子代理的过程
        second = self.adapter.requests[-1]
        self.assertEqual([m.role for m in second.messages], ["user", "assistant", "tool"])

        self.assertEqual(result.stopped, "final")
        self.assertIn("20 个", result.text)

    async def test_the_child_gets_its_own_session_and_only_the_prompt(self) -> None:
        ctx = self._ctx(
            [
                delegate("只数 .py 文件"),
                {"text": "子代理:数完了"},
                {"text": "父代理:收到"},
            ]
        )
        parent = ctx.sessions.create()

        await ctx.agents.create(parent).run("统计一下")

        # 子代理的第一次请求:只有那条自包含的 prompt,没有父会话的任何内容
        child_request = self.adapter.requests[1]
        self.assertEqual([m.role for m in child_request.messages], ["user"])
        self.assertIn("只数 .py 文件", child_request.messages[0].content)

        # 子会话是独立的一份日志,并且是完整的(结构平衡)
        started = parent.events_of("subagent/start")[0]
        child_id = started.data["child_session"]
        self.assertNotEqual(child_id, parent.id)
        self.assertNotIn(child_id, ctx.sessions._live)
        child = ctx.sessions.open(ctx.sessions.root / f"{child_id}.jsonl")
        self.assertGreater(len(child.events), 0)
        self.assertEqual(
            len(child.events_of("turn/start")), len(child.events_of("turn/end"))
        )
        self.assertIn("subagent/end", [event.type for event in parent.events])

    async def test_result_is_labelled_and_reports_the_child_session(self) -> None:
        ctx = self._ctx(
            [delegate(title="数 .py 文件"), {"text": "结论:20"}, {"text": "父:收到"}]
        )
        parent = ctx.sessions.create()

        await ctx.agents.create(parent).run("统计一下")

        content = parent.events_of("tool/result")[0].data["content"]
        self.assertIn("数 .py 文件", content)
        self.assertIn("中间过程已保存至", content)
        self.assertIn(parent.events_of("subagent/start")[0].data["child_session"], content)


class DepthTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.cwd = support.make_temp_dir()

    async def asyncTearDown(self) -> None:
        await support.remove_tree(self.cwd)

    async def test_child_cannot_delegate_further(self) -> None:
        """默认只允许派一层:子代理再想派活会被明确拒绝,而不是无限套娃。"""
        adapter = ScriptedAdapter(
            [
                delegate(),  # 父派活
                delegate(),  # 子想再派 → 会被拒
                {"text": "子代理:不让再派,我自己做完了"},
                {"text": "父代理:收到"},
            ]
        )
        ctx = build_test_context(plugin_for(adapter), self.cwd, subagent_max_depth=1)
        parent = ctx.sessions.create()

        await ctx.agents.create(parent).run("统计一下")

        # ① 只创建了一个子会话(被拒的那次没有建会话)
        self.assertEqual(len(parent.events_of("subagent/start")), 1)

        # ② 拒绝记在**子会话**的日志里 —— 因为那是"子代理尝试再派"这件事发生的地方
        child_id = parent.events_of("subagent/start")[0].data["child_session"]
        child = ctx.sessions.open(ctx.sessions.root / f"{child_id}.jsonl")
        refusals = child.events_of("subagent/refused")
        self.assertEqual(len(refusals), 1)
        self.assertEqual(refusals[0].data["stopped"], "refused")
        self.assertIn("嵌套上限", refusals[0].data["reason"])

        # ③ 子代理自己看到了这个拒绝,并且没有因此崩掉
        last_word = child.events_of("assistant/message")[-1].data["text"]
        self.assertIn("自己做完了", last_word)

    async def test_depth_two_allows_one_more_level(self) -> None:
        adapter = ScriptedAdapter(
            [
                delegate(),  # 父
                delegate(),  # 子 → 允许再派一层
                {"text": "孙代理:做完了"},  # 孙
                {"text": "子代理:收到孙的结果"},  # 子
                {"text": "父代理:全部完成"},  # 父
            ]
        )
        ctx = build_test_context(plugin_for(adapter), self.cwd, subagent_max_depth=2)
        parent = ctx.sessions.create()

        result = await ctx.agents.create(parent).run("统计一下")

        self.assertEqual(len(parent.events_of("subagent/start")), 1)  # 父只派了一次
        self.assertEqual(result.text, "父代理:全部完成")


class FailureTests(unittest.IsolatedAsyncioTestCase):
    """失败返回错误,不返回部分成功。"""

    async def asyncSetUp(self) -> None:
        self.cwd = support.make_temp_dir()

    async def asyncTearDown(self) -> None:
        await support.remove_tree(self.cwd)

    async def test_child_hitting_max_steps_is_an_error(self) -> None:
        endless = [delegate()] + [
            {"tool_calls": [{"name": "pwsh", "arguments": {"command": "echo 继续"}}]}
            for _ in range(6)
        ]
        adapter = ScriptedAdapter(endless)
        ctx = build_test_context(
            plugin_for(adapter), self.cwd, max_steps=2, early_tools=False
        )
        parent = ctx.sessions.create()

        await ctx.agents.create(parent).run("派个活")

        outcome = parent.events_of("tool/result")[0].data
        self.assertTrue(outcome["is_error"], "没跑完就必须是错误")
        self.assertIn("步数上限", outcome["content"])

    async def test_cancellation_propagates_and_the_log_stays_consistent(self) -> None:
        """父代理的取消令牌是共享的:取消会传到子代理,两边日志都收得干净。"""
        adapter = ScriptedAdapter(
            [
                delegate(),
                {"tool_calls": [{"name": "pwsh", "arguments": {"command": "echo 子代理在干活"}}]},
                {"text": "子代理:其实已经没机会说了"},
                {"text": "父代理:也不会走到这里"},
            ]
        )
        ctx = build_test_context(plugin_for(adapter), self.cwd, subagent_max_depth=1)
        parent = ctx.sessions.create()

        def cancel_inside_the_child(call, context, result) -> None:  # noqa: ANN001
            if context.session is not None and context.session.id != parent.id:
                ctx.interrupt.request("子代理干活时按了 Ctrl+C")

        from mini_harness.kernel import MODE_EMIT

        ctx.on("tools/execute", cancel_inside_the_child, mode=MODE_EMIT)

        result = await ctx.agents.create(parent).run("派个活")

        self.assertEqual(result.stopped, "cancelled")
        outcome = parent.events_of("tool/result")[0].data
        self.assertTrue(outcome["is_error"])
        self.assertIn("取消", outcome["content"])

        # 两边日志都收得干净
        self.assertEqual(
            len(parent.events_of("turn/start")), len(parent.events_of("turn/end"))
        )
        ends = [
            event.data
            for event in parent.events
            if event.type == "subagent/end"
        ]
        self.assertEqual(len(ends), 1)
        self.assertEqual(ends[0]["stopped"], "cancelled")


class PermissionTests(unittest.IsolatedAsyncioTestCase):
    """子代理不能绕开审批 —— 权限是继承的,不是新建的。"""

    async def asyncSetUp(self) -> None:
        self.cwd = support.make_temp_dir()

    async def asyncTearDown(self) -> None:
        await support.remove_tree(self.cwd)

    async def test_child_write_still_goes_through_the_gate(self) -> None:
        adapter = ScriptedAdapter(
            [
                {
                    "tool_calls": [
                        {
                            "name": "task",
                            "arguments": {
                                "description": "写文件",
                                "prompt": "把 hi 写进 note.txt",
                            },
                        }
                    ]
                },
                {
                    "tool_calls": [
                        {
                            "name": "write",
                            "arguments": {"file_path": "note.txt", "content": "hi"},
                        }
                    ]
                },
                {"text": "子代理:被拦了"},
                {"text": "父代理:知道"},
            ]
        )
        ctx = build_test_context(plugin_for(adapter), self.cwd, approval="ask")
        parent = ctx.sessions.create()

        await ctx.agents.create(parent).run("派个活")

        self.assertFalse((self.cwd / "note.txt").exists(), "子代理不该绕过审批写文件")
        # 父会话里看不到子代理的内部,所以去子会话里查那条被拒的工具结果
        child_id = parent.events_of("subagent/start")[0].data["child_session"]
        child = ctx.sessions.open(ctx.sessions.root / f"{child_id}.jsonl")
        denied = child.events_of("tool/result")[0].data
        self.assertTrue(denied["is_error"])
        self.assertIn("审批策略拒绝", denied["content"])


class TraceTests(unittest.IsolatedAsyncioTestCase):
    """子代理的轨迹要看得见,但**缩进**着看。"""

    async def asyncSetUp(self) -> None:
        self.cwd = support.make_temp_dir()

    async def asyncTearDown(self) -> None:
        await support.remove_tree(self.cwd)

    async def test_child_trace_is_indented(self) -> None:
        adapter = ScriptedAdapter(
            [
                delegate(title="数文件"),
                {"tool_calls": [{"name": "pwsh", "arguments": {"command": "echo 子壳"}}]},
                {"text": "子代理的结论"},
                {"text": "父代理的汇总"},
            ]
        )
        ctx = build_test_context(plugin_for(adapter), self.cwd)
        output: list[str] = []
        printer = TracePrinter(write=output.append)
        dispose = printer.attach(ctx)
        session = ctx.sessions.create()
        printer.observe(session)
        try:
            await ctx.agents.create(session).run("统计")
        finally:
            dispose()

        text = "".join(output)
        self.assertIn("⤷ 委派子代理", text)
        self.assertIn("⤶ 子代理完成", text)
        # 子代理的 step 行被缩进了(父的没有)
        child_lines = [line for line in text.splitlines() if line.startswith("  ") and "[step" in line]
        parent_lines = [line for line in text.splitlines() if line.startswith("[step")]
        self.assertTrue(child_lines, "子代理的 step 应当缩进输出")
        self.assertTrue(parent_lines, "父代理的 step 不缩进")


class ServiceUnitTests(unittest.IsolatedAsyncioTestCase):
    """服务本身的边角,不经过模型。"""

    async def asyncSetUp(self) -> None:
        self.cwd = support.make_temp_dir()

    async def asyncTearDown(self) -> None:
        await support.remove_tree(self.cwd)

    async def test_refusal_does_not_create_a_session(self) -> None:
        from mini_harness.adapters.mock import scripted_plugin

        ctx = build_test_context(scripted_plugin([{"text": "x"}]), self.cwd)
        service: SubagentService = ctx.subagents
        before = len(ctx.sessions._live)

        # 手工把深度顶到上限
        from mini_harness.subagent import _DEPTH

        token = _DEPTH.set(service.policy.max_depth)
        try:
            result = await service.run("再派一层", parent_session=None)
        finally:
            _DEPTH.reset(token)

        self.assertEqual(result.stopped, "refused")
        self.assertIn("嵌套上限", result.note)
        self.assertEqual(len(ctx.sessions._live), before, "被拒时不该建会话")

    async def test_render_truncates_a_huge_child_answer(self) -> None:
        from mini_harness.adapters.mock import scripted_plugin

        ctx = build_test_context(scripted_plugin([{"text": "x"}]), self.cwd)
        ctx.subagents.policy.result_max_chars = 50
        from mini_harness.subagent import SubagentResult

        rendered = ctx.subagents.render(
            SubagentResult(title="T", text="答" * 200, session_id="s1", steps=1)
        )

        self.assertIn("子代理输出被截断", rendered)
        self.assertLess(len(rendered), 200)

    async def test_policy_defaults(self) -> None:
        policy = SubagentPolicy()
        self.assertEqual(policy.max_depth, 1)
        self.assertGreater(policy.result_max_chars, 0)


if __name__ == "__main__":
    unittest.main()
