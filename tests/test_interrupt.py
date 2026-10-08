"""取消 / 中断测试。

三件事要证明:

1. 令牌本身的语义(首次 request 为真、重复为假、reset 可复用、wait 能唤醒);
2. 驱动器在检查点停下,并且**会话日志依然合法** —— 每条 ``tool/call`` 都必须有对应的
   ``tool/result``,否则下一个请求里的 ``assistant(tool_calls)`` 没有结果,协议上就是
   非法的,整个会话就废了;
3. 长跑的工具能被立刻杀掉,而不是等超时、等命令自己结束。
"""

from __future__ import annotations

import asyncio
import time
import threading
import unittest

from mini_harness.adapters.mock import scripted_plugin
from mini_harness.builtin_tools.shell import make_shell_tool
from mini_harness.interrupt import InterruptService
from mini_harness.kernel import MODE_EMIT, MODE_WATERFALL
from mini_harness.llm import ToolCall
from mini_harness.tools import ToolCallContext

from . import support
from .support import build_context_with as build_test_context
from functools import partial

# These execution tests explicitly allow the PowerShell tool.
build_test_context = partial(build_test_context, approval="allow")


class InterruptServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_cross_thread_request_wakes_bound_loop(self) -> None:
        token = InterruptService()
        token.bind_loop(asyncio.get_running_loop())
        for reason in ('来自子线程', '第二轮'):
            token.reset()
            waiter = asyncio.create_task(token.wait())
            await asyncio.sleep(0)
            thread = threading.Thread(target=token.request, args=(reason,))
            thread.start()
            self.assertEqual(await asyncio.wait_for(waiter, 2), reason)
            thread.join(timeout=2)
            self.assertFalse(thread.is_alive())

    async def test_queued_cross_thread_wake_does_not_survive_reset(self) -> None:
        token = InterruptService()
        waiter = asyncio.create_task(token.wait())
        await asyncio.sleep(0)
        # Keep this loop occupied until reset, so the queued wake runs afterward.
        thread = threading.Thread(target=token.request, args=('previous turn',))
        thread.start()
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        token.reset()
        await asyncio.sleep(0)
        self.assertFalse(waiter.done())
        token.request('current turn')
        self.assertEqual(await asyncio.wait_for(waiter, 2), 'current turn')

    def test_request_semantics(self) -> None:
        token = InterruptService()

        self.assertFalse(token.cancelled)
        self.assertTrue(token.request("第一次"))
        self.assertFalse(token.request("第二次"))  # 重复请求返回 False
        self.assertTrue(token.cancelled)
        self.assertEqual(token.reason, "第一次")  # 保留第一次的原因

        token.reset()
        self.assertFalse(token.cancelled)
        self.assertIsNone(token.reason)
        self.assertTrue(token.request("复位后再来"))

    async def test_wait_wakes_up(self) -> None:
        token = InterruptService()

        waiter = asyncio.ensure_future(token.wait())
        await asyncio.sleep(0)
        self.assertFalse(waiter.done())

        token.request("工具跑到一半被取消")
        self.assertEqual(await asyncio.wait_for(waiter, timeout=2), "工具跑到一半被取消")

    async def test_wait_returns_immediately_when_already_cancelled(self) -> None:
        token = InterruptService()
        token.request("早就取消了")
        self.assertEqual(await asyncio.wait_for(token.wait(), timeout=1), "早就取消了")

    def test_listeners_are_notified_once(self) -> None:
        token = InterruptService()
        seen: list[str] = []
        token.on_request(lambda record: seen.append(record.reason))

        token.request("一次")
        token.request("两次")

        self.assertEqual(seen, ["一次"])


class InterruptLoopRebindingTests(unittest.TestCase):
    def test_sequential_asyncio_runs_with_and_without_explicit_binding(self):
        for explicit_binding in (False, True):
            with self.subTest(explicit_binding=explicit_binding):
                token = InterruptService()
                async def turn(reason):
                    token.reset()
                    if explicit_binding:
                        token.bind_loop(asyncio.get_running_loop())
                    waiter = asyncio.create_task(token.wait())
                    await asyncio.sleep(0)
                    self.assertFalse(waiter.done())
                    token.request(reason)
                    return await asyncio.wait_for(waiter, 2)
                self.assertEqual(asyncio.run(turn('first')), 'first')
                self.assertEqual(asyncio.run(turn('second')), 'second')
                token.reset()
                token.request('requested after loop closed')
                self.assertEqual(asyncio.run(token.wait()), 'requested after loop closed')


class CancellationInTheLoopTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.cwd = support.make_temp_dir()

    async def asyncTearDown(self) -> None:
        await support.remove_tree(self.cwd)

    async def test_cancel_between_steps_stops_the_turn(self) -> None:
        """第 1 步完整跑完(工具真的执行了),之后取消 —— 第 2 步不再开始。"""
        ctx = build_test_context(
            scripted_plugin(
                [
                    {"tool_calls": [{"name": "pwsh", "arguments": {"command": "echo 1"}}]},
                    {"tool_calls": [{"name": "pwsh", "arguments": {"command": "echo 2"}}]},
                    {"text": "不该走到这里"},
                ]
            ),
            self.cwd,
        )

        session = ctx.sessions.create()

        def cancel_after_first_step(event) -> None:
            if event.type == "step/end" and event.data.get("index") == 1:
                ctx.interrupt.request("第一步之后取消")

        session.observe(cancel_after_first_step)

        result = await ctx.agents.create(session).run("跑两步")

        self.assertEqual(result.stopped, "cancelled")
        self.assertEqual(result.steps, 1)
        calls = session.events_of("tool/call")
        self.assertEqual(len(calls), 1)  # 第二步的工具从未发出
        self.assertFalse(session.events_of("tool/result")[0].data["is_error"])
        self.assertNotIn(
            "不该走到这里",
            [e.data.get("text") for e in session.events_of("assistant/message")],
        )

    async def test_cancel_mid_tools_fills_every_tool_call_with_a_result(self) -> None:
        """取消发生在两个工具之间:第二个必须补一条结果。

        这是本轮最关键的断言 —— 它保证取消之后的会话日志仍可被下一个请求使用。

        这里关掉边流边执行,是为了拿到"顺序执行"这个可预期的场景;
        提前派发跑完的情况由下一条测试专门覆盖。
        """
        ctx = build_test_context(
            scripted_plugin(
                [
                    {
                        "tool_calls": [
                            {"id": "c1", "name": "pwsh", "arguments": {"command": "echo first"}},
                            {"id": "c2", "name": "pwsh", "arguments": {"command": "echo second"}},
                        ]
                    },
                    {"text": "收尾"},
                ]
            ),
            self.cwd,
            early_tools=False,
        )

        def cancel_after_first_tool(call, context, result) -> None:  # noqa: ANN001
            if call.id == "c1":  # 第一个工具执行完(还未写回日志)就取消
                ctx.interrupt.request("第一个工具之后取消")

        ctx.on("tools/execute", cancel_after_first_tool, mode=MODE_EMIT)

        result = await ctx.agents.create(ctx.sessions.create()).run("跑两个工具")

        self.assertEqual(result.stopped, "cancelled")
        calls = result.session.events_of("tool/call")
        results = result.session.events_of("tool/result")
        self.assertEqual([e.data["call_id"] for e in calls], ["c1", "c2"])
        self.assertEqual([e.data["call_id"] for e in results], ["c1", "c2"])
        self.assertFalse(results[0].data["is_error"])
        self.assertIn("已取消", results[1].data["content"])
        self.assertTrue(results[1].data["is_error"])

    async def test_already_dispatched_call_reports_its_real_result(self) -> None:
        """提前派发跑完了才被取消:必须**如实报告真实结果**,不能写"未执行"。

        因为副作用可能真的发生了(命令跑了、文件写了)。谎报"未执行"会让模型和人都判断错。
        """
        ctx = build_test_context(
            scripted_plugin(
                [
                    {
                        "tool_calls": [
                            {"name": "read", "arguments": {"file_path": "dispatched.txt"}}
                        ],
                        "text": "我一边解释一边等它跑完……" * 20,
                    },
                    {"text": "收尾"},
                ],
                stream_delay=0.02,
            ),
            self.cwd,
        )

        (self.cwd / "dispatched.txt").write_text("dispatched", encoding="utf-8")

        def cancel_at_stream_end(session, frame) -> None:  # noqa: ANN001
            # 流刚好结束、工具早已跑完,但结果还没写进日志 —— 正落在这个窗口里
            if frame.phase == "end":
                ctx.interrupt.request("流结束时取消")

        ctx.on("agent/assistant-stream", cancel_at_stream_end, mode=MODE_EMIT)

        result = await ctx.agents.create(ctx.sessions.create()).run("跑个快命令")

        self.assertEqual(result.stopped, "cancelled")
        results = result.session.events_of("tool/result")
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].data["is_error"])
        self.assertIn("dispatched", results[0].data["content"])
        self.assertNotIn("未执行", results[0].data["content"])

    async def test_cancel_during_streaming_leaves_no_half_message(self) -> None:
        """流式途中取消:不落半条 assistant 消息,step 记为取消。"""
        ctx = build_test_context(
            scripted_plugin([{"text": "很长的一段回答" * 20}], stream_size=2), self.cwd
        )

        def cancel_on_delta(request, event) -> None:  # noqa: ANN001
            ctx.interrupt.request("流式途中取消")

        ctx.on("llm/delta", cancel_on_delta, mode=MODE_EMIT)

        result = await ctx.agents.create(ctx.sessions.create()).run("说很多话")

        self.assertEqual(result.stopped, "cancelled")
        self.assertIsNone(result.text)
        self.assertEqual(result.session.events_of("assistant/message"), [])
        self.assertTrue(result.session.events_of("step/end")[0].data.get("cancelled"))

    async def test_log_stays_usable_after_cancellation(self) -> None:
        """取消之后,下一轮请求的历史里不能出现"没有结果的工具调用"。"""
        ctx = build_test_context(
            scripted_plugin(
                [
                    {
                        "tool_calls": [
                            {"id": "c1", "name": "pwsh", "arguments": {"command": "echo a"}},
                            {"id": "c2", "name": "pwsh", "arguments": {"command": "echo b"}},
                        ]
                    },
                    {"text": "第二轮的回答"},
                ]
            ),
            self.cwd,
        )

        def cancel_after_first_tool(call, context, result) -> None:  # noqa: ANN001
            if call.id == "c1":
                ctx.interrupt.request("取消")

        ctx.on("tools/execute", cancel_after_first_tool, mode=MODE_EMIT)

        session = ctx.sessions.create()
        agent = ctx.agents.create(session)
        await agent.run("第一轮")

        # 模拟 REPL:复位之后继续下一轮
        ctx.interrupt.reset()
        second = await agent.run("第二轮")

        messages = session.derive_messages()
        announced = [
            call.id
            for message in messages
            if message.role == "assistant"
            for call in message.tool_calls
        ]
        answered = [
            message.tool_call_id for message in messages if message.role == "tool"
        ]
        self.assertEqual(sorted(announced), sorted(answered))
        self.assertEqual(second.stopped, "final")
        self.assertEqual(second.text, "第二轮的回答")

    async def test_cancel_before_the_turn_skips_everything(self) -> None:
        ctx = build_test_context(scripted_plugin([{"text": "不该发生"}]), self.cwd)
        ctx.interrupt.request("先取消")

        result = await ctx.agents.create(ctx.sessions.create()).run("随便问")

        self.assertEqual(result.stopped, "cancelled")
        self.assertEqual(result.steps, 0)


class ShellCancellationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.cwd = support.make_temp_dir()

    async def asyncTearDown(self) -> None:
        await support.remove_tree(self.cwd)

    async def test_long_command_is_killed_promptly(self) -> None:
        """`sleep 30` 在 0.3s 后被取消 —— 必须立刻返回,而不是等 30s 或等超时。"""
        token = InterruptService()
        tool = make_shell_tool(default_cwd=self.cwd, timeout=60)
        context = ToolCallContext(call_id="t", cwd=self.cwd, cancellation=token)

        loop = asyncio.get_running_loop()
        loop.call_later(0.3, lambda: token.request("Ctrl+C"))

        started = time.perf_counter()
        result = await tool.handler({"command": "sleep 30"}, context)
        elapsed = time.perf_counter() - started

        self.assertTrue(result.is_error)
        self.assertIn("已被中断并终止", result.content)
        self.assertLess(elapsed, 15, "取消应当立刻生效,而不是等命令自己结束")

    async def test_already_cancelled_token_skips_execution(self) -> None:
        token = InterruptService()
        token.request("先取消了")
        tool = make_shell_tool(default_cwd=self.cwd, timeout=60)
        context = ToolCallContext(call_id="t", cwd=self.cwd, cancellation=token)

        started = time.perf_counter()
        result = await tool.handler({"command": "sleep 30"}, context)

        self.assertTrue(result.is_error)
        self.assertIn("已被中断并终止", result.content)
        self.assertLess(time.perf_counter() - started, 15)


class ToolCallContextTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.cwd = support.make_temp_dir()

    async def asyncTearDown(self) -> None:
        await support.remove_tree(self.cwd)

    async def test_execute_passes_the_token_into_the_context(self) -> None:
        ctx = build_test_context(scripted_plugin([{"text": "ok"}]), self.cwd)
        seen: list[object] = []

        def capture(call: ToolCall, context, nxt) -> object:  # noqa: ANN001
            seen.append(context.cancellation)
            return nxt()

        ctx.on("tools/pre-execute", capture, mode=MODE_WATERFALL)
        await ctx.tools.execute(
            ToolCall(id="c", name="pwsh", arguments={"command": "echo x"}), cwd=self.cwd
        )

        self.assertIs(seen[0], ctx.interrupt)


if __name__ == "__main__":
    unittest.main()
