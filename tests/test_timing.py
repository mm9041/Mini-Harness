"""耗时统计测试。

要证明的是三件事:

1. **每个 step 和每个 turn 都记了耗时** —— 而且记在**日志里**(所以续跑、事后分析都算得出来);
2. **时间拆得开** —— 模型 / 工具 / 退避三段分开记。"这一步 12 秒"和
   "12 秒里有 10 秒在退避重试"是两件完全不同的事,混在一起就没法判断该优化哪儿;
3. **轨迹里正确渲染** —— 并且老日志(没有这个字段)不会被编一个数出来。
"""

from __future__ import annotations

import unittest

from mini_harness.adapters.mock import ScriptedAdapter, scripted_plugin
from mini_harness.console import format_event
from mini_harness.session import SessionEvent

from . import support
from .support import adapter_plugin_for as plugin_for
from .support import build_context_with as build_test_context
from functools import partial

# These execution tests explicitly allow the PowerShell tool.
build_test_context = partial(build_test_context, approval="allow")


def event(type_: str, **data) -> SessionEvent:
    return SessionEvent(seq=1, type=type_, data=data)


class RecordingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.cwd = support.make_temp_dir()

    async def asyncTearDown(self) -> None:
        await support.remove_tree(self.cwd)

    async def test_every_step_and_turn_records_its_duration(self) -> None:
        ctx = build_test_context(
            scripted_plugin(
                [
                    {"tool_calls": [{"name": "pwsh", "arguments": {"command": "echo 一"}}]},
                    {"text": "好了"},
                ]
            ),
            self.cwd,
        )

        result = await ctx.agents.create(ctx.sessions.create()).run("跑一下")

        steps = result.session.events_of("step/end")
        self.assertEqual(len(steps), 2)
        for step in steps:
            self.assertIsInstance(step.data["duration_ms"], int)
            self.assertGreaterEqual(step.data["duration_ms"], 0)

        end = result.session.events_of("turn/end")[0]
        self.assertIsInstance(end.data["duration_ms"], int)
        self.assertGreaterEqual(
            end.data["duration_ms"], sum(s.data["duration_ms"] for s in steps)
        )
        self.assertEqual(result.duration_ms, end.data["duration_ms"])

    async def test_tool_and_model_time_are_split(self) -> None:
        """工具慢还是模型慢,必须分得清。"""
        ctx = build_test_context(
            scripted_plugin(
                [
                    {
                        "tool_calls": [
                            {"name": "pwsh", "arguments": {"command": "Start-Sleep -Milliseconds 300"}}
                        ]
                    },
                    {"text": "好了"},
                ]
            ),
            self.cwd,
        )

        result = await ctx.agents.create(ctx.sessions.create()).run("跑个慢命令")

        first = result.session.events_of("step/end")[0]
        self.assertGreaterEqual(first.data["tools_ms"], 300, "工具那 0.3 秒要记上")
        # 用**相对**断言:机器负载高时两边都会变长,写死 250ms 会偶发(踩过)。
        # 要证明的是"这 0.3 秒被算到工具头上,而不是模型头上"。
        self.assertGreater(
            first.data["tools_ms"], first.data["model_ms"], "慢的是工具,不是模型"
        )
        self.assertLessEqual(first.data["model_ms"], first.data["duration_ms"])

    async def test_slow_model_is_attributed_to_the_model(self) -> None:
        ctx = build_test_context(
            scripted_plugin(
                [{"text": "答" * 80}], stream_size=8, stream_delay=0.02
            ),
            self.cwd,
        )

        result = await ctx.agents.create(ctx.sessions.create()).run("问一句")

        step = result.session.events_of("step/end")[0]
        self.assertGreaterEqual(step.data["model_ms"], 150, "流式的 10 帧应当记进模型耗时")
        self.assertNotIn("tools_ms", step.data)  # 这一步没调工具

    async def test_cancelled_step_still_records_its_duration(self) -> None:
        """取消也要留下耗时 —— 不然"取消前跑了多久"就查不到。"""
        ctx = build_test_context(
            scripted_plugin([{"text": "很长的一段回答" * 20}], stream_size=2), self.cwd
        )

        from mini_harness.kernel import MODE_EMIT

        ctx.on(
            "llm/delta",
            lambda request, event: ctx.interrupt.request("取消"),
            mode=MODE_EMIT,
        )

        result = await ctx.agents.create(ctx.sessions.create()).run("说很多话")

        self.assertEqual(result.stopped, "cancelled")
        step = result.session.events_of("step/end")[0]
        self.assertTrue(step.data["cancelled"])
        self.assertIsInstance(step.data["duration_ms"], int)


class RetryAttributionTests(unittest.IsolatedAsyncioTestCase):
    """退避要从模型耗时里拆出来(否则"模型慢"是误判)。"""

    async def asyncSetUp(self) -> None:
        import threading
        from http.server import ThreadingHTTPServer

        from .test_streaming import _SseHandler

        self.cwd = support.make_temp_dir()
        _SseHandler.chunks = []
        _SseHandler.scripts = []
        _SseHandler.payloads = []
        _SseHandler.delay = 0.0
        _SseHandler.fail_times = 1
        _SseHandler.fail_status = 500
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _SseHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    async def asyncTearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        await support.remove_tree(self.cwd)

    async def test_backoff_time_is_reported_separately(self) -> None:
        import json

        from mini_harness.adapters.openai_compat import OpenAICompatAdapter

        from .test_streaming import _SseHandler

        _SseHandler.chunks = [
            json.dumps({"choices": [{"delta": {"content": "好了"}}]})
        ]
        adapter = OpenAICompatAdapter(api_key="k", base_url=self.base_url)
        ctx = build_test_context(
            plugin_for(adapter), self.cwd, retry_base_delay=0.2, retry_attempts=3
        )
        # 关掉抖动:否则退避在 0.15–0.25s 之间晃,断言会偶发(踩过)
        ctx.retry.policy.jitter = 0.0

        result = await ctx.agents.create(ctx.sessions.create()).run("问一句")

        step = result.session.events_of("step/end")[0]
        self.assertGreaterEqual(step.data["retry_ms"], 190, "0.2 秒退避要单独记")
        self.assertIn("retry_ms", step.data)
        self.assertLessEqual(
            step.data["retry_ms"] + step.data["model_ms"], step.data["duration_ms"] + 2
        )


class RenderingTests(unittest.TestCase):
    def test_step_end_shows_the_breakdown(self) -> None:
        line = format_event(
            event("step/end", index=2, duration_ms=2700, model_ms=2100, tools_ms=600)
        )

        self.assertIsNotNone(line)
        self.assertIn("step 2", line)
        self.assertIn("用时 2.7s", line)
        self.assertIn("模型 2.1s", line)
        self.assertIn("工具 0.6s", line)

    def test_backoff_is_not_double_counted(self) -> None:
        """事件中的模型耗时已扣除退避,渲染时不能再扣一次。"""
        line = format_event(
            event("step/end", index=1, duration_ms=12000, model_ms=2000, retry_ms=10000)
        )

        self.assertIn("模型 2.0s", line)
        self.assertIn("退避 10.0s", line)
        self.assertNotIn("其中退避", line)

    def test_turn_end_shows_the_total(self) -> None:
        line = format_event(event("turn/end", stopped="final", duration_ms=12345))
        self.assertIn("turn 结束(final", line)
        self.assertIn("用时 12.3s", line)

    def test_legacy_events_without_timing_are_not_invented(self) -> None:
        """老日志没有这些字段:宁可不显示,也不编一个数出来。"""
        self.assertIsNone(format_event(event("step/end", index=1)))
        legacy_turn = format_event(event("turn/end", stopped="final"))
        self.assertIsNotNone(legacy_turn)
        self.assertNotIn("用时", legacy_turn)

    def test_cancelled_step_is_marked(self) -> None:
        line = format_event(
            event("step/end", index=1, duration_ms=800, cancelled=True)
        )
        self.assertIn("被取消", line)

    def test_tiny_segments_are_not_cluttered(self) -> None:
        """几十毫秒级别的分段不值得占一行。"""
        line = format_event(
            event("step/end", index=1, duration_ms=180, model_ms=150, tools_ms=20)
        )
        self.assertIn("用时 0.2s", line)
        self.assertNotIn("工具", line)


if __name__ == "__main__":
    unittest.main()
