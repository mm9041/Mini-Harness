"""状态行、用量表、``/model`` 与 ``/access`` 的测试。

重点在两处容易被忽略的地方:

* **状态行必须读活状态,不能读配置快照** —— 否则 ``/model``、``/access`` 切完之后
  显示会撒谎;
* **换模型必须同步系统提示里的模型名** —— 提示词里那句"模型: X"要是旧的,
  模型就会照着假话自我介绍(身份幻觉那件事的同类)。
"""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

from mini_harness.adapters.mock import ScriptedAdapter
from mini_harness.agent_loop import AgentLoopService
from mini_harness.console import (
    format_event,
    format_status,
    human_tokens,
    shorten_path,
    status_for,
)
from mini_harness.token_meter import (
    DEFAULT_WINDOW,
    TokenMeter,
    window_for,
)

from . import support
from .support import adapter_plugin_for as plugin_for
from .support import build_context_with as build_test_context


class WindowTests(unittest.TestCase):
    def test_known_models_hit_the_table(self) -> None:
        size, guessed = window_for("qwen3.7-plus")
        self.assertEqual(size, 256_000)
        self.assertTrue(guessed, "网关不返回窗口大小,所以永远是猜的")

    def test_unknown_model_falls_back(self) -> None:
        self.assertEqual(window_for("totally-unknown-model")[0], DEFAULT_WINDOW)
        self.assertEqual(window_for("")[0], DEFAULT_WINDOW)

    def test_more_specific_keywords_win(self) -> None:
        self.assertEqual(window_for("deepseek-v4-pro")[0], 1_000_000)
        self.assertEqual(window_for("kimi-k3")[0], 256_000)
        self.assertEqual(window_for("qwen-turbo")[0], 1_000_000)


class MeterTests(unittest.TestCase):
    def test_provider_windows_are_preserved_exactly(self):
        for size in (131072, 262144, 524288, 1048576, 2000000):
            with self.subTest(size=size):
                meter = TokenMeter("custom")
                meter.note_capacities({"custom": size})
                self.assertEqual(meter.window, size)
                self.assertEqual(meter.window_source, "provider")
                self.assertFalse(meter.window_is_guess)
        configured = TokenMeter("custom", window=123456)
        configured.note_capacities({"custom": 2000000})
        self.assertEqual(configured.window, 123456)
        self.assertEqual(configured.window_source, "configured")
        self.assertEqual(TokenMeter("unknown").window_source, "estimated")

    def test_invalid_usage_is_not_reported_as_measured(self):
        for invalid in (-1, True, "100"):
            meter = TokenMeter("custom")
            meter.note_response({"prompt_tokens": invalid, "completion_tokens": invalid})
            self.assertIsNone(meter.snapshot().measured)
            self.assertIsNone(meter.snapshot().output_tokens)

    def test_estimate_then_measured(self) -> None:
        meter = TokenMeter("qwen3.7-plus")

        meter.note_request(1200)
        self.assertEqual(meter.snapshot().used, 1200)
        self.assertEqual(meter.snapshot().percent, 1200 / 256_000 * 100)

        meter.note_response({"prompt_tokens": 900, "completion_tokens": 50})
        snapshot = meter.snapshot()
        self.assertEqual(snapshot.measured, 900)
        self.assertEqual(snapshot.used, 900, "有实测就该用实测值")
        self.assertEqual(snapshot.output_tokens, 50)
        self.assertEqual(snapshot.requests, 1)

    def test_response_without_usage_is_ignored(self) -> None:
        meter = TokenMeter("m")
        meter.note_request(10)
        meter.note_response(None)
        meter.note_response({})
        self.assertIsNone(meter.snapshot().measured)

    def test_switching_model_drops_the_measured_value(self) -> None:
        """实测值是上一个模型报的,换完就不能再拿它当这个模型的用量。"""
        meter = TokenMeter("old")
        meter.note_response({"prompt_tokens": 500})
        meter.set_model("new")

        snapshot = meter.snapshot()
        self.assertEqual(snapshot.model, "new")
        self.assertIsNone(snapshot.measured)

    def test_explicit_window_overrides_the_table(self) -> None:
        meter = TokenMeter("qwen3.7-plus", window=1000)
        self.assertEqual(meter.snapshot().window, 1000)
        self.assertFalse(meter.snapshot().window_is_guess)

    def test_factor_is_learned_from_the_measured_value(self) -> None:
        """校准系数 = 实测/估算 的滑动平均。

        第一次:0.5×1.0 + 0.5×(500/1000) = 0.75。
        """
        meter = TokenMeter("m")
        meter.note_request(1000)
        meter.note_response({"prompt_tokens": 500})

        self.assertAlmostEqual(meter.factor, 0.75, places=3)
        self.assertEqual(meter.snapshot().calibrated, 750)

    def test_in_flight_request_drops_the_stale_measurement(self) -> None:
        """上一个请求的实测值不该算在这个请求头上。"""
        meter = TokenMeter("m")
        meter.note_request(1000)
        meter.note_response({"prompt_tokens": 500})  # factor → 0.75
        meter.note_request(2000)  # 新请求发出去了,还没回

        snapshot = meter.snapshot()

        self.assertIsNone(snapshot.measured)
        self.assertEqual(snapshot.used, 1500)  # 2000 × 0.75

    def test_factor_is_clamped_against_absurd_samples(self) -> None:
        """某次异常响应不能把比例带飞。"""
        meter = TokenMeter("m")
        meter.note_request(10)
        meter.note_response({"prompt_tokens": 10_000})  # 离谱的样本

        self.assertLessEqual(meter.factor, 5.0)
        self.assertGreater(meter.factor, 1.0)

    def test_switching_model_resets_the_factor(self) -> None:
        """不同模型的分词器不一样,旧比例不能拿来估新模型。"""
        meter = TokenMeter("old")
        meter.note_request(1000)
        meter.note_response({"prompt_tokens": 500})

        meter.set_model("new")

        self.assertEqual(meter.factor, 1.0)


class RenderingTests(unittest.TestCase):
    def test_human_tokens(self) -> None:
        self.assertEqual(human_tokens(940), "940")
        self.assertEqual(human_tokens(1234), "1.2k")
        self.assertEqual(human_tokens(131_072), "131k")
        self.assertEqual(human_tokens(None), "?")

    def test_shorten_path_abbreviates_home(self) -> None:
        self.assertTrue(shorten_path(Path.home() / "proj").startswith("~/"))

    def test_shorten_path_keeps_the_tail_segments(self) -> None:
        """路径里有信息的是最后几段,所以省略的是中间,不是尾巴。"""
        short = shorten_path("E:/PycharmProjects/DSH/mini-harness", limit=24)

        self.assertEqual(short, "E:/…/DSH/mini-harness")
        self.assertLess(len(short), 24)

    def test_shorten_path_leaves_short_paths_alone(self) -> None:
        self.assertEqual(shorten_path("E:/proj", limit=30), "E:/proj")

    def test_shorten_posix_path_preserves_root(self) -> None:
        self.assertEqual(
            shorten_path("/opt/projects/DSH/mini-harness", limit=24),
            "/opt/…/DSH/mini-harness",
        )
        self.assertEqual(shorten_path("/opt/proj"), "/opt/proj")
        self.assertEqual(shorten_path("/"), "/")
        self.assertEqual(shorten_path("/long-directory-name", limit=8), "/long-directory-name")

    def test_repeated_segment_is_not_mistaken_for_full_path(self) -> None:
        self.assertEqual(shorten_path("/opt/very-long-directory/opt/file", limit=12), "/opt/…/opt/file")

    def test_status_line_contents(self) -> None:
        meter = TokenMeter("qwen3.7-plus")
        meter.note_request(6_000)

        line = format_status(
            "qwen3.7-plus", Path("E:/proj"), usage=meter.snapshot(), access="ask"
        )

        self.assertIn("模型 qwen3.7-plus", line)
        self.assertIn("目录", line)
        self.assertIn("上下文 6.0k/256k?", line)  # ? = 窗口是近似值
        self.assertIn("估算", line)
        self.assertIn("权限 操作前询问", line)

    def test_status_line_warns_near_the_window(self) -> None:
        meter = TokenMeter("m", window=1000)
        meter.note_request(900)
        line = format_status("m", Path("E:/p"), usage=meter.snapshot(), access="allow")
        self.assertIn("⚠", line)
        self.assertIn("权限 自动批准", line)

    def test_status_line_shows_the_calibration(self) -> None:
        """已经校过就写出来 —— 让人知道这个数是估的、而且按实测纠过。"""
        meter = TokenMeter("m", window=10**9)
        meter.note_request(1000)
        meter.note_response({"prompt_tokens": 500})
        meter.note_request(2000)

        line = format_status("m", Path("E:/p"), usage=meter.snapshot(), access="ask")

        self.assertIn("估算×0.75", line)

    def test_status_line_without_usage(self) -> None:
        line = format_status("m", Path("E:/p"), usage=None, access="deny")
        self.assertNotIn("上下文", line)
        self.assertIn("拒绝受控操作", line)


class StatusFromContextTests(unittest.IsolatedAsyncioTestCase):
    """状态行读的必须是活状态。"""

    async def asyncSetUp(self) -> None:
        self.cwd = support.make_temp_dir()
        self.config = SimpleNamespace(
            task_cwd=self.cwd, model="qwen-plus", approval="ask"
        )

    async def asyncTearDown(self) -> None:
        await support.remove_tree(self.cwd)

    async def test_reflects_a_live_access_switch(self) -> None:
        ctx = build_test_context(
            plugin_for(ScriptedAdapter([{"text": "ok"}])), self.cwd, approval="ask"
        )

        ctx.approval.set_mode("allow")

        line = status_for(ctx, self.config)
        self.assertIn("自动批准", line)
        self.assertNotIn("请求批准", line)

    async def test_reflects_a_live_model_switch(self) -> None:
        """走真实的切换路径 —— 适配器与用量表会一起被更新。"""
        ctx = build_test_context(plugin_for(ScriptedAdapter([{"text": "ok"}])), self.cwd)

        await ctx.llm.use_model("qwen3.7-max")

        line = status_for(ctx, self.config)
        self.assertIn("qwen3.7-max", line)
        self.assertEqual(ctx.tokenMeter.model, "qwen3.7-max")

    async def test_adapter_wins_over_the_config_snapshot(self) -> None:
        """离线适配器忽略配置里的模型名,状态行必须显示**真的在用**的那个。

        这里刻意让配置里的名字与适配器的不同 —— 不然两边一样,这条测试什么都证明不了。
        """
        config = SimpleNamespace(
            task_cwd=self.cwd, model="config-only-model", approval="ask"
        )
        ctx = build_test_context(plugin_for(ScriptedAdapter([{"text": "ok"}])), self.cwd)

        line = status_for(ctx, config)

        self.assertIn(ScriptedAdapter.model, line)
        self.assertNotIn("config-only-model", line)


class _ReplCase(unittest.TestCase):
    """REPL 是**同步**设计(每个 turn 用 ``loop.run_until_complete``),
    所以这些测试必须是同步 TestCase —— 放进 ``IsolatedAsyncioTestCase`` 会和
    测试自己的事件循环撞车("This event loop is already running")。"""

    def make_repl(self, **overrides):
        from mini_harness.console import TracePrinter
        from mini_harness.repl import ReplSession

        self.cwd = support.make_temp_dir()
        adapter = ScriptedAdapter([{"text": "ok"}])
        ctx = build_test_context(plugin_for(adapter), self.cwd, **overrides)
        output: list[str] = []
        repl = ReplSession(
            ctx,
            printer=TracePrinter(write=output.append),  # 轨迹也收进列表,别打到 stdout
            write=output.append,
            input_fn=lambda prompt: "",
        )
        self.addCleanup(repl.close)
        self.addCleanup(repl.loop.close)
        self.addCleanup(lambda: __import__("shutil").rmtree(self.cwd, ignore_errors=True))
        return repl, output


class ModelCommandTests(_ReplCase):
    def test_model_switch_updates_the_prompt_and_the_meter(self) -> None:
        repl, output = self.make_repl()

        repl.handle_command("/model mock-strong")

        self.assertIn("→ mock-strong", "".join(output))
        self.assertEqual(repl.ctx.tokenMeter.model, "mock-strong")
        # 关键:系统提示里那一行必须跟着变,否则提示词在说谎
        self.assertIn("模型: mock-strong", repl.ctx.systemPrompt.render())

    def test_model_without_arguments_lists_the_catalogue(self) -> None:
        repl, output = self.make_repl()

        repl.handle_command("/model")

        text = "".join(output)
        self.assertIn("当前模型", text)
        self.assertIn("mock-strong", text)  # 来自离线适配器的固定清单

    def test_model_name_must_not_look_like_a_command(self) -> None:
        """把命令误当模型名要拦住 —— 一行里写两个斜杠命令是很容易犯的错。"""
        repl, output = self.make_repl()

        repl.handle_command("/model /access")

        self.assertIn("看起来不是模型名", "".join(output))
        self.assertEqual(repl.ctx.llm.active.model, ScriptedAdapter.model)  # 没被改掉

    def test_current_model_asks_the_adapter_first(self) -> None:
        """适配器持有的名字才是真的在用的那个,不能先读配置。"""
        repl, _ = self.make_repl()
        repl.ctx.tokenMeter.set_model("stale-from-meter")

        self.assertEqual(repl._current_model(), ScriptedAdapter.model)

    def test_model_switch_reports_adapter_failure(self) -> None:
        repl, output = self.make_repl()

        class Deaf:
            name = "deaf"

            async def generate(self, request):  # noqa: ANN001, ARG002
                raise AssertionError("不该被调用")

        repl.ctx.llm.register_adapter("deaf", Deaf())
        repl.ctx.llm.use("deaf")

        repl.handle_command("/model whatever")

        self.assertIn("换模型失败", "".join(output))


class StatusLineInReplTests(_ReplCase):
    def test_status_line_is_written_before_each_prompt(self) -> None:
        repl, output = self.make_repl()
        lines = ["问一句", "/exit"]
        repl.input_fn = lambda prompt: lines.pop(0)

        repl.start()

        text = "".join(output)
        self.assertGreaterEqual(text.count("── 模型"), 2)  # 提示符之前各画一次
        self.assertIn("目录", text)


class AccessCommandTests(_ReplCase):
    def test_access_accepts_plain_words(self) -> None:
        repl, output = self.make_repl()

        repl.handle_command("/access 完全访问")

        self.assertIn("自动批准", "".join(output))
        self.assertEqual(repl.ctx.approval.mode_for(repl.session), "allow")

    def test_access_without_arguments_shows_both_levels(self) -> None:
        repl, output = self.make_repl()

        repl.handle_command("/access")

        text = "".join(output)
        self.assertIn("当前权限", text)
        self.assertIn("操作前询问", text)
        self.assertIn("自动批准", text)

    def test_unknown_access_level_is_rejected(self) -> None:
        repl, output = self.make_repl()

        repl.handle_command("/access 随便什么")

        self.assertIn("未知档位", "".join(output))
        self.assertEqual(repl.ctx.approval.mode_for(repl.session), "ask")


class AccessGateTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.cwd = support.make_temp_dir()

    async def asyncTearDown(self) -> None:
        await support.remove_tree(self.cwd)

    async def test_access_switch_actually_changes_the_gate(self) -> None:
        """切到完全访问之后,写文件不再被拦 —— 证明它真的换了档,不只是显示变了。"""
        from mini_harness.llm import ToolCall

        ctx = build_test_context(
            plugin_for(ScriptedAdapter([{"text": "ok"}])), self.cwd, approval="ask"
        )
        call = ToolCall(
            id="c1",
            name="write",
            arguments={"file_path": "note.txt", "content": "hi"},
        )

        denied = await ctx.tools.execute(call, cwd=self.cwd)
        self.assertTrue(denied.is_error)
        self.assertFalse((self.cwd / "note.txt").exists())

        ctx.approval.set_mode("allow")  # /access full 走的就是这个
        allowed = await ctx.tools.execute(call, cwd=self.cwd)

        self.assertFalse(allowed.is_error, allowed.content)
        self.assertEqual((self.cwd / "note.txt").read_text(encoding="utf-8"), "hi")


class TurnEndsOnErrorTests(unittest.IsolatedAsyncioTestCase):
    """模型调用炸了的时候,turn 也必须被关上。

    实测踩到:网关返回 403(额度用尽)时,异常直接穿出循环,日志里留下
    "只有 turn/start、没有 turn/end" 的 turn。这类尾巴本该靠重启后的修复兜底,
    但更该在源头就补上 —— 日志的基本不变量不该依赖下游补救。
    """

    async def asyncSetUp(self) -> None:
        self.cwd = support.make_temp_dir()

    async def asyncTearDown(self) -> None:
        await support.remove_tree(self.cwd)

    async def test_failed_model_call_still_closes_the_turn(self) -> None:
        from mini_harness.llm import LLMError

        class Broken:
            name = "broken"

            async def generate(self, request):  # noqa: ANN001, ARG002
                raise LLMError("模型接口返回 HTTP 403:额度用尽")

        ctx = build_test_context(plugin_for(Broken()), self.cwd)
        session = ctx.sessions.create()

        with self.assertRaises(LLMError):
            await ctx.agents.create(session).run("问一句")

        ends = session.events_of("turn/end")
        self.assertEqual(len(ends), 1, "turn 必须被关上")
        self.assertEqual(ends[0].data["stopped"], "error")
        self.assertIn("403", ends[0].data["error"])

        # 结构平衡:turn/start 与 turn/end 数量一致
        self.assertEqual(
            len(session.events_of("turn/start")), len(session.events_of("turn/end"))
        )

    async def test_the_turn_can_be_resumed_afterwards(self) -> None:
        """出错之后还能接着问 —— 这条是用户实际会碰到的手感。"""
        from mini_harness.llm import LLMError, GenerateResult

        class Flaky:
            name = "flaky"
            model = "flaky"

            def __init__(self):
                self.calls = 0

            async def generate(self, request):  # noqa: ANN001, ARG002
                self.calls += 1
                if self.calls == 1:
                    raise LLMError("第一次故意失败")
                return GenerateResult(text="第二次好了")

        ctx = build_test_context(plugin_for(Flaky()), self.cwd)
        session = ctx.sessions.create()
        agent = ctx.agents.create(session)

        with self.assertRaises(LLMError):
            await agent.run("第一次")
        result = await agent.run("第二次")

        self.assertEqual(result.stopped, "final")
        self.assertEqual(result.text, "第二次好了")


class LoopWiringTests(unittest.IsolatedAsyncioTestCase):
    """驱动器要真的把用量喂给表,并把估算写进 step 事件。"""

    async def asyncSetUp(self) -> None:
        self.cwd = support.make_temp_dir()

    async def asyncTearDown(self) -> None:
        await support.remove_tree(self.cwd)

    async def test_estimate_counts_the_tool_schemas(self) -> None:
        """估算必须把工具 schema 算进来 —— 它是真实内容,而且是 prompt 的大头。

        回归测试:漏掉 tools 时实测比估算低 3.5 倍(实测 728 vs 估算 204),
        连带把状态行的用量和压缩触发点一起带偏。
        """
        from mini_harness.llm import Message, ToolSchema

        messages = [Message(role="user", content="问一句")]
        tools = [
            ToolSchema(
                name="big_tool",
                description="描述" * 200,
                parameters={"type": "object", "properties": {}},
            )
        ]

        without = AgentLoopService._estimated_tokens("sys", messages)
        with_tools = AgentLoopService._estimated_tokens("sys", messages, tools)

        self.assertGreater(with_tools - without, 200)
        self.assertEqual(without, AgentLoopService._estimated_tokens("sys", messages, []))

    async def test_step_event_and_meter_get_the_estimate(self) -> None:
        ctx = build_test_context(
            plugin_for(
                ScriptedAdapter(
                    [{"text": "答", "usage": {"prompt_tokens": 321, "completion_tokens": 7}}]
                )
            ),
            self.cwd,
        )

        result = await ctx.agents.create(ctx.sessions.create()).run("问一句")

        step = result.session.events_of("step/start")[0]
        self.assertIsInstance(step.data.get("context_tokens"), int)
        self.assertGreater(step.data["context_tokens"], 0)
        self.assertEqual(
            step.data["context_tokens_calibrated"], step.data["context_tokens"]
        )  # 还没校过,两个数一样

        snapshot = ctx.tokenMeter.snapshot()
        self.assertEqual(snapshot.estimated, step.data["context_tokens"])
        self.assertEqual(snapshot.measured, 321)  # provider 报的才是权威值
        self.assertEqual(snapshot.output_tokens, 7)

    async def test_calibration_shows_up_in_the_step_trace(self) -> None:
        """校准之后,轨迹里要能看到"原始估算 → 校准后"两个数。"""
        ctx = build_test_context(
            plugin_for(
                ScriptedAdapter(
                    [
                        {"text": "第一次", "usage": {"prompt_tokens": 400}},
                        {"text": "第二次"},
                    ]
                )
            ),
            self.cwd,
        )
        session = ctx.sessions.create()
        agent = ctx.agents.create(session)

        await agent.run("第一句")
        await agent.run("第二句")

        second_step = session.events_of("step/start")[-1]
        # 第一次实测只有 400,而估算约有 800 → 系数被拉到 1.0 以下
        self.assertLess(ctx.tokenMeter.factor, 1.0)
        self.assertLess(
            second_step.data["context_tokens_calibrated"],
            second_step.data["context_tokens"],
        )
        self.assertIn("校准后", format_event(second_step))

    async def test_request_no_longer_pins_the_model(self) -> None:
        """模型名归适配器持有 —— 否则 /model 换完之后请求还会带着旧名字。"""
        adapter = ScriptedAdapter([{"text": "答"}])
        ctx = build_test_context(plugin_for(adapter), self.cwd)

        await ctx.agents.create(ctx.sessions.create()).run("问一句")
        await ctx.llm.use_model("switched")
        await ctx.agents.create(ctx.sessions.create()).run("再问一句")

        self.assertIsNone(adapter.requests[0].model)
        self.assertIsNone(adapter.requests[1].model)
        self.assertEqual(adapter.model, "switched")


if __name__ == "__main__":
    unittest.main()
