"""端到端闭环测试:不联网、不花 token,验证"插件树 → turn/step → 工具 → 日志"整条链路。

    python -m unittest discover -s tests -t .
"""

from __future__ import annotations

import unittest

from mini_harness import adapters
from mini_harness.adapters.mock import ScriptedAdapter, scripted_plugin
from mini_harness.adapters.openai_compat import OpenAICompatAdapter, parse_completion
from mini_harness.kernel import MODE_WATERFALL
from mini_harness.tools import ToolResult

from . import support
from .support import build_context_with as build_test_context
from functools import partial

# These execution tests explicitly allow the PowerShell tool.
build_test_context = partial(build_test_context, approval="allow")


class ClosedLoopTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        # 用 mkdtemp 而不是 TemporaryDirectory:清理要能容忍 Windows 上子进程退出后
        # 残留的目录句柄,否则断言全过的测试会因为 teardown 里的 WinError 32 变成 error。
        self.cwd = support.make_temp_dir()

    async def asyncTearDown(self) -> None:
        await support.remove_tree(self.cwd)

    async def test_turn_runs_tool_then_answers(self) -> None:
        adapter_plugin = scripted_plugin(
            [
                {"tool_calls": [{"name": "pwsh", "arguments": {"command": "echo hello-loop"}}]},
                {"text": "命令执行完了"},
            ]
        )
        ctx = build_test_context(adapter_plugin, self.cwd)
        session = ctx.sessions.create()
        agent = ctx.agents.create(session)

        result = await agent.run("跑一下 echo")

        # 1) 循环正常收尾
        self.assertEqual(result.stopped, "final")
        self.assertEqual(result.steps, 2)
        self.assertEqual(result.text, "命令执行完了")

        # 2) 日志里的事实齐全,顺序正确
        self.assertEqual(
            [event.type for event in session.events],
            [
                "turn/start",
                "user/message",
                "step/start",
                "assistant/message",
                "tool/call",
                "approval/asked",
                "approval/decided",
                "tool/result",
                "step/end",
                "step/start",
                "assistant/message",
                "step/end",
                "turn/end",
            ],
        )

        # 3) 工具真的执行了(用的是真实 shell 子进程)
        tool_result = session.events_of("tool/result")[0]
        self.assertFalse(tool_result.data["is_error"])
        self.assertIn("hello-loop", tool_result.data["content"])
        self.assertIn('"returncode": 0', tool_result.data["content"])

        # 4) 模型历史是从日志投影出来的:user -> assistant(工具调用) -> tool -> assistant(收尾)
        messages = session.derive_messages()
        self.assertEqual(
            [message.role for message in messages],
            ["user", "assistant", "tool", "assistant"],
        )
        self.assertEqual(messages[1].tool_calls[0].name, "pwsh")
        self.assertEqual(messages[3].content, "命令执行完了")

        # 5) 第二次请求里带着工具结果 —— 说明结果确实回到了模型视野
        adapter = ctx.llm.active
        self.assertEqual(len(adapter.requests), 2)
        second = adapter.requests[1]
        self.assertEqual([m.role for m in second.messages], ["user", "assistant", "tool"])
        self.assertIn("hello-loop", second.messages[-1].content or "")

        # 6) 工具 schema 进了协议字段,运行时上下文进了系统提示
        self.assertEqual(
            [schema.name for schema in second.tools],
            ["read", "write", "edit", "glob", "grep", "read_image", "pwsh", "job_list", "job_output", "job_kill", "web_search", "web_fetch", "ask_user_question", "present", "task"],
        )
        self.assertIn("运行时上下文", second.system)

    async def test_max_steps_stops_runaway_loop(self) -> None:
        looping = {"tool_calls": [{"name": "pwsh", "arguments": {"command": "echo x"}}]}
        ctx = build_test_context(scripted_plugin([looping] * 10), self.cwd, max_steps=3)

        agent = ctx.agents.create(ctx.sessions.create())
        result = await agent.run("无限工具调用")

        self.assertEqual(result.stopped, "max-steps")
        self.assertEqual(result.steps, 3)
        self.assertEqual(len(result.session.events_of("tool/result")), 3)
        end = result.session.events_of("turn/end")[-1]
        self.assertEqual((end.data["steps"], end.data["step_limit"]), (3, 3))

    async def test_default_budget_finishes_more_than_eight_tool_steps(self):
        from mini_harness.app import HarnessConfig
        (self.cwd / "probe.txt").write_text("tool result", encoding="utf-8")
        script = [{"tool_calls": [{"name": "read", "arguments": {"file_path": "probe.txt"}}]} for _ in range(12)]
        script.append({"text": "全部检查完成"})
        ctx = build_test_context(scripted_plugin(script), self.cwd, max_steps=HarnessConfig().max_steps)
        self.addCleanup(ctx.dispose)
        result = await ctx.agents.create(ctx.sessions.create()).run("逐项检查")
        self.assertEqual(result.stopped, "final")
        self.assertEqual(result.steps, 13)
        self.assertEqual(result.text, "全部检查完成")
        self.assertEqual(len(result.session.events_of("tool/result")), 12)

    async def test_continuing_after_limit_preserves_results_without_replaying_tools(self):
        (self.cwd / "probe.txt").write_text("retained result", encoding="utf-8")
        script = [{"tool_calls": [{"name": "read", "arguments": {"file_path": "probe.txt"}}]} for _ in range(4)]
        script.append({"text": "完成"})
        adapter = ScriptedAdapter(script)
        ctx = build_test_context(support.adapter_plugin_for(adapter), self.cwd, max_steps=3)
        self.addCleanup(ctx.dispose)
        session = ctx.sessions.create()
        agent = ctx.agents.create(session)
        first = await agent.run("检查")
        self.assertEqual(first.stopped, "max-steps")
        second = await agent.run("继续")
        self.assertEqual(second.stopped, "final")
        self.assertEqual(second.steps, 2)
        self.assertEqual(len(session.events_of("tool/result")), 4)
        self.assertEqual(sum(m.role == "tool" for m in adapter.requests[3].messages), 3)

    async def test_pre_execute_seam_can_deny_a_tool(self) -> None:
        """自己挂的策略能拦下工具。

        刻意选一个**内置危险模式不覆盖**的命令(`shred`),否则被 approval 插件先拦掉,
        这条测试就会"因为别的原因通过"。
        """
        ctx = build_test_context(
            scripted_plugin(
                [
                    {
                        "tool_calls": [
                            {"name": "pwsh", "arguments": {"command": "shred notes.txt"}}
                        ]
                    },
                    {"text": "被拒绝了"},
                ]
            ),
            self.cwd,
        )

        def deny_destructive(call, context, nxt):  # noqa: ANN001
            if "shred" in str(call.arguments.get("command", "")):
                return ToolResult("已被自定义策略拒绝:禁止 shred", is_error=True)
            return nxt()

        ctx.on("tools/pre-execute", deny_destructive, mode=MODE_WATERFALL)

        result = await ctx.agents.create(ctx.sessions.create()).run("销毁文件")

        tool_result = result.session.events_of("tool/result")[0]
        self.assertTrue(tool_result.data["is_error"])
        self.assertIn("自定义策略拒绝", tool_result.data["content"])

    async def test_post_execute_seam_can_rewrite_output(self) -> None:
        ctx = build_test_context(
            scripted_plugin(
                [
                    {"tool_calls": [{"name": "pwsh", "arguments": {"command": "echo secret"}}]},
                    {"text": "好了"},
                ]
            ),
            self.cwd,
        )

        async def redact(call, result, context, nxt):  # noqa: ANN001
            # post-execute 契约: (call, result, context) + next —— 带上了 context,
            # 因为结果外溢要按会话落盘(见 spill.py)。
            result.content = result.content.replace("secret", "***")
            return await nxt()

        ctx.on("tools/post-execute", redact, mode=MODE_WATERFALL)

        result = await ctx.agents.create(ctx.sessions.create()).run("打印")

        content = result.session.events_of("tool/result")[0].data["content"]
        self.assertNotIn("secret", content)
        self.assertIn("***", content)

    async def test_unknown_tool_becomes_structured_error(self) -> None:
        ctx = build_test_context(
            scripted_plugin(
                [
                    {"tool_calls": [{"name": "no_such_tool", "arguments": {}}]},
                    {"text": "知道了"},
                ]
            ),
            self.cwd,
        )

        result = await ctx.agents.create(ctx.sessions.create()).run("调一个不存在的工具")

        tool_result = result.session.events_of("tool/result")[0]
        self.assertTrue(tool_result.data["is_error"])
        self.assertIn("未知工具", tool_result.data["content"])

    async def test_pre_step_seam_can_reject_input(self) -> None:
        ctx = build_test_context(scripted_plugin([{"text": "不该走到这里"}]), self.cwd)
        ctx.on("agent/pre-step", lambda text, nxt: None, mode=MODE_WATERFALL)

        result = await ctx.agents.create(ctx.sessions.create()).run("   ")

        self.assertEqual(result.stopped, "empty-turn")
        self.assertEqual(result.steps, 0)
        self.assertNotIn("assistant/message", [e.type for e in result.session.events])

    async def test_reasoning_only_response_still_finishes(self) -> None:
        """思考型模型(如 Qwen3 系)可能只给 reasoning_content,别让 turn 空着收场。"""
        ctx = build_test_context(
            scripted_plugin([{"text": None, "reasoning": "想清楚了,没有需要执行的动作"}]), self.cwd
        )

        result = await ctx.agents.create(ctx.sessions.create()).run("随便问一句")

        self.assertEqual(result.stopped, "final")
        self.assertEqual(result.text, "想清楚了,没有需要执行的动作")

    async def test_offline_echo_adapter_closes_the_loop(self) -> None:
        """不写脚本,直接用 --mock 用的那个适配器跑一遍。"""
        ctx = build_test_context(adapters.echo_plugin("echo mini-harness-ok"), self.cwd)

        result = await ctx.agents.create(ctx.sessions.create()).run("自我检查")

        self.assertEqual(result.stopped, "final")
        self.assertIn("mini-harness-ok", result.text or "")

    async def test_session_round_trips_through_jsonl(self) -> None:
        ctx = build_test_context(
            scripted_plugin(
                [
                    {"tool_calls": [{"name": "pwsh", "arguments": {"command": "echo persist"}}]},
                    {"text": "完成"},
                ]
            ),
            self.cwd,
        )
        result = await ctx.agents.create(ctx.sessions.create()).run("落盘")

        path = result.session.save(self.cwd / "session.jsonl")
        restored = ctx.sessions.open(path)

        self.assertEqual(len(restored.events), len(result.session.events))
        self.assertEqual(restored.derive_messages(), result.session.derive_messages())

    async def test_provider_swap_keeps_agent_loop_untouched(self) -> None:
        """同一个 agent loop,换 provider 就换模型来源 —— 接缝的意义。"""
        ctx = build_test_context(scripted_plugin([{"text": "来自假模型"}]), self.cwd)
        agent = ctx.agents.create(ctx.sessions.create())

        before = await agent.run("你好")
        self.assertEqual(before.text, "来自假模型")

        # 热插拔第二个 provider 并切过去,循环代码一行未改。
        originals = ScriptedAdapter([{"text": "来自第二个假模型"}])
        ctx.effect(ctx.llm.register_adapter("second", originals))
        ctx.llm.use("second")

        after = await agent.run("再来一次")
        self.assertEqual(after.text, "来自第二个假模型")


class WireFormatTests(unittest.IsolatedAsyncioTestCase):
    """不打网络:请求体怎么拼、响应怎么解析,都在这里断言。

    没有 key 也能验证协议这一层,靠的是把 ``_post``(唯一的网络出口)换掉。
    """

    async def test_request_payload_matches_openai_protocol(self) -> None:
        from mini_harness.builtin_tools.shell import SHELL_PARAMETERS
        from mini_harness.llm import (
            GenerateRequest,
            Message,
            ToolCall,
            ToolSchema,
        )

        adapter = OpenAICompatAdapter(
            api_key="test-key", base_url="https://example.invalid/v1", model="deepseek-chat"
        )
        self.assertEqual(adapter.endpoint, "https://example.invalid/v1/chat/completions")

        captured: dict = {}

        def fake_post(payload: dict) -> dict:
            captured.update(payload)
            return {
                "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}]
            }

        adapter._post = fake_post  # type: ignore[method-assign]

        request = GenerateRequest(
            system="SYS",
            messages=[
                Message(role="user", content="hi"),
                Message(
                    role="assistant",
                    content=None,
                    tool_calls=[ToolCall(id="c1", name="pwsh", arguments={"command": "ls"})],
                ),
                Message(role="tool", content="out", tool_call_id="c1"),
            ],
            tools=[
                ToolSchema(name="pwsh", description="d", parameters=SHELL_PARAMETERS)
            ],
        )

        result = await adapter.generate(request)

        self.assertEqual(captured["model"], "deepseek-chat")
        self.assertFalse(captured["stream"])
        self.assertEqual(captured["messages"][0], {"role": "system", "content": "SYS"})
        self.assertEqual(captured["messages"][1], {"role": "user", "content": "hi"})
        assistant = captured["messages"][2]
        self.assertEqual(assistant["role"], "assistant")
        self.assertEqual(assistant["tool_calls"][0]["function"]["name"], "pwsh")
        self.assertEqual(
            assistant["tool_calls"][0]["function"]["arguments"], '{"command": "ls"}'
        )
        self.assertEqual(
            captured["messages"][3],
            {"role": "tool", "tool_call_id": "c1", "content": "out"},
        )
        self.assertEqual(captured["tools"][0]["type"], "function")
        self.assertEqual(captured["tools"][0]["function"]["name"], "pwsh")
        self.assertEqual(result.text, "ok")

    def test_parse_text_and_tool_calls(self) -> None:
        payload = {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_abc",
                                "type": "function",
                                "function": {
                                    "name": "pwsh",
                                    "arguments": '{"command": "ls -la"}',
                                },
                            }
                        ],
                    },
                }
            ],
            "usage": {"total_tokens": 42},
        }

        result = parse_completion(payload)

        self.assertIsNone(result.text)
        self.assertEqual(result.finish_reason, "tool_calls")
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].name, "pwsh")
        self.assertEqual(result.tool_calls[0].arguments, {"command": "ls -la"})
        self.assertEqual(result.usage, {"total_tokens": 42})

    def test_invalid_arguments_do_not_crash(self) -> None:
        payload = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {"id": "1", "function": {"name": "pwsh", "arguments": "{oops"}}
                        ]
                    }
                }
            ]
        }

        result = parse_completion(payload)

        self.assertEqual(result.tool_calls[0].arguments, {})
        self.assertEqual(result.tool_calls[0].raw_arguments, "{oops")

    def test_empty_choices_raises_clear_error(self) -> None:
        from mini_harness.llm import LLMError

        with self.assertRaises(LLMError):
            parse_completion({"error": "boom"})


if __name__ == "__main__":
    unittest.main()
