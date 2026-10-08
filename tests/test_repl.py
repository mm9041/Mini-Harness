"""REPL 测试 —— 重点是**多轮复用同一个 Session**。

这条测试是"日志是唯一真相"的验收:第二轮请求里必须带着第一轮的用户消息与助手回复,
而两边唯一的"记忆"就是那份 append-only 日志 —— 没有任何对话缓存。
"""

from __future__ import annotations

import asyncio
import json
import unittest

from mini_harness.adapters.mock import ScriptedAdapter, scripted_plugin
from mini_harness.console import TracePrinter
from mini_harness.repl import ReplSession

from . import support
from .support import build_context_with as build_test_context


class _ReplHarness(unittest.TestCase):
    """建一个可脚本化输入输出的 REPL。"""

    def make(self, script, lines, cwd=None, **config_overrides):
        from mini_harness.app import HarnessConfig
        from mini_harness.kernel import Plugin

        self.cwd = cwd or support.make_temp_dir()
        # 直接用同一个 ScriptedAdapter 实例注册,方便断言它收到了哪些请求
        self.adapter = ScriptedAdapter(script)

        def apply(ctx) -> None:
            ctx.effect(ctx.llm.register_adapter(self.adapter.name, self.adapter))
            ctx.llm.use(self.adapter.name)

        config = HarnessConfig(task_cwd=self.cwd, offline=True, **config_overrides)
        ctx = build_test_context(
            Plugin(name="llm-scripted", apply=apply, inject=("llm",)),
            self.cwd,
            **config_overrides,
        )

        self.output: list[str] = []
        self.loop = asyncio.new_event_loop()
        pending = list(lines)

        def input_fn(prompt: str) -> str:  # noqa: ARG001
            if not pending:
                raise EOFError
            return pending.pop(0)

        # REPL 自身的话术与轨迹渲染都收进同一个列表,方便断言
        self.repl = ReplSession(
            ctx,
            config,
            printer=TracePrinter(
                write=self.output.append, show_trace=True, show_stream=True
            ),
            loop=self.loop,
            input_fn=input_fn,
            write=self.output.append,
        )
        return self.repl

    def tearDown(self) -> None:
        repl = getattr(self, "repl", None)
        if repl is not None:
            repl.close()
        loop = getattr(self, "loop", None)
        if loop is not None and not loop.is_closed():
            loop.close()
        cwd = getattr(self, "cwd", None)
        if cwd is not None:
            import shutil

            shutil.rmtree(cwd, ignore_errors=True)

    @property
    def text(self) -> str:
        return "".join(self.output)


class ReplCommandTests(_ReplHarness):
    def test_bad_resume_keeps_current_session_and_observer(self):
        repl = self.make([{'text':'still works'}], ['hello', '/exit'])
        original = repl.session
        path = self.cwd / 'duplicate.jsonl'
        path.write_text('\n'.join(json.dumps({'seq':1,'type':'user/message','data':{'text':'bad'}}) for _ in range(2)), encoding='utf-8')
        self.assertTrue(repl.handle_command('/resume duplicate.jsonl'))
        self.assertIs(repl.session, original)
        original.append('command/result', text='observer still attached')
        self.assertIn('observer still attached', self.text)
        self.assertEqual(repl.start(), 0)
        self.assertIn('序号重复或倒退', self.text)
        self.assertIn('still works', self.text)

    def test_changed_source_save_failure_allows_save_as_and_continue(self):
        repl = self.make([], ['/save', '/save recovered.jsonl', '/events', '/exit'])
        path = self.cwd / 'damaged.jsonl'
        path.write_text('{"seq":1,"type":"user/message","data":{"text":"keep"}}\n{broken\n', encoding='utf-8')
        repl.switch_session(path)
        changed = path.read_bytes() + b'changed externally\n'
        path.write_bytes(changed)
        self.assertEqual(repl.start(), 0)
        self.assertIn('命令失败:OSError', self.text)
        self.assertIn('已改变', self.text)
        self.assertEqual(path.read_bytes(), changed)
        self.assertTrue((self.cwd/'recovered.jsonl').is_file())
        self.assertIn('再见', self.text)

    def test_help_tools_events_and_unknown_command(self) -> None:
        self.make([{"text": "ok"}], ["/help", "/tools", "/events", "/nope", "/exit"])

        code = self.repl.start()

        self.assertEqual(code, 0)
        self.assertIn("/save", self.text)          # /help 的内容
        self.assertIn("read", self.text)      # /tools 列出了文件工具
        self.assertIn("未知命令 /nope", self.text)

    def test_exit_command_ends_the_loop(self) -> None:
        self.make([{"text": "ok"}], ["/exit"])
        self.assertEqual(self.repl.start(), 0)

    def test_eof_also_ends_the_loop(self) -> None:
        self.make([{"text": "ok"}], [])  # 立刻 EOF(等价 Ctrl+D)
        self.assertEqual(self.repl.start(), 0)

    def test_keyboard_interrupt_at_the_prompt_does_not_exit(self) -> None:
        self.make([{"text": "ok"}], [])

        calls = {"count": 0}
        original = self.repl.input_fn

        def input_fn(prompt: str) -> str:
            calls["count"] += 1
            if calls["count"] == 1:
                raise KeyboardInterrupt
            return original(prompt)

        self.repl.input_fn = input_fn

        code = self.repl.start()

        self.assertEqual(code, 0)
        self.assertIn("用 Ctrl+D 或 /exit 退出", self.text)


class ReplMultiTurnTests(_ReplHarness):
    def test_second_turn_sees_the_first_turn(self) -> None:
        self.make([{"text": "第一次的回答"}, {"text": "第二次的回答"}], ["问题一", "问题二", "/exit"])

        self.repl.start()

        # ① 一个会话、两次 turn
        events = self.repl.session.events
        self.assertEqual(len(self.repl.session.events_of("turn/start")), 2)
        self.assertEqual(len(self.repl.session.events_of("user/message")), 2)
        self.assertEqual(len(self.repl.session.events_of("assistant/message")), 2)
        self.assertEqual(self.repl.turns, 2)

        # ② 第二次请求确实带着第一轮的历史 —— 而且历史是从日志投影出来的
        self.assertEqual(len(self.adapter.requests), 2)
        roles = [message.role for message in self.adapter.requests[1].messages]
        self.assertEqual(roles, ["user", "assistant", "user"])
        self.assertEqual(
            self.adapter.requests[1].messages[0].content, "问题一"
        )
        self.assertEqual(
            self.adapter.requests[1].messages[1].content, "第一次的回答"
        )

        # ③ 输出里能看到两轮
        self.assertIn("第 1 轮回答", self.text)
        self.assertIn("第 2 轮回答", self.text)

    def test_blank_input_is_ignored(self) -> None:
        self.make([{"text": "只有一轮"}], ["   ", "问一句", "/exit"])

        self.repl.start()

        self.assertEqual(len(self.adapter.requests), 1)
        self.assertEqual(self.repl.turns, 1)

    def test_tool_history_carries_over_to_the_next_turn(self) -> None:
        self.make(
            [
                {"tool_calls": [{"name": "pwsh", "arguments": {"command": "echo carried"}}]},
                {"text": "第一轮完成"},
                {"text": "第二轮完成"},
            ],
            ["跑个命令", "继续", "/exit"],
        )

        self.repl.start()

        roles = [message.role for message in self.adapter.requests[2].messages]
        self.assertEqual(roles, ["user", "assistant", "tool", "assistant", "user"])

    def test_save_command_writes_jsonl(self) -> None:
        repl = self.make([{"text": "回答"}], ["问题", "/save session.jsonl", "/exit"])
        target = self.cwd / "session.jsonl"

        self.repl.start()

        self.assertTrue(target.exists())
        lines = [
            json.loads(line)
            for line in target.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(lines), len(repl.session.events))
        self.assertIn("已保存", self.text)


if __name__ == "__main__":
    unittest.main()
