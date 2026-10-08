"""会话续跑测试。

核心就两条:

1. **续跑之后历史完整** —— 新一轮请求里必须带着续跑前的对话。这一点不需要任何
   "记忆"机制,因为模型历史本来就是从日志投影出来的;
2. **未闭合的尾巴会被补齐** —— 强杀(第二次 Ctrl+C、断电)留下的半截日志,
   不该让整个会话作废。
"""

from __future__ import annotations

import argparse
import json
import time
import unittest
from pathlib import Path

from mini_harness import cli
from mini_harness.adapters.mock import ScriptedAdapter, scripted_plugin
from mini_harness.session import Session, SessionsService, repair_interrupted_tail

from . import support
from .support import adapter_plugin_for as plugin_for
from .support import build_context_with as build_test_context


def write_jsonl(path: Path, events: list[tuple[str, dict]]) -> Path:
    """手工造一份会话日志(用来模拟"强杀留下的半截日志")。"""
    lines = [
        json.dumps({"seq": index, "ts": 1.0 + index, "type": type_, "data": data})
        for index, (type_, data) in enumerate(events, start=1)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class RepairTests(unittest.TestCase):
    def test_well_formed_log_needs_no_repair(self) -> None:
        session = Session("s")
        session.append("turn/start")
        session.append("user/message", text="hi")
        session.append("step/start", index=1)
        session.append("step/end", index=1)
        session.append("turn/end", stopped="final")

        self.assertEqual(repair_interrupted_tail(session), [])

    def test_open_turn_is_closed(self) -> None:
        session = Session("s")
        session.append("turn/start")
        session.append("user/message", text="hi")

        notes = repair_interrupted_tail(session)

        self.assertEqual(len(notes), 1)
        last = session.events[-1]
        self.assertEqual(last.type, "turn/end")
        self.assertEqual(last.data["stopped"], "interrupted")
        self.assertTrue(last.data["repaired"])

    def test_open_step_and_turn_are_closed(self) -> None:
        session = Session("s")
        session.append("turn/start")
        session.append("step/start", index=1)

        notes = repair_interrupted_tail(session)

        self.assertEqual([e.type for e in session.events][-2:], ["step/end", "turn/end"])
        self.assertEqual(len(notes), 2)
        self.assertTrue(session.events[-2].data["repaired"])


class OpenTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.cwd = support.make_temp_dir()

    async def asyncTearDown(self) -> None:
        await support.remove_tree(self.cwd)

    async def test_open_repairs_an_interrupted_tail(self) -> None:
        path = write_jsonl(
            self.cwd / "half.jsonl",
            [
                ("turn/start", {}),
                ("user/message", {"text": "问题"}),
                ("step/start", {"index": 1}),
                ("assistant/message", {"text": "答了一半", "tool_calls": []}),
            ],
        )
        sessions = SessionsService(self.cwd)

        session = sessions.open(path)

        self.assertEqual(len(session.repairs), 2)  # 一个 step + 一个 turn
        self.assertEqual(session.events[-1].type, "turn/end")
        self.assertEqual(session.events[-1].data["stopped"], "interrupted")
        self.assertEqual(session.source_path, path)

    async def test_open_fills_missing_tool_results(self) -> None:
        """`tool/call` 有、结果没有 —— 这是**协议非法**的历史,必须补上。"""
        path = write_jsonl(
            self.cwd / "dangling.jsonl",
            [
                ("turn/start", {}),
                ("user/message", {"text": "跑个命令"}),
                ("step/start", {"index": 1}),
                (
                    "assistant/message",
                    {
                        "text": None,
                        "tool_calls": [
                            {"id": "c1", "name": "shell", "arguments": {"command": "echo x"}},
                            {"id": "c2", "name": "shell", "arguments": {"command": "echo y"}},
                        ],
                    },
                ),
                ("tool/call", {"call_id": "c1", "name": "shell", "arguments": {"command": "echo x"}}),
                ("tool/result", {"call_id": "c1", "name": "shell", "content": "exit=0", "is_error": False}),
            ],
        )

        session = SessionsService(self.cwd).open(path)

        results = {event.data["call_id"]: event.data for event in session.events_of("tool/result")}
        self.assertEqual(sorted(results), ["c1", "c2"])
        self.assertTrue(results["c2"]["is_error"])
        self.assertTrue(results["c2"]["repaired"])
        self.assertIn("中断", results["c2"]["content"])
        # 已经有的 c1 结果不能被覆盖
        self.assertFalse(results["c1"].get("repaired", False))

    async def test_open_tolerates_a_corrupt_line(self) -> None:
        path = self.cwd / "broken.jsonl"
        write_jsonl(path, [("turn/start", {}), ("user/message", {"text": "hi"})])
        text = path.read_text(encoding="utf-8")
        path.write_text(text + "这不是 JSON\n" + '{"seq": 4, "type": "turn/end"}\n', encoding="utf-8")

        session = SessionsService(self.cwd).open(path)

        self.assertTrue(any("跳过" in note for note in session.repairs))
        self.assertEqual(session.events_of("user/message")[0].data["text"], "hi")

    async def test_save_defaults_to_the_source_file(self) -> None:
        path = write_jsonl(self.cwd / "s.jsonl", [("turn/start", {}), ("turn/end", {})])
        sessions = SessionsService(self.cwd / "other")
        session = sessions.open(path)

        session.append("user/message", text="续跑的追加")
        saved = sessions.save(session)

        self.assertEqual(saved, path)
        self.assertIn("续跑的追加", path.read_text(encoding="utf-8"))


class SessionListingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.cwd = support.make_temp_dir()

    async def asyncTearDown(self) -> None:
        await support.remove_tree(self.cwd)

    async def test_lists_most_recent_first(self) -> None:
        sessions = SessionsService(self.cwd)
        older = write_jsonl(self.cwd / "older.jsonl", [("turn/start", {}), ("turn/end", {})])
        newer = write_jsonl(self.cwd / "newer.jsonl", [("turn/start", {}), ("turn/end", {})])
        stamp = time.time()
        import os

        os.utime(older, (stamp - 100, stamp - 100))
        os.utime(newer, (stamp, stamp))

        self.assertEqual([p.name for p in sessions.list_sessions()], ["newer.jsonl", "older.jsonl"])
        self.assertEqual(sessions.latest(), newer)

    async def test_latest_is_none_when_there_is_nothing(self) -> None:
        self.assertIsNone(SessionsService(self.cwd / "nope").latest())


class ResumeEndToEndTests(unittest.IsolatedAsyncioTestCase):
    """模拟"第二次启动进程,接着上次聊"。"""

    async def asyncSetUp(self) -> None:
        self.cwd = support.make_temp_dir()

    async def asyncTearDown(self) -> None:
        await support.remove_tree(self.cwd)

    async def test_second_process_sees_the_first_turns_history(self) -> None:
        first = build_test_context(scripted_plugin([{"text": "第一轮回答"}]), self.cwd)
        session = first.sessions.create()
        await first.agents.create(session).run("问题一")
        path = first.sessions.save(session, self.cwd / "resume.jsonl")

        # 第二个"进程":全新的 ctx,从文件读回
        adapter = ScriptedAdapter([{"text": "第二轮回答"}])
        second = build_test_context(plugin_for(adapter), self.cwd)
        resumed = second.sessions.open(path)
        result = await second.agents.create(resumed).run("问题二")

        self.assertEqual(result.text, "第二轮回答")
        self.assertEqual(result.stopped, "final")

        roles = [message.role for message in adapter.requests[0].messages]
        self.assertEqual(roles, ["user", "assistant", "user"])
        self.assertEqual(adapter.requests[0].messages[0].content, "问题一")
        self.assertEqual(adapter.requests[0].messages[1].content, "第一轮回答")

        # seq 接着往下排,没有重号
        seqs = [event.seq for event in resumed.events]
        self.assertEqual(seqs, sorted(seqs))
        self.assertEqual(len(seqs), len(set(seqs)))

    async def test_resume_after_an_interrupted_tail_still_produces_valid_history(self) -> None:
        """强杀留下的半截日志:补齐尾巴之后,新一轮的请求仍然合法。"""
        path = write_jsonl(
            self.cwd / "killed.jsonl",
            [
                ("turn/start", {}),
                ("user/message", {"text": "上一轮的问题"}),
                ("step/start", {"index": 1}),
                (
                    "assistant/message",
                    {
                        "text": None,
                        "tool_calls": [
                            {"id": "c1", "name": "shell", "arguments": {"command": "echo x"}}
                        ],
                    },
                ),
                ("tool/call", {"call_id": "c1", "name": "shell", "arguments": {"command": "echo x"}}),
            ],
        )
        adapter = ScriptedAdapter([{"text": "好的"}])
        ctx = build_test_context(plugin_for(adapter), self.cwd)

        session = ctx.sessions.open(path)
        self.assertTrue(session.repairs)

        await ctx.agents.create(session).run("新一轮")

        announced = [
            call.id
            for message in adapter.requests[0].messages
            if message.role == "assistant"
            for call in message.tool_calls
        ]
        answered = [
            message.tool_call_id
            for message in adapter.requests[0].messages
            if message.role == "tool"
        ]
        # 半截日志里那条工具调用没有结果 —— 修复必须把它补成"未执行",
        # 否则 assistant(tool_calls) 后面缺 tool 消息,请求会被 API 直接拒。
        self.assertEqual(announced, ["c1"])
        self.assertEqual(answered, ["c1"])

        repaired = [e for e in session.events_of("tool/result") if e.data.get("repaired")]
        self.assertEqual(len(repaired), 1)
        self.assertIn("中断", repaired[0].data["content"])


class CliSessionSelectionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.cwd = support.make_temp_dir()

    async def asyncTearDown(self) -> None:
        await support.remove_tree(self.cwd)

    def _args(self, **overrides) -> argparse.Namespace:
        base = {"resume": None, "continue_session": False}
        base.update(overrides)
        return argparse.Namespace(**base)

    async def test_resume_missing_file_is_an_error(self) -> None:
        ctx = build_test_context(scripted_plugin([{"text": "x"}]), self.cwd)
        with self.assertRaises(FileNotFoundError):
            cli._make_session(ctx, self._args(resume=str(self.cwd / "nope.jsonl")))

    async def test_resume_reports_notes(self) -> None:
        path = write_jsonl(
            self.cwd / "s.jsonl", [("turn/start", {}), ("user/message", {"text": "hi"})]
        )
        ctx = build_test_context(scripted_plugin([{"text": "x"}]), self.cwd)

        session, notes = cli._make_session(ctx, self._args(resume=str(path)))

        self.assertEqual(len(session.events), 3)  # 2 条 + 补上的 turn/end
        self.assertTrue(any("续跑" in note for note in notes))
        self.assertTrue(any("修复" in note for note in notes))

    async def test_continue_picks_the_latest_session(self) -> None:
        root = self.cwd / "sessions"
        SessionsService(root).root.mkdir(parents=True, exist_ok=True)
        write_jsonl(root / "a.jsonl", [("turn/start", {}), ("turn/end", {})])
        ctx = build_test_context(scripted_plugin([{"text": "x"}]), self.cwd, session_root=root)

        session, notes = cli._make_session(ctx, self._args(continue_session=True))

        self.assertEqual(session.source_path, root / "a.jsonl")
        self.assertTrue(any("续跑" in note for note in notes))

    async def test_continue_without_any_session_starts_a_new_one(self) -> None:
        ctx = build_test_context(
            scripted_plugin([{"text": "x"}]), self.cwd, session_root=self.cwd / "empty"
        )

        session, notes = cli._make_session(ctx, self._args(continue_session=True))

        self.assertIsNone(session.source_path)
        self.assertEqual(len(session.events), 0)
        self.assertTrue(any("没有可续" in note for note in notes))


if __name__ == "__main__":
    unittest.main()
