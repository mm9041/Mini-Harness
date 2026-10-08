import asyncio
import unittest
from pathlib import Path

from mini_harness.adapters.mock import ScriptedAdapter
from mini_harness.agent_loop import AssistantStreamFrame
from mini_harness.history import ConversationHistory
from mini_harness.llm import GenerateResult
from mini_harness.webui import ui_message
from .support import adapter_plugin_for, build_context_with, make_temp_dir, remove_tree


class HistoryAndReasoningTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.cwd = make_temp_dir()
        self.config = dict(session_root=self.cwd / "sessions", compaction=False)
        self.ctx = build_context_with(
            adapter_plugin_for(ScriptedAdapter([GenerateResult(text="已完成", reasoning="检查输入与边界。")], stream_delay=.001)),
            self.cwd, **self.config,
        )
        self.ui = self.ctx.webui
        self.ui.attach()
        self.ui.bind_loop(asyncio.get_running_loop())
        self.ui.autosave = True

    async def asyncTearDown(self):
        await remove_tree(self.cwd)

    async def test_history_survives_new_ui_and_continues_original_file(self):
        await self.ui._run_turn("编写解析器")
        original = self.ui.session.source_path
        self.ui.new_session()
        await self.ui._run_turn("另一项工作")
        other = self.ui.session.source_path
        other_content = other.read_bytes()
        # Recreate the complete context/catalogue to simulate process restart.
        ctx = build_context_with(adapter_plugin_for(ScriptedAdapter([{"text": "已完成后续修改"}])), self.cwd, **self.config)
        ui = ctx.webui
        ui.attach()
        ui.bind_loop(asyncio.get_running_loop())
        ui.autosave = True
        history = ui.history.list()
        self.assertEqual(len(history), 2)
        wanted = next(item for item in history if item["title"] == "编写解析器")
        ui.open_history(wanted["id"])
        await ui._run_turn("增加边界测试")
        self.assertEqual(ui.session.source_path, original)
        self.assertEqual(len(ui.session.events_of("user/message")), 2)
        self.assertEqual(other.read_bytes(), other_content)

    async def test_explicit_external_save_is_discoverable_after_restart(self):
        self.ui.save_path = self.cwd / "export" / "custom.jsonl"
        await self.ui._run_turn("外部位置")
        history = ConversationHistory(self.cwd / "sessions")
        self.assertEqual(len(history.list()), 1)
        self.assertEqual(history.resolve(history.list()[0]["id"]), self.ui.save_path.resolve())

    async def test_browsing_history_preserves_timestamp_until_next_message(self):
        await self.ui._run_turn("第一项工作")
        first = self.ui.history.list()[0]
        path = self.ui.session.source_path
        before = path.stat().st_mtime_ns
        self.ui.new_session()
        await self.ui._run_turn("第二项工作")
        second = self.ui.history.list()[0]
        self.ui.open_history(first["id"])
        self.ui.open_history(second["id"])
        self.ui.open_history(first["id"])
        self.assertEqual(path.stat().st_mtime_ns, before)
        listed = next(row for row in self.ui.history.list() if row["id"] == first["id"])
        self.assertEqual(listed["updated_at"], first["updated_at"])
        await self.ui._run_turn("继续第一项工作")
        latest = self.ui.history.list()[0]
        self.assertEqual(latest["id"], first["id"])
        self.assertGreater(latest["updated_at"], first["updated_at"])

    async def test_arbitrary_path_cannot_be_used_as_history_id(self):
        await self.ui._run_turn("saved")
        for value in (str(self.ui.session.source_path), "../session.jsonl", "missing"):
            with self.assertRaises(ValueError):
                self.ui.open_history(value)

    async def test_history_selection_preserves_active_turn(self):
        await self.ui._run_turn("saved")
        self.ui._busy = True
        worker = self.ui.target()
        snapshot = self.ui.open_history(self.ui.history.list()[0]["id"])
        self.assertTrue(snapshot["busy"])
        self.assertIs(self.ui.target(), worker)
        self.ui._busy = False

    async def test_committed_user_message_is_saved_before_model_finishes(self):
        self.ui.session.append("user/message", text="已接收的任务")
        path = self.ui.session.source_path
        self.assertIsNotNone(path)
        self.assertIn("已接收的任务", path.read_text(encoding="utf-8"))

    async def test_reasoning_streamed_saved_but_not_in_model_context(self):
        queue = self.ui.subscribe()
        await self.ui._run_turn("show reasoning")
        events = []
        while not queue.empty():
            events.append(queue.get_nowait())
        self.assertTrue(any(e.get("reasoning") for e in events if e["kind"] == "delta"))
        assistant = self.ui.session.events_of("assistant/message")[0]
        self.assertEqual(ui_message(assistant)["reasoning"], "检查输入与边界。")
        restored = self.ctx.sessions.open(self.ui.session.source_path)
        self.assertEqual(restored.events_of("assistant/message")[0].data["reasoning"], "检查输入与边界。")
        self.assertNotIn("检查输入与边界", str([m.to_wire() for m in restored.derive_messages()]))

    async def test_reconnect_contains_uncommitted_stream_and_finishes_cleanly(self):
        session = self.ui.session
        await self.ui._on_stream(session, AssistantStreamFrame("start"))
        await self.ui._on_stream(session, AssistantStreamFrame("chunk", reasoning="思考片段"))
        stream = self.ui.snapshot()["stream"]
        self.assertEqual(stream["reasoning"], "思考片段")
        self.assertFalse(stream["ended"])
        session.append("turn/end", stopped="cancelled")
        self.assertIsNone(self.ui.snapshot()["stream"])

    async def test_missing_saved_workspace_does_not_switch_session(self):
        await self.ui._run_turn("saved")
        key = self.ui.history.list()[0]["id"]
        self.ui.session.append("session/workspace", cwd=str(self.cwd / "removed"))
        self.ui.new_session()
        current = self.ui.session
        with self.assertRaises(ValueError):
            self.ui.open_history(key)
        self.assertIs(self.ui.session, current)
