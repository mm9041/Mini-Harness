"""Regression coverage for interrupted streams and the local web lifecycle."""
import asyncio
import json
import os
import urllib.error
import urllib.request
import unittest
from pathlib import Path
from unittest.mock import patch

from mini_harness.llm import GenerateResult, LLMError, StreamEvent, ToolCall
from mini_harness.tools import Tool, ToolResult
from mini_harness.webui import serve
from .support import adapter_plugin_for, build_context_with, make_temp_dir, remove_tree


class FinalAdapter:
    name = "review"
    model = "review"

    async def generate(self, request):
        return GenerateResult(text="done")


class ReviewRegressions(unittest.IsolatedAsyncioTestCase):
    async def test_restoring_other_model_does_not_reuse_its_measurement(self):
        session = self.ctx.sessions.create()
        session.append("step/start", model="another-model", context_tokens=100)
        session.append("assistant/message", model="another-model", text="old answer",
                       usage={"prompt_tokens": 999})
        self.ui._bind_session(session)
        context = self.ui.status()["context"]
        self.assertIsNone(context["last_input_measured"])
        self.assertIsNone(self.ctx.tokenMeter.snapshot().measured)
        self.assertEqual(self.ctx.tokenMeter.factor, 1.0)

    async def test_context_projection_includes_reply_without_mutating_meter(self):
        self.ui.session.append("user/message", text="question")
        self.ctx.tokenMeter.note_request(100)
        self.ctx.tokenMeter.note_response({"prompt_tokens": 90})
        before = self.ui.status()["context"]["next_estimated"]
        self.ui.session.append("assistant/message", text="answer " * 1000,
                               model=self.ctx.llm.active.model, usage={"prompt_tokens": 90})
        meter_before = self.ctx.tokenMeter.snapshot()
        count = len(self.ui.session.events)
        context = self.ui.status()["context"]
        self.assertGreater(context["next_estimated"], before)
        self.assertEqual(context["last_input_measured"], 90)
        self.assertEqual(self.ctx.tokenMeter.snapshot(), meter_before)
        self.assertEqual(len(self.ui.session.events), count)
        self.ui.new_session()
        self.assertIsNone(self.ui.status()["context"]["last_input_measured"])

    async def asyncSetUp(self):
        self.cwd = make_temp_dir()
        self.ctx = build_context_with(
            adapter_plugin_for(FinalAdapter()), self.cwd,
            compaction=False, approval="allow", retry_base_delay=0,
            session_root=self.cwd / "sessions",
        )
        self.ui = self.ctx.webui
        self.ui.attach()
        self.ui.bind_loop(asyncio.get_running_loop())

    async def asyncTearDown(self):
        await asyncio.to_thread(self.ui.stop)
        await remove_tree(self.cwd)

    async def test_stop_does_not_poison_following_turns(self):
        self.ui._busy = True
        self.assertTrue(self.ui.interrupt())
        self.ui._busy = False
        await self.ui._run_turn("next")
        await self.ui._run_turn("again")
        self.assertEqual(
            [e.data["stopped"] for e in self.ui.session.events_of("turn/end")],
            ["final", "final"],
        )

    async def test_back_to_back_send_is_reserved_before_scheduling(self):
        self.ui.send("first")
        with self.assertRaises(RuntimeError):
            self.ui.send("second")
        previous = self.ui.target()
        self.ui.new_session()
        await asyncio.wrap_future(previous._turn_future)
        self.assertEqual(len(previous.session.events_of("user/message")), 1)
        self.assertEqual(self.ui.session.events_of("user/message"), [])

    async def test_failed_stream_never_repeats_mutating_tool(self):
        class BrokenStream(FinalAdapter):
            attempts = 0

            async def stream(self, request):
                self.attempts += 1
                if self.attempts > 2:
                    yield StreamEvent(kind="done", result=GenerateResult(text="done"))
                    return
                call = ToolCall(str(self.attempts), "append_probe", {})
                yield StreamEvent(kind="tool_call", tool_call=call)
                await asyncio.sleep(0.02)
                if self.attempts == 1:
                    raise LLMError("reset after tool arguments", retryable=True)
                yield StreamEvent(kind="done", result=GenerateResult(tool_calls=[call]))

        adapter = BrokenStream()
        self.ctx.llm.register_adapter(adapter.name + "-broken", adapter)
        self.ctx.llm.use(adapter.name + "-broken")
        target = self.cwd / "effects.txt"

        async def mutate(args, context):
            with target.open("a") as stream:
                stream.write("effect\n")
            return ToolResult("done")

        self.ctx.tools.register(Tool("append_probe", "mutating", handler=mutate, permission="write"))
        result = await self.ctx.agents.create(self.ctx.sessions.create()).run("do it")
        self.assertEqual(target.read_text(), "effect\n")
        self.assertEqual(len(result.session.events_of("tool/result")), 1)
        self.assertFalse(self.ctx.tools.can_prefetch("pwsh"))
        self.assertFalse(self.ctx.tools.can_prefetch("task"))
        self.assertTrue(self.ctx.tools.can_prefetch("read"))

    async def test_fatal_stream_cleans_up_speculative_read(self):
        started = asyncio.Event()
        finished = asyncio.Event()

        async def read(args, context):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                finished.set()

        class Fatal(FinalAdapter):
            async def stream(self, request):
                yield StreamEvent(kind="tool_call", tool_call=ToolCall("r", "slow_read", {}))
                await started.wait()
                raise LLMError("fatal", status=401)

        self.ctx.tools.register(Tool("slow_read", "read", handler=read, safe_to_prefetch=True, permission="read"))
        self.ctx.llm.register_adapter("fatal", Fatal())
        self.ctx.llm.use("fatal")
        with self.assertRaises(LLMError):
            await self.ctx.agents.create(self.ctx.sessions.create()).run("read")
        self.assertTrue(finished.is_set())

    async def test_web_resume_save_and_new_session_keep_old_file(self):
        old = self.ctx.sessions.create("resumed")
        old.append("user/message", text="old input")
        path = self.ctx.sessions.save(old)
        loaded = self.ctx.sessions.open(path)
        with patch.object(self.ui, "start"):
            serve(self.ctx, None, loop=asyncio.get_running_loop(), session=loaded)
        self.assertIs(self.ui.session, loaded)
        await self.ui._run_turn("new input")
        self.assertIn("new input", path.read_text(encoding="utf-8"))
        previous = path.read_bytes()
        self.ui.new_session()
        await self.ui._run_turn("separate input")
        self.assertEqual(path.read_bytes(), previous)
        self.assertNotEqual(self.ui.session.source_path, path)
        self.assertTrue(self.ui.session.source_path.is_file())

    async def test_explicit_save_path_is_used(self):
        target = self.cwd / "chosen.jsonl"
        self.ui.autosave = True
        self.ui.save_path = target
        await self.ui._run_turn("saved")
        self.assertIn("saved", target.read_text(encoding="utf-8"))

    async def test_stop_releases_browser_approval(self):
        from mini_harness.approval import ApprovalRequest
        request = ApprovalRequest(ToolCall("w", "write", {}), "write")
        self.ctx.interrupt.bind_loop(asyncio.get_running_loop())
        task = asyncio.create_task(self.ui._ask_the_browser(request))
        await asyncio.sleep(0)
        self.assertEqual(len(self.ui.snapshot()["approvals"]), 1)
        self.ctx.interrupt.request()
        result = await asyncio.wait_for(task, 1)
        self.assertFalse(result.approved)
        self.assertEqual(self.ui.snapshot()["approvals"], [])

    async def test_host_origin_and_content_type_guards(self):
        self.ui.port = 0
        self.ui.start()
        base = f"http://127.0.0.1:{self.ui.port}"

        def request(headers, body=b'{"mode":"deny"}'):
            req = urllib.request.Request(base + "/api/access", data=body, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=3) as response:
                    return response.status
            except urllib.error.HTTPError as exc:
                return exc.code

        cases = [
            ({"Content-Type": "application/json", "Origin": "https://evil.example"}, 403),
            ({"Content-Type": "text/plain"}, 415),
            ({"Content-Type": "application/json", "Host": "evil.example"}, 403),
            ({"Content-Type": "application/json", "Origin": base}, 200),
        ]
        for headers, expected in cases:
            with self.subTest(headers=headers):
                self.assertEqual(await asyncio.to_thread(request, headers), expected)

    async def test_atomic_save_preserves_previous_file_on_replace_failure(self):
        session = self.ctx.sessions.create()
        session.append("user/message", text="before")
        target = session.save(self.cwd / "atomic.jsonl")
        before = target.read_bytes()
        session.append("user/message", text="after")
        with patch("mini_harness.session.os.replace", side_effect=OSError("disk failure")):
            with self.assertRaises(OSError):
                session.save(target)
        self.assertEqual(target.read_bytes(), before)
        self.assertFalse(list(self.cwd.glob(".session-*.tmp")))

    async def test_stop_between_send_and_coroutine_start_is_not_lost(self):
        self.ui.send("first")
        self.assertTrue(self.ui.interrupt())
        await asyncio.wrap_future(self.ui._turn_future)
        self.assertEqual(self.ui.session.events_of("turn/end")[-1].data["stopped"], "cancelled")
        self.ui.send("next")
        await asyncio.wrap_future(self.ui._turn_future)
        self.assertEqual(self.ui.session.events_of("turn/end")[-1].data["stopped"], "final")

    async def test_serve_honors_explicit_host_and_port(self):
        with patch.object(self.ui, "start"):
            serve(self.ctx, None, "localhost", 19991, asyncio.get_running_loop())
        self.assertEqual((self.ui.host, self.ui.port), ("localhost", 19991))

    async def test_folder_picker_updates_workspace(self):
        target = self.cwd / "chosen folder"
        target.mkdir()
        with patch("mini_harness.webui.choose_directory", return_value=str(target)) as picker:
            result = await asyncio.to_thread(self.ui.pick_cwd)
        picker.assert_called_once_with(str(self.cwd))
        self.assertFalse(result["cancelled"])
        self.assertEqual(self.ctx.agentLoop.cwd, target.resolve())
        self.assertFalse(self.ui._choosing_directory)

    async def test_folder_picker_cancel_keeps_workspace(self):
        with patch("mini_harness.webui.choose_directory", return_value=None):
            result = await asyncio.to_thread(self.ui.pick_cwd)
        self.assertTrue(result["cancelled"])
        self.assertEqual(self.ctx.agentLoop.cwd, self.cwd)

    async def test_existing_chat_cannot_change_workspace_but_can_preview_new_one(self):
        target = self.cwd / "new workspace"
        target.mkdir()
        current = self.ui.session
        current.append("user/message", text="existing conversation")
        with self.assertRaises(RuntimeError):
            self.ui.set_cwd(str(target))
        with patch("mini_harness.webui.choose_directory", return_value=str(target)) as picker:
            with self.assertRaises(RuntimeError):
                self.ui.pick_cwd()
            picker.assert_not_called()
            selected = self.ui.pick_cwd(preview=True)
        self.assertEqual(selected["cwd"], str(target.resolve()))
        self.assertIs(self.ui.session, current)
        self.assertEqual(self.ctx.agentLoop.cwd, self.cwd)

    async def test_new_chat_binds_directory_and_invalid_path_preserves_old_chat(self):
        self.ui.autosave = True
        old = self.ui.session
        old.append("user/message", text="keep this conversation")
        with self.assertRaises(ValueError):
            self.ui.new_session(str(self.cwd / "does-not-exist"))
        self.assertIs(self.ui.session, old)
        target = self.cwd / "new workspace"
        target.mkdir()
        state = self.ui.new_session(str(target))
        self.assertNotEqual(state["session"], old.id)
        self.assertEqual(state["transcript"], [])
        self.assertEqual(self.ui.ctx.agentLoop.cwd, target.resolve())
        self.assertEqual(self.ctx.agentLoop.cwd, self.cwd)
        self.assertTrue(old.source_path.is_file())

    async def test_open_workspace_uses_active_directory(self):
        if os.name == "nt":
            with patch("mini_harness.webui.os.startfile") as opener:
                self.assertTrue(self.ui.open_workspace()["ok"])
            opener.assert_called_once_with(str(self.cwd.resolve()))
        else:
            with patch("mini_harness.webui.subprocess.Popen") as opener:
                self.assertTrue(self.ui.open_workspace()["ok"])
            self.assertEqual(opener.call_args.args[0][-1], str(self.cwd.resolve()))

    async def test_folder_picker_blocks_parallel_send_and_recovers_after_error(self):
        def picker(initial):
            with self.assertRaises(RuntimeError):
                self.ui.send("cannot run while choosing")
            raise RuntimeError("desktop unavailable")
        with patch("mini_harness.webui.choose_directory", side_effect=picker):
            with self.assertRaises(RuntimeError):
                await asyncio.to_thread(self.ui.pick_cwd)
        self.assertFalse(self.ui._choosing_directory)
