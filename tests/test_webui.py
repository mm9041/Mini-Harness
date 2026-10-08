"""网页版 UI 测试。

用**真的 HTTP + 真的 SSE** 跑(服务起在随机端口,客户端用 ``urllib``)——
因为这一层的坑几乎全在"线程 ↔ 事件循环"的桥和协议细节上,断言内部状态测不出来。

最值得看的一条是**审批**:浏览器点"允许"要能真的让工具跑起来 —— 那证明
"策略在插件、问法在 UI"这个拆分是成立的,换个前端就换了个审批人。
"""

from __future__ import annotations

import asyncio
import base64
import json
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

from mini_harness.adapters.mock import ScriptedAdapter
from mini_harness.session import SessionEvent
from mini_harness.webui import WebUi, transcript_of, ui_message

from . import support
from .support import adapter_plugin_for as plugin_for
from .support import build_context_with as build_test_context


def post_json(base: str, path: str, payload: dict, timeout: float = 10) -> tuple[int, dict]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        base + path,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8") or "{}")


def get_json(base: str, path: str, timeout: float = 10) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(base + path, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8") or "{}")


def collect_sse(
    base: str, until, timeout: float = 15, connected=None
) -> list[dict]:
    """读 SSE 直到 ``until(payload)`` 为真(或超时)。在**别的线程**里跑,别堵住事件循环。

    ``connected`` 用来把「连接已建立」这件事**确定性地**告诉调用方 —— 否则调用方
    只能 sleep 一个魔法数字来等它连上,负载一高就丢事件(踩过)。
    """
    seen: list[dict] = []
    deadline = time.time() + timeout
    with urllib.request.urlopen(base + "/api/events", timeout=timeout) as response:
        if connected is not None:
            connected.set()
        for raw in response:
            if time.time() > deadline:
                break
            line = raw.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue  # 心跳(``: ping``)与注释
            payload = json.loads(line[len("data: ") :])
            seen.append(payload)
            if until(payload):
                break
    return seen


def kinds(events: list[dict]) -> list[str]:
    return [event.get("kind") for event in events]


class MappingTests(unittest.TestCase):
    """事件 → UI 消息的映射是纯函数,单独测。"""

    def test_known_events_get_their_own_kind(self) -> None:
        cases = {
            "user/message": "user",
            "assistant/message": "assistant",
            "tool/call": "tool-call",
            "tool/result": "tool-result",
            "turn/start": "turn-start",
            "turn/end": "turn-end",
            "step/start": "step-start",
            "step/end": "step-end",
        }
        for session_type, expected in cases.items():
            with self.subTest(session_type=session_type):
                message = ui_message(SessionEvent(seq=1, type=session_type, data={}))
                self.assertEqual(message["kind"], expected)

    def test_unknown_events_degrade_to_notice(self) -> None:
        """以后内核加了新事件,前端不改也能显示出来。"""
        message = ui_message(
            SessionEvent(seq=1, type="retry/scheduled", data={"attempt": 1, "delay": 0.5})
        )

        self.assertEqual(message["kind"], "notice")
        self.assertEqual(message["type"], "retry/scheduled")
        self.assertEqual(message["data"]["delay"], 0.5)

    def test_reply_completion_time_comes_from_persisted_event(self) -> None:
        from mini_harness.session import Session
        session = Session('timestamp-test')
        for seq, kind in enumerate(('assistant/message', 'turn/end'), 1):
            session.events.append(SessionEvent(seq=seq, type=kind, ts=1791115800 + seq,
                                               data={'text':'reply','stopped':'final'}))
        first = transcript_of(session)
        replayed = transcript_of(session)
        self.assertEqual([row['completed_at'] for row in first], [1791115801,1791115802])
        self.assertEqual(first, replayed)

    def test_long_fields_are_clipped(self) -> None:
        message = ui_message(
            SessionEvent(seq=1, type="compaction", data={"summary": "很长的摘要" * 500})
        )
        self.assertEqual(message["kind"], "command-result")
        self.assertLess(len(message["text"]), 400)
        self.assertNotIn("summary", message)

    def test_transcript_is_a_projection_of_the_log(self) -> None:
        from mini_harness.session import Session

        session = Session("s")
        session.append("turn/start")
        session.append("user/message", text="你好")
        session.append("step/start", index=1)
        session.append("assistant/message", text="在", tool_calls=[])
        session.append("turn/end", stopped="final", duration_ms=1200)

        transcript = transcript_of(session)

        self.assertEqual(
            [m["kind"] for m in transcript],
            ["turn-start", "user", "step-start", "assistant", "turn-end"],
        )
        self.assertEqual(transcript[1]["text"], "你好")


class WebServerTests(unittest.IsolatedAsyncioTestCase):
    """起真服务、发真请求。"""

    async def asyncSetUp(self) -> None:
        self.cwd = support.make_temp_dir()
        self.adapter = ScriptedAdapter([{"text": "网页版回答"}])
        self.ctx = build_test_context(plugin_for(self.adapter), self.cwd)
        self.ui: WebUi = self.ctx.webui
        self.ui.config = SimpleNamespace(
            task_cwd=self.cwd, model="mock-model", approval="ask"
        )
        self.ui.attach()
        self.ui.bind_loop(asyncio.get_running_loop())
        # 端口给 0:测试不该和别的实例抢固定端口
        # (踩过:后台还挂着一个 --web 实例占着 8770,整个测试文件就一起超时)
        _, port = self.ui.start()
        self.base = f"http://127.0.0.1:{port}"

    async def asyncTearDown(self) -> None:
        self.ui.stop()
        await support.remove_tree(self.cwd)

    # ---------------------------------------------------------------- 静态与状态
    async def test_upload_reaches_model_and_preserves_file_bytes(self) -> None:
        content = "上传内容：你好\n".encode("utf-8")
        status, _ = await asyncio.to_thread(post_json, self.base, "/api/message", {
            "text": "", "files": [{"name": "附件.txt", "data": base64.b64encode(content).decode()}]
        })
        self.assertEqual(status, 200)
        await asyncio.wrap_future(self.ui._turn_future)
        files = list((self.cwd / ".mini-harness" / "uploads").glob("*/*.txt"))
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0].read_bytes(), content)
        event = self.ui.session.events_of("user/message")[0]
        self.assertEqual(event.data["text"], "")
        self.assertEqual(event.data["attachments"][0]["path"], files[0].resolve().as_posix())
        message = ui_message(event)
        self.assertEqual(message["text"], "")
        self.assertEqual(message["attachments"], [{"name": "附件.txt", "size": len(content)}])
        self.assertIn(files[0].resolve().as_posix(), str(self.adapter.requests[0].messages))

    async def test_upload_rejects_bad_batch_before_writing(self) -> None:
        for bad in ({"name": "../escape.txt", "data": ""},
                    {"name": "bad.txt", "data": "!invalid!"}):
            status, _ = await asyncio.to_thread(post_json, self.base, "/api/message", {
                "text": "检查附件", "files": [{"name": "good.txt", "data": "YQ=="}, bad]
            })
            self.assertEqual(status, 400)
        self.assertFalse((self.cwd / ".mini-harness" / "uploads").exists())

    async def test_http_distinguishes_input_conflict_and_internal_failure(self):
        from unittest.mock import patch
        from mini_harness.webui_errors import UiConflictError
        worker = self.ui.target()
        for error, expected in ((ValueError("bad input"), 400),
                                (UiConflictError("busy"), 409),
                                (RuntimeError("internal failure"), 500)):
            with self.subTest(error=type(error).__name__):
                with patch.object(worker, "switch_reasoning", side_effect=error):
                    status, body = await asyncio.to_thread(
                        post_json, self.base, "/api/reasoning", {"effort": "none"})
                self.assertEqual(status, expected)
                self.assertIn(str(error), body["error"])

    async def test_upload_limit_and_duplicate_names(self) -> None:
        from mini_harness.webui import save_attachments
        from unittest.mock import patch
        with patch("mini_harness.webui.MAX_ATTACHMENT_BYTES", 1):
            with self.assertRaises(ValueError):
                save_attachments(self.cwd, [{"name": "big.txt", "data": "YWI="}])
        with self.assertRaises(ValueError):
            save_attachments(self.cwd, [{}] * 9)
        saved = save_attachments(self.cwd, [{"name": "same.txt", "data": data} for data in ("YQ==", "Yg==")])
        self.assertNotEqual(saved[0]["path"], saved[1]["path"])
        self.assertEqual([Path(item["path"]).read_bytes() for item in saved], [b"a", b"b"])

    async def test_serves_the_ui_and_the_state(self) -> None:
        page = await asyncio.to_thread(
            lambda: urllib.request.urlopen(self.base + "/", timeout=10).read().decode("utf-8")
        )
        self.assertIn("mini-harness", page)
        self.assertIn("EventSource", page)  # 前端确实靠 SSE 拿事件

        status, state = await asyncio.to_thread(get_json, self.base, "/api/state")
        self.assertEqual(status, 200)
        self.assertEqual(state["kind"], "state")
        self.assertTrue(state["session"])
        self.assertIn("task", state["tools"])
        self.assertEqual(state["access"], "ask")
        self.assertIn("操作前询问", str(state["access_labels"]))

    async def test_unknown_path_is_404(self) -> None:
        status, body = await asyncio.to_thread(get_json, self.base, "/api/nope")
        self.assertEqual(status, 404)
        self.assertIn("error", body)

    # ------------------------------------------------------------------ 一轮对话
    async def test_a_turn_streams_back_over_sse(self) -> None:
        ready = threading.Event()
        collector = asyncio.create_task(
            asyncio.to_thread(
                collect_sse,
                self.base,
                lambda m: m.get("kind") == "turn-end",
                15,
                ready,
            )
        )
        await asyncio.to_thread(ready.wait, 5)  # 等 SSE 真的连上,不靠 sleep 赌
        status, body = await asyncio.to_thread(
            post_json, self.base, "/api/message", {"text": "问一句"}
        )
        self.assertEqual(status, 200, body)

        events = await collector

        self.assertIn("turn-start", kinds(events))
        self.assertIn("delta", kinds(events))
        self.assertIn("assistant", kinds(events))
        self.assertEqual(events[-1]["kind"], "turn-end")
        self.assertIn("网页版回答", "".join(e.get("text") or "" for e in events))

        # 落进日志的仍是拼装好的整条消息,不是碎片
        assistant = self.ui.session.events_of("assistant/message")
        self.assertEqual(assistant[0].data["text"], "网页版回答")

    async def test_empty_message_is_rejected(self) -> None:
        status, body = await asyncio.to_thread(
            post_json, self.base, "/api/message", {"text": "   "}
        )
        self.assertEqual(status, 400)
        self.assertIn("text", body["error"])

    async def test_interrupt_stops_the_turn(self) -> None:
        slow = ScriptedAdapter([{"text": "很长" * 200}], stream_size=2, stream_delay=0.05)
        self.ctx.llm.register_adapter(slow.name, slow)
        self.ctx.llm.use(slow.name)

        ready = threading.Event()
        collector = asyncio.create_task(
            asyncio.to_thread(
                collect_sse, self.base, lambda m: m.get("kind") == "delta", 15, ready
            )
        )
        await asyncio.to_thread(ready.wait, 5)
        await asyncio.to_thread(post_json, self.base, "/api/message", {"text": "说很多"})
        await collector  # 等到真的开始流式了再中断
        status, body = await asyncio.to_thread(post_json, self.base, "/api/interrupt", {})

        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])

    async def test_second_message_while_busy_is_rejected(self) -> None:
        slow = ScriptedAdapter([{"text": "慢" * 50}], stream_size=2, stream_delay=0.05)
        self.ctx.llm.register_adapter(slow.name, slow)
        self.ctx.llm.use(slow.name)

        await asyncio.to_thread(post_json, self.base, "/api/message", {"text": "第一句"})
        status, body = await asyncio.to_thread(
            post_json, self.base, "/api/message", {"text": "第二句"}
        )

        self.assertEqual(status, 409)
        self.assertIn("还没跑完", body["error"])

    # ------------------------------------------------------------------ 运行时切换
    async def test_switch_access_from_the_browser(self) -> None:
        status, body = await asyncio.to_thread(
            post_json, self.base, "/api/access", {"mode": "allow"}
        )

        self.assertEqual(status, 200)
        self.assertEqual(body["previous"], "ask")
        self.assertEqual(self.ctx.approval.mode_for(self.ui.session), "allow")
        self.assertEqual(self.ctx.approval.mode, "ask")

    async def test_approval_endpoint_rejects_non_boolean_values(self):
        for value in ('false', 'true', 1, 0, None, [], {}):
            status, body = await asyncio.to_thread(post_json, self.base, '/api/approval',
                                                  {'id': 'unused', 'approved': value})
            self.assertEqual(status, 400)
            self.assertIn('布尔', body['error'])

    async def test_switch_model_from_the_browser(self) -> None:
        status, body = await asyncio.to_thread(
            post_json, self.base, "/api/model", {"name": "mock-strong"}
        )

        self.assertEqual(status, 200)
        self.assertEqual(body["model"], "mock-strong")
        self.assertEqual(self.adapter.model, "mock-strong")

    async def test_bad_model_name_is_rejected(self) -> None:
        status, _body = await asyncio.to_thread(
            post_json, self.base, "/api/model", {"name": "/access"}
        )
        self.assertEqual(status, 400)


class ApprovalThroughTheBrowserTests(unittest.IsolatedAsyncioTestCase):
    """浏览器当审批人 —— 这条最能说明"问法可换"。"""

    async def asyncSetUp(self) -> None:
        self.cwd = support.make_temp_dir()
        self.adapter = ScriptedAdapter(
            [
                {
                    "tool_calls": [
                        {
                            "name": "write",
                            "arguments": {"file_path": "note.txt", "content": "hi"},
                        }
                    ]
                },
                {"text": "写好了"},
            ]
        )
        self.ctx = build_test_context(plugin_for(self.adapter), self.cwd, approval="ask")
        self.ui: WebUi = self.ctx.webui
        self.ui.config = SimpleNamespace(
            task_cwd=self.cwd, model="mock-model", approval="ask"
        )
        self.ui.attach()
        self.ui.bind_loop(asyncio.get_running_loop())
        # 端口给 0:测试不该和别的实例抢固定端口
        # (踩过:后台还挂着一个 --web 实例占着 8770,整个测试文件就一起超时)
        _, port = self.ui.start()
        self.base = f"http://127.0.0.1:{port}"

    async def asyncTearDown(self) -> None:
        self.ui.stop()
        await support.remove_tree(self.cwd)

    async def test_browser_can_approve_and_the_tool_runs(self) -> None:
        ready = threading.Event()
        collector = asyncio.create_task(
            asyncio.to_thread(
                collect_sse, self.base, lambda m: m.get("kind") == "approval", 15, ready
            )
        )
        await asyncio.to_thread(ready.wait, 5)
        await asyncio.to_thread(
            post_json, self.base, "/api/message", {"text": "写个文件"}
        )
        events = await collector

        approval = events[-1]
        self.assertEqual(approval["tool"], "write")
        self.assertIn("note.txt", json.dumps(approval["arguments"]))
        self.assertFalse((self.cwd / "note.txt").exists(), "还没批准,文件不该存在")

        # 浏览器点"允许"
        ready = threading.Event()
        waiter = asyncio.create_task(
            asyncio.to_thread(
                collect_sse, self.base, lambda m: m.get("kind") == "turn-end", 15, ready
            )
        )
        self.assertTrue(await asyncio.to_thread(ready.wait, 5))
        status, body = await asyncio.to_thread(
            post_json, self.base, "/api/approval", {"id": approval["id"], "approved": True}
        )
        self.assertEqual(status, 200)
        self.assertTrue(body["ok"])

        events = await waiter
        self.assertEqual(events[-1]["stopped"], "final")
        self.assertEqual((self.cwd / "note.txt").read_text(encoding="utf-8"), "hi")

    async def test_browser_can_deny_and_nothing_happens(self) -> None:
        ready = threading.Event()
        collector = asyncio.create_task(
            asyncio.to_thread(
                collect_sse, self.base, lambda m: m.get("kind") == "approval", 15, ready
            )
        )
        await asyncio.to_thread(ready.wait, 5)
        await asyncio.to_thread(
            post_json, self.base, "/api/message", {"text": "写个文件"}
        )
        approval = (await collector)[-1]

        ready = threading.Event()
        waiter = asyncio.create_task(
            asyncio.to_thread(
                collect_sse, self.base, lambda m: m.get("kind") == "turn-end", 15, ready
            )
        )
        self.assertTrue(await asyncio.to_thread(ready.wait, 5))
        await asyncio.to_thread(
            post_json,
            self.base,
            "/api/approval",
            {"id": approval["id"], "approved": False},
        )
        events = await waiter

        self.assertFalse((self.cwd / "note.txt").exists(), "拒绝了就不该有副作用")
        results = [e for e in events if e.get("kind") == "tool-result"]
        self.assertTrue(results and results[0]["is_error"])
        self.assertIn("审批策略拒绝", results[0]["content"])

    async def test_unknown_approval_id_is_ignored(self) -> None:
        status, body = await asyncio.to_thread(
            post_json, self.base, "/api/approval", {"id": "不存在", "approved": True}
        )
        self.assertEqual(status, 200)
        self.assertFalse(body["ok"])

    async def test_terminal_seam_is_replaced_not_layered(self) -> None:
        """网页版装的是同一个审批接缝 —— 不是另开一条路。"""
        approval = self.ctx.approval
        self.assertIsNotNone(approval._approver)
        self.assertEqual(
            getattr(approval._approver, "__self__", None), self.ui.target()
        )


class TranscriptEndpointTests(unittest.IsolatedAsyncioTestCase):
    """新连上的浏览器靠 ``state`` 里的 transcript 补齐历史(刷新页面不丢对话)。"""

    async def asyncSetUp(self) -> None:
        self.cwd = support.make_temp_dir()
        self.adapter = ScriptedAdapter([{"text": "第一句"}, {"text": "第二句"}])
        self.ctx = build_test_context(plugin_for(self.adapter), self.cwd)
        self.ui: WebUi = self.ctx.webui
        self.ui.config = SimpleNamespace(task_cwd=self.cwd, model="m", approval="ask")
        self.ui.attach()
        self.ui.bind_loop(asyncio.get_running_loop())
        host, port = self.ui.start()
        self.base = f"http://{host}:{port}"

    async def asyncTearDown(self) -> None:
        self.ui.stop()
        await support.remove_tree(self.cwd)

    async def test_reconnect_sees_the_whole_conversation(self) -> None:
        for text in ("第一轮", "第二轮"):
            ready = threading.Event()
            waiter = asyncio.create_task(
                asyncio.to_thread(
                    collect_sse,
                    self.base,
                    lambda m: m.get("kind") == "turn-end",
                    15,
                    ready,
                )
            )
            await asyncio.to_thread(ready.wait, 5)
            await asyncio.to_thread(post_json, self.base, "/api/message", {"text": text})
            await waiter

        _status, state = await asyncio.to_thread(get_json, self.base, "/api/state")
        transcript = state["transcript"]
        users = [m["text"] for m in transcript if m["kind"] == "user"]

        self.assertEqual(users, ["第一轮", "第二轮"])
        self.assertEqual(len([m for m in transcript if m["kind"] == "turn-end"]), 2)


class NewSessionTests(unittest.IsolatedAsyncioTestCase):
    """「新建对话」要真的让浏览器清屏 —— 后端必须把空快照推出去。

    实测踩过:后端换了会话,但只把 status 返回给了 HTTP 响应(被丢弃),
    SSE 上什么都没发 → 页面纹丝不动,用户以为按钮坏了。
    """

    async def asyncSetUp(self) -> None:
        self.cwd = support.make_temp_dir()
        self.adapter = ScriptedAdapter([{"text": "旧会话的回答"}])
        self.ctx = build_test_context(plugin_for(self.adapter), self.cwd)
        self.ui: WebUi = self.ctx.webui
        self.ui.config = SimpleNamespace(task_cwd=self.cwd, model="m", approval="ask")
        self.ui.attach()
        self.ui.bind_loop(asyncio.get_running_loop())
        _, port = self.ui.start()
        self.base = f"http://127.0.0.1:{port}"

    async def asyncTearDown(self) -> None:
        self.ui.stop()
        await support.remove_tree(self.cwd)

    async def test_new_session_publishes_an_empty_snapshot(self) -> None:
        # 先跑一轮,让旧会话有内容
        ready = threading.Event()
        collector = asyncio.create_task(
            asyncio.to_thread(
                collect_sse, self.base, lambda m: m.get("kind") == "turn-end", 15, ready
            )
        )
        await asyncio.to_thread(ready.wait, 5)
        await asyncio.to_thread(post_json, self.base, "/api/message", {"text": "旧对话"})
        await collector
        self.assertGreater(len(self.ui.session.events), 0)
        old_id = self.ui.session.id

        # 新建会话:浏览器要在 SSE 上收到一份**空 transcript 的快照**
        waiter = asyncio.create_task(
            asyncio.to_thread(
                collect_sse,
                self.base,
                lambda m: m.get("kind") == "state" and not m.get("transcript"),
                15,
                threading.Event(),
            )
        )
        await asyncio.to_thread(post_json, self.base, "/api/new-session", {})
        events = await waiter

        snapshot = events[-1]
        self.assertEqual(snapshot["transcript"], [])
        self.assertNotEqual(snapshot["session"], old_id)
        self.assertEqual(len(self.ui.session.events), 0, "新会话应该是空的")

    async def test_new_session_keeps_busy_conversation_alive(self) -> None:
        self.ui._busy = True  # 模拟正在跑
        previous = self.ui.target()
        status, body = await asyncio.to_thread(
            post_json, self.base, "/api/new-session", {}
        )
        self.assertEqual(status, 200)
        self.assertTrue(previous._busy)
        self.assertFalse(self.ui._busy)
        self.assertNotEqual(body['session'], previous.session.id)
        previous._busy = False


if __name__ == "__main__":
    unittest.main()
