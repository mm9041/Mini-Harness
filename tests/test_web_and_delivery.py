import asyncio
import json
import socket
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from mini_harness.adapters.mock import ScriptedAdapter
from mini_harness.llm import ToolCall
from mini_harness import web_tools
from mini_harness.web_search import rss_results
from mini_harness.tools import ToolCallContext
from .support import adapter_plugin_for, build_context_with, make_temp_dir, remove_tree
from .test_webui import post_json, get_json


class WebToolsTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.root = make_temp_dir()
        self.ctx = build_context_with(adapter_plugin_for(ScriptedAdapter([])), self.root)
        self.session = self.ctx.sessions.create()

    async def asyncTearDown(self):
        self.ctx.dispose()
        await remove_tree(self.root)

    async def execute(self, name, args):
        return await self.ctx.tools.execute(ToolCall("probe", name, args), self.session, self.root)

    async def test_fetch_extracts_html_and_reports_truncation(self):
        text = '<title>页面 &amp; 标题</title><script>secret()</script><style>.hidden{}</style><h1>Hello</h1><p>' + '中' * 2000 + '</p>'
        with patch.object(web_tools, "download", return_value={"url": "https://example.com/final", "text": text, "content_type": "text/html"}):
            result = await self.execute("web_fetch", {"url": "https://example.com", "max_chars": 1000})
        self.assertFalse(result.is_error, result.content)
        data = json.loads(result.content)
        self.assertEqual(data["title"], "页面 & 标题")
        self.assertTrue(data["truncated"])
        self.assertEqual(len(data["content"]), 1000)
        self.assertNotIn("secret", data["content"])
        self.assertIn("Hello\n", data["content"])

    async def test_multiple_searches_keep_partial_success(self):
        def query(q, count, cancellation):
            if q == "bad":
                raise ValueError("provider unavailable")
            return {"query": q, "results": [{"title": "Python", "url": "https://python.org/", "snippet": "Official"}]}
        with patch.object(web_tools, "search_one", side_effect=query) as search:
            result = await self.execute("web_search", {"queries": ["Python", "bad"], "count": 2})
        self.assertEqual(search.call_count, 2)
        self.assertFalse(result.is_error)
        data = json.loads(result.content)["queries"]
        self.assertEqual(data[0]["results"][0]["url"], "https://python.org/")
        self.assertIn("error", data[1])

    async def test_rss_parser_and_input_limits(self):
        rss = '<rss><channel><item><title>A</title><link>https://example.com/a</link><description>Info</description></item><item><link>javascript:alert(1)</link></item></channel></rss>'
        self.assertEqual(len(rss_results(rss, "https://example.com")), 1)
        for args in ({"queries": []}, {"queries": [" "]}, {"queries": ["a"] * 6}, {"queries": ["a"], "count": True}):
            self.assertTrue((await self.execute("web_search", args)).is_error)
        with patch.object(web_tools, "download", return_value={"url": "https://example.com", "text": "data", "content_type": "application/pdf"}):
            self.assertTrue((await self.execute("web_fetch", {"url": "https://example.com"})).is_error)

    async def test_public_url_rejects_credentials_local_and_redirect(self):
        for url in ("file:///etc/passwd", "https://user:pass@example.com", "http://127.0.0.1/", "http://[::1]/", "http://169.254.169.254/"):
            with self.assertRaises(ValueError):
                web_tools.public_url(url)
        handler = web_tools.PublicRedirect()
        with self.assertRaises(ValueError):
            handler.redirect_request(urllib.request.Request("https://example.com"), None, 302, "Found", {}, "http://127.0.0.1/secret")

    async def test_invalid_ports_have_chinese_errors_before_dns_or_http(self):
        for url in ('http://example.com:abc/', 'https://example.com:65536/', 'http://example.com:-1/'):
            with self.subTest(url=url), patch.object(web_tools.socket, 'getaddrinfo') as dns, patch.object(web_tools.urllib.request, 'build_opener') as opener:
                with self.assertRaisesRegex(ValueError, 'URL 端口无效'):
                    web_tools.public_url(url)
                result = await self.execute('web_fetch', {'url':url})
                self.assertTrue(result.is_error)
                self.assertIn('URL 端口无效', result.content)
                self.assertNotIn('Port could not', result.content)
                dns.assert_not_called()
                opener.assert_not_called()
        with self.assertRaisesRegex(ValueError, 'URL 无效'):
            web_tools.public_url('http://[::1')

    async def test_public_url_uses_valid_explicit_or_default_ports(self):
        addresses = [(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('93.184.216.34', 80))]
        for url, port in (('http://example.com/',80), ('https://example.com/',443),
                          ('https://example.com:8443/',8443), ('https://example.com:65535/',65535)):
            with self.subTest(url=url), patch.object(web_tools.socket, 'getaddrinfo', return_value=addresses) as dns:
                self.assertEqual(web_tools.public_url(url), url)
                dns.assert_called_once_with('example.com', port, type=socket.SOCK_STREAM)

    async def test_real_http_fetch_and_response_size_limit(self):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                body = b'<title>Fixture</title><p>Read this page</p>'
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            def log_message(self, *args):
                pass
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        url = f"http://127.0.0.1:{server.server_port}/"
        try:
            # The production validator still rejects loopback; only this local fixture bypasses it.
            with patch.object(web_tools, "public_url", side_effect=lambda value: value):
                result = await self.execute("web_fetch", {"url": url})
                self.assertIn("Read this page", result.content)
                self.assertFalse(result.is_error)
                with patch.object(web_tools, "MAX_BYTES", 10):
                    self.assertTrue((await self.execute("web_fetch", {"url": url})).is_error)
        finally:
            await asyncio.to_thread(server.shutdown)
            server.server_close()
            worker.join()

    async def test_cancelled_network_tool_does_not_start_request(self):
        self.ctx.interrupt.request("stop")
        with patch.object(web_tools, "download") as download:
            result = await self.execute("web_fetch", {"url": "https://example.com"})
        self.assertTrue(result.is_error)
        download.assert_not_called()


class InteractionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.root = make_temp_dir()
        (self.root / "报告.html").write_text('<script>alert(1)</script><h1>交付完成</h1>', encoding="utf-8")
        self.adapter = ScriptedAdapter([
            {"tool_calls": [{"name": "ask_user_question", "arguments": {"question": "选择报告格式", "options": ["HTML", "文本"]}}]},
            {"tool_calls": [{"name": "present", "arguments": {"files": [{"file_path": "报告.html", "title": "分析报告"}]}}]},
            {"text": "已完成报告。"},
        ])
        self.ctx = build_context_with(adapter_plugin_for(self.adapter), self.root, compaction=False)
        self.ui = self.ctx.webui
        self.ui.port = 0
        self.ui.attach()
        self.ui.autosave = True
        self.ui.bind_loop(asyncio.get_running_loop())
        _, port = self.ui.start()
        self.base = f"http://127.0.0.1:{port}"

    async def asyncTearDown(self):
        self.ctx.interrupt.request("test cleanup")
        if self.ui._turn_future:
            await asyncio.wrap_future(self.ui._turn_future)
        await asyncio.to_thread(self.ui.stop)
        self.ctx.dispose()
        await remove_tree(self.root)

    async def pending(self):
        for _ in range(200):
            if self.ctx.userQuestions.pending:
                return next(iter(self.ctx.userQuestions.pending))
            await asyncio.sleep(.01)
        self.fail("question did not appear")

    async def test_http_question_answer_delivers_file_and_survives_reopen(self):
        status, _ = await asyncio.to_thread(post_json, self.base, "/api/message", {"text": "生成报告"})
        self.assertEqual(status, 200)
        question_id = await self.pending()
        _, state = await asyncio.to_thread(get_json, self.base, "/api/state")
        self.assertTrue(state["busy"])
        self.assertEqual(state["questions"][0]["id"], question_id)
        status, _ = await asyncio.to_thread(post_json, self.base, "/api/question", {"id": question_id, "answer": "HTML，增加摘要"})
        self.assertEqual(status, 200)
        await asyncio.wrap_future(self.ui._turn_future)
        self.assertIn("HTML，增加摘要", str(self.adapter.requests[1].messages))
        _, state = await asyncio.to_thread(get_json, self.base, "/api/state")
        self.assertEqual(state["questions"], [])
        artifact = next(m for m in state["transcript"] if m["kind"] == "artifact")
        saved = self.ui.session.source_path
        self.ui.new_session()
        self.ui.open_history(self.ui.history.key(saved))
        self.assertTrue(any(m.get("id") == artifact["id"] for m in self.ui.snapshot()["transcript"]))

        def fetch_artifact():
            with urllib.request.urlopen(self.base + "/api/artifacts/" + artifact["id"]) as response:
                return response.headers, response.read()
        headers, body = await asyncio.to_thread(fetch_artifact)
        self.assertEqual(headers.get_content_type(), "text/plain")
        self.assertIn("sandbox", headers["Content-Security-Policy"])
        self.assertIn("交付完成", body.decode())
        self.assertEqual((await asyncio.to_thread(post_json, self.base, "/api/question", {"id": question_id, "answer": "again"}))[0], 409)
        self.assertEqual((await asyncio.to_thread(get_json, self.base, "/api/artifacts/missing"))[0], 404)

    async def test_question_cancel_skip_invalid_and_headless(self):
        questions = self.ctx.userQuestions
        call = ToolCallContext("q", self.root, self.ui.session, self.ctx, self.ctx.interrupt)
        waiting = asyncio.create_task(questions.ask({"question": "继续？"}, call))
        question_id = await self.pending()
        with self.assertRaises(ValueError):
            questions.answer(question_id, "")
        questions.answer(question_id, "", True)
        self.assertTrue((await waiting).is_error)
        waiting = asyncio.create_task(questions.ask({"question": "再确认？"}, call))
        await self.pending()
        self.ctx.interrupt.request("stop")
        self.assertTrue((await waiting).is_error)
        self.assertEqual(questions.pending, {})
        self.ctx.interrupt.reset()
        questions.browser = False
        self.assertTrue((await questions.ask({"question": "headless?"}, call)).is_error)
        questions.responder = lambda message: "terminal answer"
        self.assertIn("terminal answer", (await questions.ask({"question": "terminal?"}, call)).content)

    async def test_present_rejects_missing_and_outside_files_atomically(self):
        for file_path in ("missing.txt", "../outside.txt"):
            result = await self.ctx.tools.execute(ToolCall("p", "present", {"files": [{"file_path": "报告.html"}, {"file_path": file_path}]}), self.ui.session, self.root)
            self.assertTrue(result.is_error)
        self.assertEqual(self.ui.session.events_of("artifact/presented"), [])
        result = await self.ctx.tools.execute(ToolCall("p", "present", {"files": [{"file_path": "报告.html"}]}), self.ui.session, self.root)
        self.assertFalse(result.is_error)
        item = self.ui.session.events_of("artifact/presented")[0].data
        (self.root / "报告.html").unlink()
        with self.assertRaises(ValueError):
            self.ui.artifact(item["id"])

    async def test_tool_registration_and_prefetch_policy(self):
        names = {schema.name for schema in self.ctx.tools.schemas()}
        self.assertTrue({"web_search", "web_fetch", "ask_user_question", "present"} <= names)
        self.assertFalse({"read_file", "write_file", "shell"} & names)
        self.assertTrue(self.ctx.tools.can_prefetch("web_search"))
        self.assertFalse(self.ctx.tools.can_prefetch("ask_user_question"))
        self.assertFalse(self.ctx.tools.can_prefetch("present"))

    async def test_question_timeout_is_not_confirmation(self):
        self.ctx.userQuestions.timeout = .01
        result = await self.ctx.tools.execute(ToolCall("q", "ask_user_question", {"question": "确认？"}), self.ui.session, self.root)
        self.assertTrue(result.is_error)
        self.assertTrue(json.loads(result.content)["skipped"])
        self.assertEqual(self.ctx.userQuestions.pending, {})
        self.assertEqual(self.ui.snapshot()["questions"], [])

    async def test_pdf_download_headers_and_other_session_isolation(self):
        content = b"%PDF-1.4\n% fixture bytes\n"
        (self.root / "test.pdf").write_bytes(content)
        result = await self.ctx.tools.execute(ToolCall("p", "present", {"files": [{"file_path": "test.pdf"}]}), self.ui.session, self.root)
        artifact = json.loads(result.content)["artifacts"][0]
        def read(suffix):
            with urllib.request.urlopen(self.base + "/api/artifacts/" + artifact["id"] + suffix) as response:
                return response.headers, response.read()
        headers, body = await asyncio.to_thread(read, "")
        self.assertEqual(headers.get_content_type(), "application/pdf")
        self.assertEqual(body, content)
        headers, body = await asyncio.to_thread(read, "?download=1")
        self.assertIn("attachment", headers["Content-Disposition"])
        self.assertEqual(body, content)
        self.ui.new_session()
        self.assertEqual((await asyncio.to_thread(get_json, self.base, "/api/artifacts/" + artifact["id"]))[0], 404)
