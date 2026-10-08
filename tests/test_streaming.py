"""流式输出测试。

分两个层次,刻意都不联网也不花 token:

* **纯逻辑** —— ``_StreamAccumulator`` 拼接增量,重点在工具调用碎片(名字一次、
  参数分多次、多个调用按 index 交错),这块最容易写错;
* **真链路** —— 起一个本地 HTTP 服务器吐 SSE,让适配器真的走一遍"线程读流 →
  队列 → async 生成器"的桥。这才是"流式到底通没通"的证据。
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from mini_harness import llm
from mini_harness.adapters.mock import scripted_plugin
from mini_harness.adapters.openai_compat import (
    OpenAICompatAdapter,
    StreamAccumulator,
    _JsonDepth,
)
from mini_harness.kernel import MODE_EMIT, Context, mount
from mini_harness.llm import GenerateRequest, GenerateResult, Message

from .support import adapter_plugin_for as plugin_for
from .support import build_context_with as build_test_context


class _PlainAdapter:
    """只有 ``generate``、没有 ``stream`` —— 用来验证退化路径。"""

    name = "plain"

    def __init__(self, result: GenerateResult) -> None:
        self.result = result
        self.calls = 0

    async def generate(self, request: GenerateRequest) -> GenerateResult:
        self.calls += 1
        return self.result


class AccumulatorTests(unittest.TestCase):
    def test_text_and_reasoning_deltas(self) -> None:
        acc = StreamAccumulator()

        events = acc.feed(
            {"choices": [{"delta": {"reasoning_content": "想一下", "content": "你好"}}]}
        )
        events += acc.feed({"choices": [{"delta": {"content": "世界"}}]})
        events += acc.feed({"choices": [{"delta": {}, "finish_reason": "stop"}]})
        result = acc.finish()

        self.assertEqual([e.text for e in events if e.text], ["你好", "世界"])
        self.assertEqual([e.reasoning for e in events if e.reasoning], ["想一下"])
        self.assertEqual(result.text, "你好世界")
        self.assertEqual(result.reasoning, "想一下")
        self.assertEqual(result.finish_reason, "stop")

    def test_tool_call_arguments_are_assembled_from_fragments(self) -> None:
        acc = StreamAccumulator()

        acc.feed(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "function": {"name": "pwsh", "arguments": '{"comm'},
                                }
                            ]
                        }
                    }
                ]
            }
        )
        acc.feed(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": 'and": "echo hi"}'}}
                            ]
                        }
                    }
                ]
            }
        )
        result = acc.finish()

        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].id, "call_1")
        self.assertEqual(result.tool_calls[0].name, "pwsh")
        self.assertEqual(result.tool_calls[0].arguments, {"command": "echo hi"})

    def test_interleaved_tool_calls_are_kept_by_index(self) -> None:
        acc = StreamAccumulator()

        acc.feed(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "id": "a", "function": {"name": "pwsh"}},
                                {"index": 1, "id": "b", "function": {"name": "read"}},
                            ]
                        }
                    }
                ]
            }
        )
        acc.feed(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 1, "function": {"arguments": '{"file_path": "x"}'}},
                                {"index": 0, "function": {"arguments": '{"command": "ls"}'}},
                            ]
                        }
                    }
                ]
            }
        )
        result = acc.finish()

        self.assertEqual([call.id for call in result.tool_calls], ["a", "b"])
        self.assertEqual(result.tool_calls[0].arguments, {"command": "ls"})
        self.assertEqual(result.tool_calls[1].arguments, {"file_path": "x"})

    def test_invalid_json_arguments_do_not_crash(self) -> None:
        acc = StreamAccumulator()
        acc.feed(
            {
                "choices": [
                    {"delta": {"tool_calls": [{"index": 0, "function": {"name": "pwsh", "arguments": "{oops"}}]}}
                ]
            }
        )
        result = acc.finish()

        self.assertEqual(result.tool_calls[0].arguments, {})
        self.assertEqual(result.tool_calls[0].raw_arguments, "{oops")

    def test_usage_and_empty_chunk(self) -> None:
        acc = StreamAccumulator()
        self.assertEqual(acc.feed({"choices": []}), [])
        acc.feed({"usage": {"total_tokens": 7}, "choices": []})
        self.assertEqual(acc.finish().usage, {"total_tokens": 7})


class _SseHandler(BaseHTTPRequestHandler):
    """按剧本吐 SSE 的假模型端点。

    ``scripts`` 是"每个请求一份剧本";用完了就回落到 ``chunks``。
    ``delay`` 是每帧之间的间隔 —— 有了它才能测出"边流边执行"的重叠收益。
    ``fail_times`` / ``fail_status`` 让**前 N 个请求直接返回 HTTP 错误** ——
    重试测试靠它造出真实的 5xx,而不是去断言内部状态。
    """

    chunks: list[str] = []
    scripts: list[list[str]] = []
    payloads: list[dict] = []
    delay: float = 0.0
    fail_times: int = 0
    fail_status: int = 500

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length)
        type(self).payloads.append(json.loads(body.decode("utf-8")))

        if type(self).fail_times > 0:
            type(self).fail_times -= 1
            payload = json.dumps(
                {"error": {"message": "test server failure", "type": "server_error"}}
            ).encode("utf-8")
            self.send_response(type(self).fail_status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        script = type(self).scripts.pop(0) if type(self).scripts else type(self).chunks

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        try:
            for chunk in script:
                if type(self).delay:
                    time.sleep(type(self).delay)
                self.wfile.write(f"data: {chunk}\n\n".encode("utf-8"))
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except ConnectionError:
            # 客户端提前断开(取消流)—— 预期行为,不要打 traceback。
            # 注意 handle_error 是 server 的方法,覆写在 handler 上不起作用。
            pass

    def log_message(self, *args) -> None:  # noqa: ARG002
        pass


class SseRoundTripTests(unittest.IsolatedAsyncioTestCase):
    """真的起 HTTP 服务器,验证"线程读 SSE → asyncio 队列"这座桥。"""

    async def asyncSetUp(self) -> None:
        _SseHandler.chunks = []
        _SseHandler.scripts = []
        _SseHandler.payloads = []
        _SseHandler.delay = 0.0
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _SseHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        self.adapter = OpenAICompatAdapter(
            api_key="test-key", base_url=self.base_url
        )

    async def asyncTearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def _request(self) -> GenerateRequest:
        return GenerateRequest(system="SYS", messages=[Message(role="user", content="hi")])

    async def test_streams_text_over_http(self) -> None:
        _SseHandler.chunks = [
            json.dumps({"choices": [{"delta": {"content": "你"}}]}),
            json.dumps({"choices": [{"delta": {"content": "好"}}]}),
            json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        ]

        events = [event async for event in self.adapter.stream(self._request())]

        self.assertEqual("".join(e.text for e in events if e.kind == "delta"), "你好")
        self.assertEqual(events[-1].kind, "done")
        self.assertEqual(events[-1].result.text, "你好")
        self.assertTrue(_SseHandler.payloads[0]["stream"])
        self.assertEqual(_SseHandler.payloads[0]["model"], "qwen-plus")

    async def test_streams_tool_call_over_http(self) -> None:
        _SseHandler.chunks = [
            json.dumps(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "function": {
                                            "name": "pwsh",
                                            "arguments": '{"command": "echo hi"}',
                                        },
                                    }
                                ]
                            }
                        }
                    ]
                }
            ),
            json.dumps({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}),
        ]

        events = [event async for event in self.adapter.stream(self._request())]

        result = events[-1].result
        self.assertEqual(result.tool_calls[0].name, "pwsh")
        self.assertEqual(result.tool_calls[0].arguments, {"command": "echo hi"})

    async def test_ignores_heartbeat_and_broken_lines(self) -> None:
        _SseHandler.chunks = [
            ": keep-alive",
            "not-json-garbage",
            json.dumps({"choices": [{"delta": {"content": "ok"}}]}),
        ]

        events = [event async for event in self.adapter.stream(self._request())]

        self.assertEqual(events[-1].result.text, "ok")

    async def test_early_exit_does_not_hang(self) -> None:
        """消费方提前退出(取消)时,生成器必须能立刻关掉,而不是等模型说完。"""
        _SseHandler.chunks = [
            json.dumps({"choices": [{"delta": {"content": f"{index}"}}]})
            for index in range(50)
        ]

        stream = self.adapter.stream(self._request())
        first = await stream.__anext__()
        self.assertEqual(first.text, "0")
        await stream.aclose()  # 不该挂住


class StreamProtocolTests(unittest.IsolatedAsyncioTestCase):
    def service(self, frames):
        class Adapter:
            name = 'protocol-test'
            async def stream(self, request):
                for frame in frames:
                    yield frame
        ctx = Context()
        self.addCleanup(ctx.dispose)
        mount(ctx, [llm.plugin()])
        ctx.llm.register_adapter('protocol-test', Adapter())
        return ctx

    async def test_missing_done_is_a_protocol_error(self):
        for frames in ([], [llm.StreamEvent('delta', text='partial')]):
            with self.subTest(frames=frames):
                ctx = self.service(frames)
                with self.assertRaisesRegex(llm.LLMError, '没有给出 done 帧') as raised:
                    [event async for event in ctx.llm.stream(GenerateRequest(system=''))]
                self.assertFalse(raised.exception.retryable)

    async def test_unknown_kind_and_missing_payload_are_rejected(self):
        for frame, message in ((llm.StreamEvent('tool-calls'), '未知的流事件类型'),
                               (llm.StreamEvent('tool_call'), '缺少有效 ToolCall'),
                               (llm.StreamEvent('done'), '缺少有效 GenerateResult')):
            with self.subTest(frame=frame):
                ctx = self.service([frame])
                with self.assertRaisesRegex(llm.LLMError, message):
                    [event async for event in ctx.llm.stream(GenerateRequest(system=''))]

    async def test_valid_empty_result_is_not_missing_done(self):
        ctx = self.service([llm.StreamEvent('done', result=GenerateResult())])
        events = [event async for event in ctx.llm.stream(GenerateRequest(system=''))]
        self.assertEqual(len(events), 1)

    async def test_early_close_is_not_a_missing_done_error(self):
        ctx = self.service([llm.StreamEvent('delta', text='partial')])
        stream = ctx.llm.stream(GenerateRequest(system=''))
        await anext(stream)
        await stream.aclose()

    async def test_task_cancellation_remains_cancellation(self):
        ready = asyncio.Event()
        class Adapter:
            name = 'blocked-test'
            async def stream(self, request):
                yield llm.StreamEvent('delta', text='partial')
                ready.set()
                await asyncio.Event().wait()
        ctx = self.service([])
        ctx.llm.register_adapter('protocol-test', Adapter())
        async def consume():
            return [event async for event in ctx.llm.stream(GenerateRequest(system=''))]
        task = asyncio.create_task(consume())
        await asyncio.wait_for(ready.wait(), 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task


class ServiceFallbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_falls_back_to_generate_when_adapter_has_no_stream(self) -> None:
        ctx = Context()
        mount(ctx, [llm.plugin()])
        adapter = _PlainAdapter(GenerateResult(text="abcdefghijklmnop"))
        ctx.llm.register_adapter(adapter.name, adapter)
        ctx.llm.use(adapter.name)

        events = [
            event
            async for event in ctx.llm.stream(
                GenerateRequest(system="s", messages=[Message(role="user", content="hi")])
            )
        ]

        self.assertEqual(adapter.calls, 1)
        self.assertEqual("".join(e.text for e in events), "abcdefghijklmnop")
        self.assertEqual(events[-1].kind, "done")

    async def test_delta_frames_are_broadcast_on_the_seam(self) -> None:
        ctx = Context()
        mount(ctx, [llm.plugin()])
        adapter = _PlainAdapter(GenerateResult(text="abcdefghijklmnop"))
        ctx.llm.register_adapter(adapter.name, adapter)
        ctx.llm.use(adapter.name)

        seen: list[str] = []
        ctx.on("llm/delta", lambda request, event: seen.append(event.text), mode=MODE_EMIT)

        [event async for event in ctx.llm.stream(self._request())]

        self.assertEqual("".join(seen), "abcdefghijklmnop")

    @staticmethod
    def _request() -> GenerateRequest:
        return GenerateRequest(system="s", messages=[Message(role="user", content="hi")])


class AgentStreamingTests(unittest.IsolatedAsyncioTestCase):
    """驱动器层的流式:帧序列、日志一致性、与非流式等价。"""

    async def asyncSetUp(self) -> None:
        from . import support

        self.cwd = support.make_temp_dir()

    async def asyncTearDown(self) -> None:
        from . import support

        await support.remove_tree(self.cwd)

    async def test_missing_done_records_error_instead_of_cancellation(self) -> None:
        class BrokenAdapter:
            name = 'missing-done'
            async def stream(self, request):
                yield llm.StreamEvent('delta', text='partial')
        ctx = build_test_context(plugin_for(BrokenAdapter()), self.cwd)
        self.addCleanup(ctx.dispose)
        session = ctx.sessions.create()
        with self.assertRaisesRegex(llm.LLMError, '没有给出 done 帧'):
            await ctx.agents.create(session).run('hello')
        ended = session.events_of('turn/end')
        self.assertEqual(len(ended), 1)
        self.assertEqual(ended[0].data['stopped'], 'error')
        self.assertIn('没有给出 done 帧', ended[0].data['error'])
        self.assertEqual(session.events_of('assistant/message'), [])

    async def test_frames_and_log_are_consistent(self) -> None:
        ctx = build_test_context(
            scripted_plugin([{"text": "一二三四五六七八九十"}]), self.cwd
        )
        frames: list[tuple[str, str]] = []
        ctx.on(
            "agent/assistant-stream",
            lambda session, frame: frames.append((frame.phase, frame.text)),
            mode=MODE_EMIT,
        )

        result = await ctx.agents.create(ctx.sessions.create()).run("说点什么")

        phases = [phase for phase, _ in frames]
        self.assertEqual(phases[0], "start")
        self.assertEqual(phases[-1], "end")
        self.assertGreater(phases.count("chunk"), 1)  # 真的分了多帧
        self.assertEqual("".join(text for phase, text in frames if phase == "chunk"), "一二三四五六七八九十")

        # 落日志的仍然是拼装好的整条消息,不是碎片
        assistant = result.session.events_of("assistant/message")
        self.assertEqual(len(assistant), 1)
        self.assertEqual(assistant[0].data["text"], "一二三四五六七八九十")

    async def test_streaming_and_non_streaming_produce_the_same_log(self) -> None:
        script = [
            {"tool_calls": [{"name": "pwsh", "arguments": {"command": "echo same"}}]},
            {"text": "结论:一致"},
        ]

        ctx_stream = build_test_context(scripted_plugin(list(script)), self.cwd, streaming=True)
        result_stream = await ctx_stream.agents.create(ctx_stream.sessions.create()).run("跑一次")

        ctx_plain = build_test_context(scripted_plugin(list(script)), self.cwd, streaming=False)
        result_plain = await ctx_plain.agents.create(ctx_plain.sessions.create()).run("跑一次")

        self.assertEqual(result_stream.text, result_plain.text)
        self.assertEqual(result_stream.stopped, "final")

        def shape(result):
            return [
                (event.type, event.data.get("name"), event.data.get("content"))
                for event in result.session.events
            ]

        self.assertEqual(shape(result_stream), shape(result_plain))


class EarlyToolCallTests(unittest.TestCase):
    """累加器:参数一闭合就报出可执行帧 —— 这是"边流边执行"的触发点。"""

    def test_reports_the_call_as_soon_as_arguments_close(self) -> None:
        acc = StreamAccumulator()

        not_yet = acc.feed(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "c1",
                                    "function": {"name": "pwsh", "arguments": '{"comm'},
                                }
                            ]
                        }
                    }
                ]
            }
        )
        self.assertEqual([event.kind for event in not_yet], [])  # 还没闭合

        ready = acc.feed(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": 'and": "echo hi"}'}}
                            ]
                        }
                    }
                ]
            }
        )
        self.assertEqual([event.kind for event in ready], ["tool_call"])
        self.assertEqual(ready[0].tool_call.id, "c1")
        self.assertEqual(ready[0].tool_call.arguments, {"command": "echo hi"})

        # 后续空 delta 不会重复报
        self.assertEqual(acc.feed({"choices": [{"delta": {}}]}), [])
        # finish 仍然给出完整列表(日志里的 assistant(tool_calls) 要列全)
        self.assertEqual([call.id for call in acc.finish().tool_calls], ["c1"])

    def test_does_not_report_before_id_and_name_arrive(self) -> None:
        acc = StreamAccumulator()
        events = acc.feed(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": '{"command": "ls"}'}}
                            ]
                        }
                    }
                ]
            }
        )
        self.assertEqual(events, [])  # 缺 id/name,不能派发(结果没法对应回去)

    def test_depth_scanner_ignores_braces_inside_strings(self) -> None:
        scanner = _JsonDepth()
        scanner.feed('{"command": "echo }"')
        self.assertFalse(scanner.closed)  # 字符串里的 } 不作数
        scanner.feed("}")
        self.assertTrue(scanner.closed)

    def test_depth_scanner_handles_escaped_quotes(self) -> None:
        scanner = _JsonDepth()
        scanner.feed('{"command": "echo \\"}"}')  # 内容里有转义引号
        self.assertTrue(scanner.closed)


class EarlyDispatchTests(unittest.IsolatedAsyncioTestCase):
    """驱动器:工具在流还没结束时就已经开跑(顺序与并发都验)。"""

    async def asyncSetUp(self) -> None:
        from . import support

        self.cwd = support.make_temp_dir()

    async def asyncTearDown(self) -> None:
        from . import support

        await support.remove_tree(self.cwd)

    async def test_tool_starts_before_the_stream_finishes(self) -> None:
        # 正文很长(分很多帧),工具调用排在它前面 —— 给足重叠的余地
        script = [
            {
                "tool_calls": [{"name": "read", "arguments": {"file_path": "missing.txt"}}],
                "text": "我还在继续说话…" * 40,
            }
        ]
        ctx = build_test_context(
            scripted_plugin(script, stream_size=8, stream_delay=0.01), self.cwd
        )

        timeline: list[str] = []
        ctx.on("llm/delta", lambda request, event: timeline.append("delta"), mode=MODE_EMIT)
        ctx.on(
            "agent/tool-ready",
            lambda session, call: timeline.append("ready"),
            mode=MODE_EMIT,
        )
        ctx.on(
            "tools/execute",
            lambda call, context, result: timeline.append("tool"),
            mode=MODE_EMIT,
        )

        await ctx.agents.create(ctx.sessions.create()).run("跑一下")

        self.assertIn("ready", timeline)
        self.assertEqual(timeline[0], "ready")  # 参数一闭合就报出来了
        self.assertGreater(timeline.count("delta"), 5)
        # 关键断言:工具执行发生在**最后一条正文增量之前** —— 也就是没等流结束
        self.assertLess(
            timeline.index("tool"),
            len(timeline) - 1 - timeline[::-1].index("delta"),
        )

    async def test_approval_gated_calls_are_not_prefetched(self) -> None:
        """需要审批的调用不提前派发 —— 否则审批提示会和流式输出抢终端。"""
        script = [
            {"tool_calls": [{"name": "write", "arguments": {"file_path": "a.txt", "content": "x"}}]},
            {"text": "写好了"},
        ]
        ctx = build_test_context(
            scripted_plugin(script, stream_delay=0.01), self.cwd, approval="allow"
        )

        timeline: list[str] = []
        ctx.on(
            "agent/tool-ready",
            lambda session, call: timeline.append("ready"),
            mode=MODE_EMIT,
        )
        ctx.on(
            "tools/execute",
            lambda call, context, result: timeline.append("tool"),
            mode=MODE_EMIT,
        )

        await ctx.agents.create(ctx.sessions.create()).run("写个文件")

        self.assertIn("ready", timeline)   # 帧照样报(UI 可以显示"准备调用…")
        self.assertIn("tool", timeline)    # 但执行发生在流结束之后
        self.assertLess(timeline.index("ready"), 2)


class EarlyDispatchTimingTests(unittest.IsolatedAsyncioTestCase):
    """真 SSE + 每帧延时:量一量"边流边执行"到底省了多少。

    两条路径除了 ``early_tools`` 之外完全相同 —— 同一个服务器、同一份剧本、同一个工具。
    剧本的设计是:工具参数在第一帧就补全,之后**还有约 1 秒的正文要流**,
    所以"工具没等流结束"这件事会直接体现成墙钟时间的差别。
    """

    DELAY = 0.2
    TOOL_SECONDS = 0.5

    async def asyncSetUp(self) -> None:
        from . import support

        _SseHandler.chunks = []
        _SseHandler.scripts = []
        _SseHandler.payloads = []
        _SseHandler.delay = 0.0
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _SseHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        self.cwd = support.make_temp_dir()

    async def asyncTearDown(self) -> None:
        from . import support

        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        await support.remove_tree(self.cwd)

    def _install_scripts(self) -> None:
        arguments = json.dumps({"command": f"sleep {self.TOOL_SECONDS}"})
        _SseHandler.scripts = [
            [
                json.dumps(
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "c1",
                                            "function": {"name": "slow_read", "arguments": arguments},
                                        }
                                    ]
                                }
                            }
                        ]
                    }
                ),
                json.dumps({"choices": [{"delta": {"content": "正文一"}}]}),
                json.dumps({"choices": [{"delta": {"content": "正文二"}}]}),
                json.dumps({"choices": [{"delta": {"content": "正文三"}}]}),
                json.dumps({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}),
            ],
            [json.dumps({"choices": [{"delta": {"content": "完成"}}]})],
        ]
        _SseHandler.delay = self.DELAY

    async def _run(self, early_tools: bool) -> float:
        self._install_scripts()
        adapter = OpenAICompatAdapter(api_key="k", base_url=self.base_url)
        ctx = build_test_context(
            plugin_for(adapter), self.cwd, early_tools=early_tools, shell_timeout=30
        )

        # A controlled, side-effect-free test tool opts into speculative execution.
        from mini_harness.tools import Tool, ToolResult
        async def slow_read(args, context):
            await asyncio.sleep(self.TOOL_SECONDS)
            return ToolResult('"exit_code": 0')
        ctx.tools.register(Tool("slow_read", "test read", handler=slow_read, safe_to_prefetch=True, permission="read"))
        started = time.perf_counter()
        result = await ctx.agents.create(ctx.sessions.create()).run("跑个慢命令")
        elapsed = time.perf_counter() - started

        self.assertEqual(result.stopped, "final")
        self.assertIn('"exit_code": 0', result.session.events_of("tool/result")[0].data["content"])
        return elapsed

    async def test_early_dispatch_overlaps_with_the_remaining_stream(self) -> None:
        sequential = await self._run(early_tools=False)
        early = await self._run(early_tools=True)

        print(
            f"\n    [边流边执行] 关闭 {sequential:.2f}s → 开启 {early:.2f}s"
            f"(工具 {self.TOOL_SECONDS}s,流里还剩约 1.0s)"
        )
        self.assertLess(
            early,
            sequential - 0.3,
            "工具应当与剩余流式重叠执行,总耗时不该是两者之和",
        )


if __name__ == "__main__":
    unittest.main()
