"""重试插件测试。

分三层,每层都在验一件具体的事:

1. **分类** —— 靠结构化字段(状态码),不靠猜消息;唯一必须看文本的是"上下文超了",
   而且它必须排在状态码判断**之前**(各家都用 400 报它,先看状态码就永远归进"不该重试");
2. **决策与退避** —— 有界 / 无限两档、指数退避封顶、认不出就不重试;
3. **端到端** —— 用本地假 HTTP 服务器**真的返回 500**,断言请求次数、日志内容与顺序
   (尤其是"排定的重试要在退避之前先落日志"),以及取消能否中断退避。
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import unittest
from http.server import ThreadingHTTPServer

from mini_harness.adapters.openai_compat import OpenAICompatAdapter
from mini_harness.llm import GenerateRequest, GenerateResult, LLMError, Message
from mini_harness.retry import RetryDecision, RetryPolicy, RetryService

from . import support
from .support import adapter_plugin_for as plugin_for
from .support import build_context_with as build_test_context
from .test_streaming import _SseHandler


class ClassificationTests(unittest.TestCase):
    def test_chinese_reverse_context_bound_wording(self):
        for message in ('输入长度超出模型的最大上下文长度', '请求超过了当前模型支持的上下文窗口上限',
                        '超过允许的最大上下文容量', '输入已经超出该模型的上下文限制',
                        '请求超过模型上下文'):
            with self.subTest(message=message):
                self.assertEqual(self.service.classify(LLMError(message, status=400)), 'overflow')
        for message in ('超过最大请求次数，请检查上下文', '超出速率限制，上下文长度正常',
                        '超过重试次数而非上下文长度', '输入没有超过模型的上下文长度',
                        'max_tokens 超过允许的输出上限'):
            with self.subTest(message=message):
                self.assertEqual(self.service.classify(LLMError(message, status=400)), 'fatal')

    def test_parameter_validation_and_non_context_limits_are_not_overflow(self):
        for message in ('max_tokens must be less than 8192', 'token limit reached',
                        'context window size must be positive', '超出最大请求频率',
                        'input exceeds maximum allowed value', 'reduce the length of tool name'):
            with self.subTest(message=message):
                self.assertEqual(self.service.classify(LLMError(message, status=400)), 'fatal')

    def test_structured_context_code_and_priority(self):
        self.assertEqual(self.service.classify(LLMError('invalid request', status=400, provider_code='context_length_exceeded')), 'overflow')
        for status, expected in ((401, 'fatal'), (403, 'fatal'), (429, 'retryable'), (503, 'retryable')):
            self.assertEqual(self.service.classify(LLMError('maximum context length', status=status)), expected)
        self.assertEqual(self.service.classify(LLMError('maximum context length', status=429, provider_code='insufficient_quota')), 'fatal')

    def test_http_error_keeps_provider_fields_and_ignores_unrelated_body(self):
        import io
        import urllib.error
        from mini_harness.adapters.openai_compat import _wrap_transport_exception
        for code, message, expected in (('context_length_exceeded', '请求无效', 'overflow'),
                                       ('invalid_parameter', 'max_tokens must be less than 8192', 'fatal')):
            body = json.dumps({'error':{'code':code,'type':'invalid_request_error','message':message},
                               'echo':'maximum context length'}).encode()
            error = urllib.error.HTTPError('https://example.invalid', 400, 'bad', {}, io.BytesIO(body))
            wrapped = _wrap_transport_exception(error, error.url, 30)
            self.assertEqual(wrapped.provider_code, code)
            self.assertEqual(wrapped.provider_message, message)
            self.assertEqual(self.service.classify(wrapped), expected)

    def setUp(self) -> None:
        self.service = RetryService(RetryPolicy(max_attempts=3))

    def test_transient_statuses_are_retryable(self) -> None:
        for status in (429, 500, 502, 503, 504, 408):
            with self.subTest(status=status):
                self.assertEqual(
                    self.service.classify(LLMError("boom", status=status)), "retryable"
                )

    def test_client_errors_are_fatal(self) -> None:
        for status in (400, 401, 403, 404, 422):
            with self.subTest(status=status):
                self.assertEqual(
                    self.service.classify(LLMError("boom", status=status)), "fatal"
                )

    def test_connection_errors_are_retryable_without_a_status(self) -> None:
        self.assertEqual(
            self.service.classify(LLMError("连接被重置", retryable=True)), "retryable"
        )

    def test_unknown_errors_are_not_retried(self) -> None:
        """认不出来就快失败 —— 别在不确定的地方空转。"""
        self.assertEqual(self.service.classify(ValueError("谁知道呢")), "fatal")

    def test_overflow_wins_over_the_status_code(self) -> None:
        """上下文溢出通常也是 400,但它的处理不是"放弃"而是"先压缩再试"。

        顺序错了(先看状态码)它就永远被判成 fatal —— 这条测试就是在钉这个顺序。
        """
        exc = LLMError(
            "This model's maximum context length is 131072 tokens", status=400
        )
        self.assertEqual(self.service.classify(exc), "overflow")

    def test_explicit_overflow_kind(self) -> None:
        self.assertEqual(
            self.service.classify(LLMError("随便什么原因", kind="overflow")), "overflow"
        )

    def test_chinese_overflow_wording(self) -> None:
        self.assertEqual(
            self.service.classify(LLMError("输入的上下文长度超过模型上限")), "overflow"
        )


class DecisionTests(unittest.TestCase):
    def test_retries_until_the_budget_runs_out(self) -> None:
        service = RetryService(RetryPolicy(max_attempts=3, base_delay=0.0, jitter=0))

        self.assertTrue(service.decide(LLMError("x", status=500), 1).will_retry)
        self.assertTrue(service.decide(LLMError("x", status=500), 2).will_retry)
        third = service.decide(LLMError("x", status=500), 3)
        self.assertEqual(third.kind, "give-up")
        self.assertIn("上限", third.reason)

    def test_always_mode_ignores_the_budget(self) -> None:
        service = RetryService(RetryPolicy(max_attempts=2, always=True, base_delay=0.0))

        self.assertTrue(service.decide(LLMError("x", status=500), 99).will_retry)

    def test_fatal_never_retries(self) -> None:
        service = RetryService(RetryPolicy(max_attempts=99))
        decision = service.decide(LLMError("额度用尽", status=403), 1)

        self.assertEqual(decision.kind, "give-up")
        self.assertEqual(decision.delay, 0.0)

    def test_delay_is_exponential_and_capped(self) -> None:
        service = RetryService(
            RetryPolicy(base_delay=1.0, max_delay=4.0, jitter=0.0)
        )

        self.assertEqual(service.delay_for(1), 1.0)
        self.assertEqual(service.delay_for(2), 2.0)
        self.assertEqual(service.delay_for(3), 4.0)
        self.assertEqual(service.delay_for(9), 4.0)  # 封顶

    def test_jitter_keeps_the_delay_in_a_band(self) -> None:
        service = RetryService(RetryPolicy(base_delay=1.0, jitter=0.25))
        delays = {round(service.delay_for(1), 3) for _ in range(20)}

        self.assertGreater(len(delays), 1, "抖动没生效")
        for delay in delays:
            self.assertGreaterEqual(delay, 0.7)
            self.assertLessEqual(delay, 1.3)

    def test_zero_delay_disables_the_wait(self) -> None:
        self.assertEqual(RetryService(RetryPolicy(base_delay=0.0)).delay_for(3), 0.0)


class _ServerCase(unittest.IsolatedAsyncioTestCase):
    """起一个假模型端点,并把它接到真实的适配器上。"""

    async def asyncSetUp(self) -> None:
        self.cwd = support.make_temp_dir()
        _SseHandler.chunks = []
        _SseHandler.scripts = []
        _SseHandler.payloads = []
        _SseHandler.delay = 0.0
        _SseHandler.fail_times = 0
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

    def make_ctx(self, **overrides):
        # 注意用 json.dumps 拼 SSE:直接 f-string 插 dict 会得到单引号的 Python repr,
        # 那不是合法 JSON —— 而解析器**按设计静默跳过**坏行,于是表现成"空响应",
        # 排查起来会绕远路(踩过)。
        _SseHandler.chunks = [
            json.dumps({"choices": [{"delta": {"content": "好了"}}]})
        ]
        adapter = OpenAICompatAdapter(api_key="k", base_url=self.base_url)
        settings = {"retry_base_delay": 0.0, **overrides}  # 测试不等真退避
        return build_test_context(plugin_for(adapter), self.cwd, **settings)


class RetryThroughTheLoopTests(_ServerCase):
    async def test_retries_then_succeeds(self) -> None:
        _SseHandler.fail_times = 2  # 前两次 500
        ctx = self.make_ctx()
        session = ctx.sessions.create()

        result = await ctx.agents.create(session).run("问一句")

        self.assertEqual(result.stopped, "final")
        self.assertEqual(result.text, "好了")
        self.assertEqual(len(_SseHandler.payloads), 3, "应当一共请求 3 次")

        scheduled = session.events_of("retry/scheduled")
        self.assertEqual([event.data["attempt"] for event in scheduled], [1, 2])
        self.assertEqual(session.events_of("retry/recovered")[0].data["attempts"], 3)
        self.assertEqual(session.events_of("retry/gave-up"), [])

    async def test_scheduled_event_is_logged_before_the_backoff(self) -> None:
        """**先记账再退避** —— 进程在等待期间被杀,日志也说得清发生过什么。

        做法:把 wait 换成探针,在它被调用的那一刻检查日志里有没有 retry/scheduled。
        """
        _SseHandler.fail_times = 1
        ctx = self.make_ctx(retry_base_delay=5.0)  # 真等就太慢了,靠探针绕开
        session = ctx.sessions.create()
        seen: list[bool] = []
        original_wait = ctx.retry.wait

        async def probe(delay, cancellation=None):  # noqa: ANN001, ARG001
            seen.append(any(e.type == "retry/scheduled" for e in session.events))
            return await original_wait(0.0, cancellation)

        ctx.retry.wait = probe  # type: ignore[method-assign]
        await ctx.agents.create(session).run("问一句")

        self.assertEqual(seen, [True], "退避开始时日志里必须已经有排定的重试")

    async def test_client_error_gives_up_immediately(self) -> None:
        """403(额度用尽)重试也没用:只请求一次,并如实记一笔放弃。"""
        _SseHandler.fail_times = 5
        _SseHandler.fail_status = 403
        ctx = self.make_ctx()
        session = ctx.sessions.create()

        with self.assertRaises(LLMError):
            await ctx.agents.create(session).run("问一句")

        self.assertEqual(len(_SseHandler.payloads), 1)
        self.assertEqual(session.events_of("retry/scheduled"), [])
        self.assertEqual(len(session.events_of("retry/gave-up")), 1)
        self.assertEqual(session.events_of("turn/end")[-1].data["stopped"], "error")

    async def test_gives_up_after_the_budget(self) -> None:
        _SseHandler.fail_times = 99
        ctx = self.make_ctx(retry_attempts=3)
        session = ctx.sessions.create()

        with self.assertRaises(LLMError):
            await ctx.agents.create(session).run("问一句")

        self.assertEqual(len(_SseHandler.payloads), 3)
        self.assertEqual(len(session.events_of("retry/scheduled")), 2)  # 第 1、2 次之后
        self.assertIn("上限", session.events_of("retry/gave-up")[0].data["reason"])

    async def test_always_mode_keeps_going(self) -> None:
        """always 模式:即使超过 max_attempts 也继续,直到成功。"""
        _SseHandler.fail_times = 4
        ctx = self.make_ctx(retry_attempts=2, retry_always=True)
        session = ctx.sessions.create()

        result = await ctx.agents.create(session).run("问一句")

        self.assertEqual(result.text, "好了")
        self.assertEqual(len(_SseHandler.payloads), 5)

    async def test_disabling_retry_means_one_attempt(self) -> None:
        _SseHandler.fail_times = 99
        ctx = self.make_ctx(retry_attempts=0)
        session = ctx.sessions.create()

        with self.assertRaises(LLMError):
            await ctx.agents.create(session).run("问一句")

        self.assertEqual(len(_SseHandler.payloads), 1)
        self.assertIsNone(ctx.get("retry", None))

    async def test_cancellation_interrupts_the_backoff(self) -> None:
        """退避期间被取消:立刻停下,不留悬空请求,历史按取消收尾。"""
        _SseHandler.fail_times = 99
        ctx = self.make_ctx(retry_attempts=99, retry_base_delay=30.0)
        session = ctx.sessions.create()

        loop = asyncio.get_running_loop()
        loop.call_later(0.2, lambda: ctx.interrupt.request("退避时按了 Ctrl+C"))

        started = time.perf_counter()
        result = await ctx.agents.create(session).run("问一句")
        elapsed = time.perf_counter() - started

        self.assertEqual(result.stopped, "cancelled")
        self.assertLess(elapsed, 10, "取消应当立刻中断退避,而不是等满 30 秒")
        self.assertEqual(len(_SseHandler.payloads), 1)
        self.assertEqual(len(session.events_of("retry/scheduled")), 1)
        # 结构平衡:取消之后日志仍然是合法的
        self.assertEqual(
            len(session.events_of("turn/start")), len(session.events_of("turn/end"))
        )


class SingleAttemptForDirectCallsTests(unittest.IsolatedAsyncioTestCase):
    """重试是**驱动器的政策**,不是适配器的 —— 直接调 ctx.llm.stream() 仍是单次。"""

    async def asyncSetUp(self) -> None:
        self.cwd = support.make_temp_dir()

    async def asyncTearDown(self) -> None:
        await support.remove_tree(self.cwd)

    async def test_direct_stream_is_not_retried(self) -> None:
        class AlwaysBroken:
            name = "broken"
            model = "broken"

            def __init__(self) -> None:
                self.calls = 0

            async def generate(self, request):  # noqa: ANN001, ARG002
                self.calls += 1
                raise LLMError("服务器炸了", status=500)

        adapter = AlwaysBroken()
        ctx = build_test_context(plugin_for(adapter), self.cwd, retry_attempts=5)
        request = GenerateRequest(system="s", messages=[Message(role="user", content="hi")])

        with self.assertRaises(LLMError):
            async for _event in ctx.llm.stream(request):
                pass

        self.assertEqual(adapter.calls, 1)

    async def test_the_loop_does_retry_the_same_adapter(self) -> None:
        class Flaky:
            name = "flaky"
            model = "flaky"

            def __init__(self) -> None:
                self.calls = 0

            async def generate(self, request):  # noqa: ANN001, ARG002
                self.calls += 1
                if self.calls < 3:
                    raise LLMError("暂时性故障", status=503)
                return GenerateResult(text="第三次成功")

        adapter = Flaky()
        ctx = build_test_context(
            plugin_for(adapter), self.cwd, retry_attempts=3, retry_base_delay=0.0
        )

        result = await ctx.agents.create(ctx.sessions.create()).run("问一句")

        self.assertEqual(adapter.calls, 3)
        self.assertEqual(result.text, "第三次成功")


class OverflowTests(unittest.IsolatedAsyncioTestCase):
    """上下文溢出 → 先压缩 → 重建请求再试。"""

    async def asyncSetUp(self) -> None:
        self.cwd = support.make_temp_dir()

    async def asyncTearDown(self) -> None:
        await support.remove_tree(self.cwd)

    async def test_max_tokens_parameter_error_never_summarizes_or_retries(self):
        from unittest.mock import AsyncMock, patch
        class InvalidParameter:
            name = 'invalid-parameter'
            model = 'test'
            async def generate(self, request):
                raise LLMError('max_tokens must be less than 8192', status=400)
        ctx = build_test_context(plugin_for(InvalidParameter()), self.cwd, max_history_tokens=10**9)
        self.addCleanup(ctx.dispose)
        session = ctx.sessions.create()
        self._long_session(session)
        with patch.object(ctx.compaction, 'condense_now', new_callable=AsyncMock) as compact:
            with self.assertRaises(LLMError):
                await ctx.agents.create(session).run('continue')
            compact.assert_not_awaited()
        self.assertEqual(session.compactions(), [])
        self.assertEqual(session.events_of('compaction/start'), [])
        self.assertEqual(session.events_of('retry/scheduled'), [])

    def _long_session(self, session, turns: int = 6) -> None:
        for index in range(turns):
            session.append("turn/start")
            session.append("user/message", text=f"第 {index} 轮" + "问" * 300)
            session.append("step/start", index=1)
            session.append("assistant/message", text="答" * 300, tool_calls=[])
            session.append("step/end", index=1)
            session.append("turn/end", stopped="final")

    async def test_persistent_overflow_is_bounded_even_with_retry_always(self):
        class AlwaysOverflow:
            name = "persistent-overflow"
            model = "m"
            calls = 0

            async def generate(self, request):
                self.calls += 1
                if "## Pending Jobs" in (request.messages[-1].content or ""):
                    return GenerateResult(text="摘要")
                raise LLMError("context length exceeded", status=400)

        adapter = AlwaysOverflow()
        ctx = build_test_context(plugin_for(adapter), self.cwd, retry_always=True,
                                 max_history_tokens=10**9, retry_base_delay=0)
        session = ctx.sessions.create()
        self._long_session(session)
        with self.assertRaises(LLMError):
            await ctx.agents.create(session).run("继续")
        self.assertEqual(adapter.calls, 3)
        self.assertEqual(len(session.compactions()), 1)
        self.assertIn("上限", session.events_of("retry/gave-up")[-1].data["reason"])

    async def test_overflow_condenses_then_retries_smaller(self) -> None:
        class OverflowOnce:
            name = "overflow-once"
            model = "m"

            def __init__(self) -> None:
                self.calls = 0
                self.sizes: list[int] = []

            async def generate(self, request):  # noqa: ANN001
                self.calls += 1
                self.sizes.append(sum(len(m.content or "") for m in request.messages))
                if self.calls == 1:
                    raise LLMError(
                        "maximum context length exceeded", status=400
                    )
                return GenerateResult(text="压缩之后成功了")

        adapter = OverflowOnce()
        ctx = build_test_context(
            plugin_for(adapter),
            self.cwd,
            retry_base_delay=0.0,
            max_history_tokens=10**9,  # 关掉自动触发,只让溢出路径来压
            keep_recent_messages=2,
        )
        session = ctx.sessions.create()
        self._long_session(session)

        result = await ctx.agents.create(session).run("再来一句")

        self.assertEqual(result.text, "压缩之后成功了")
        # 3 次调用的账:① 溢出的那次 step 请求 ② 压缩要用的**摘要**请求
        # ③ 压缩之后重试的 step 请求。别漏算第 ②——摘要本身也是一次模型调用。
        self.assertEqual(adapter.calls, 3)
        self.assertTrue(session.compactions(), "溢出必须先压缩")
        self.assertLess(
            adapter.sizes[2], adapter.sizes[0], "重试那次的请求应当比原来小"
        )
        self.assertEqual(len(session.events_of("retry/scheduled")), 1)
        self.assertEqual(
            session.events_of("retry/scheduled")[0].data["kind"], "overflow"
        )

    async def test_overflow_gives_up_when_nothing_can_be_compressed(self) -> None:
        class AlwaysOverflow:
            name = "overflow"
            model = "m"

            def __init__(self) -> None:
                self.calls = 0

            async def generate(self, request):  # noqa: ANN001, ARG002
                self.calls += 1
                raise LLMError("context length exceeded", status=400)

        adapter = AlwaysOverflow()
        ctx = build_test_context(plugin_for(adapter), self.cwd, retry_base_delay=0.0)
        session = ctx.sessions.create()  # 空会话:没有可压的东西

        with self.assertRaises(LLMError):
            await ctx.agents.create(session).run("问一句")

        self.assertEqual(adapter.calls, 1, "压不动就不该反复试")
        gave_up = session.events_of("retry/gave-up")
        self.assertEqual(len(gave_up), 1)
        self.assertIn("没有可压缩", gave_up[0].data["reason"])


class KeepAliveTests(unittest.TestCase):
    """``wait`` 的边角:延迟为 0、已取消、带取消令牌。"""

    def setUp(self) -> None:
        self.service = RetryService(RetryPolicy(base_delay=0.0))

    async def _run(self, coro):
        return await coro

    def test_zero_delay_returns_immediately(self) -> None:
        self.assertTrue(asyncio.run(self.service.wait(0.0, None)))

    def test_already_cancelled_skips_the_wait(self) -> None:
        from mini_harness.interrupt import InterruptService

        token = InterruptService()
        token.request("先取消了")
        self.assertFalse(asyncio.run(self.service.wait(5.0, token)))

    def test_cancellation_wakes_the_wait(self) -> None:
        """取消应当**立刻**唤醒退避,而不是等满延迟。``wait`` 返回 False = 被取消。"""
        from mini_harness.interrupt import InterruptService

        token = InterruptService()

        async def scenario() -> tuple[bool, float]:
            loop = asyncio.get_running_loop()
            loop.call_later(0.05, lambda: token.request("取消"))
            started = time.perf_counter()
            result = await self.service.wait(30.0, token)
            return result, time.perf_counter() - started

        result, elapsed = asyncio.run(scenario())

        self.assertFalse(result, "被取消时 wait 应当返回 False")
        self.assertLess(elapsed, 5, "不该等满 30 秒")

    def test_no_cancellation_sleeps_the_full_delay(self) -> None:
        async def scenario() -> bool:
            started = time.perf_counter()
            ok = await self.service.wait(0.05, None)
            return ok and (time.perf_counter() - started) >= 0.04

        self.assertTrue(asyncio.run(scenario()))


class DecisionDataclassTests(unittest.TestCase):
    def test_will_retry_and_is_overflow(self) -> None:
        self.assertTrue(RetryDecision("retry", 1, 0.0, "").will_retry)
        self.assertTrue(RetryDecision("overflow", 1, 0.0, "").will_retry)
        self.assertFalse(RetryDecision("give-up", 1, 0.0, "").will_retry)
        self.assertTrue(RetryDecision("overflow", 1, 0.0, "").is_overflow)


if __name__ == "__main__":
    unittest.main()
