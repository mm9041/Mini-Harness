import asyncio
from email.message import Message
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, Mock, patch

from mini_harness.webui import ConversationUi, _make_handler


class RequestTrustTests(TestCase):
    def trusted(self, bind, host, origin=None, site=None, destination='192.168.1.5', duplicate=False):
        handler = _make_handler(SimpleNamespace(host=bind, port=8770))
        request = object.__new__(handler)
        request.headers = Message()
        request.headers['Host'] = host
        if duplicate:
            request.headers['Host'] = host
        if origin is not None:
            request.headers['Origin'] = origin
        if site is not None:
            request.headers['Sec-Fetch-Site'] = site
        request.connection = SimpleNamespace(getsockname=lambda: (destination, 8770))
        return request._trusted_request()

    def test_wildcard_accepts_only_actual_destination_and_loopback(self):
        for host in ('192.168.1.5:8770', '127.0.0.1:8770', 'localhost:8770'):
            self.assertTrue(self.trusted('0.0.0.0', host))
        for host in ('0.0.0.0:8770', 'evil.com:8770', '192.168.1.6:8770', '192.168.1.5:9999'):
            self.assertFalse(self.trusted('0.0.0.0', host))

    def test_explicit_binding_and_origin_checks(self):
        self.assertTrue(self.trusted('127.0.0.1', '127.0.0.1:8770'))
        self.assertFalse(self.trusted('127.0.0.1', '192.168.1.5:8770'))
        self.assertTrue(self.trusted('192.168.1.5', '192.168.1.5:8770'))
        for origin in ('http://evil.com', 'https://127.0.0.1:8770', 'null'):
            self.assertFalse(self.trusted('127.0.0.1', '127.0.0.1:8770', origin))
        self.assertTrue(self.trusted('127.0.0.1', '127.0.0.1:8770', 'http://127.0.0.1:8770'))
        self.assertFalse(self.trusted('127.0.0.1', '127.0.0.1:8770', site='cross-site'))
        self.assertFalse(self.trusted('127.0.0.1', '127.0.0.1:8770', duplicate=True))


class LoopBridgeTests(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.ui = object.__new__(ConversationUi)
        self.ui.loop = asyncio.get_running_loop()
        self.ui._busy = False
        self.ui.providers = SimpleNamespace(set_model=Mock())
        self.ui.ctx = SimpleNamespace(llm=SimpleNamespace(use_model=AsyncMock(return_value='old'),
                                                        list_models=AsyncMock(return_value=['test'])),
                                      userQuestions=SimpleNamespace(answer=Mock(return_value=True)))

    async def test_same_loop_sync_calls_fail_immediately_and_async_works(self):
        for action in (self.ui.list_models, lambda: self.ui.switch_model('new')):
            with self.assertRaisesRegex(RuntimeError, 'async'):
                action()
        self.assertEqual(await self.ui.list_models_async(), ['test'])
        self.assertEqual(await self.ui.switch_model_async('new'), 'old')
        self.assertTrue(self.ui.resolve_question('id', 'yes'))
        self.ui.providers.set_model.assert_called_once_with('new')

    async def test_request_thread_bridge(self):
        self.assertEqual(await asyncio.to_thread(self.ui.list_models), ['test'])
        self.assertEqual(await asyncio.to_thread(self.ui.switch_model, 'new'), 'old')
        self.assertTrue(await asyncio.to_thread(self.ui.resolve_question, 'id', 'yes'))

    async def test_snapshot_builds_schemas_once(self):
        from .support import build_context_with, adapter_plugin_for, make_temp_dir, remove_tree
        from mini_harness.adapters.mock import ScriptedAdapter
        root = make_temp_dir()
        ctx = build_context_with(adapter_plugin_for(ScriptedAdapter([])), root)
        try:
            ui = ctx.webui
            ui.attach()
            original = ctx.tools.schemas
            with patch.object(ctx.tools, 'schemas', wraps=original) as schemas:
                ui.snapshot()
                self.assertEqual(schemas.call_count, 1)
        finally:
            ctx.dispose()
            await remove_tree(root)


class BridgeTimeoutTests(TestCase):
    def test_timeout_cancels_future_on_modern_and_legacy_python(self):
        # Python 3.10 used a distinct concurrent.futures.TimeoutError class.
        class LegacyTimeoutError(Exception):
            pass

        ui = object.__new__(ConversationUi)
        ui.loop = Mock()
        ui.loop.is_running.return_value = True
        for error_type in (TimeoutError, LegacyTimeoutError):
            with self.subTest(error=error_type.__name__):
                error = error_type('timed out')
                future = Mock()
                future.result.side_effect = error
                factory = Mock(return_value=object())
                with patch('mini_harness.webui.concurrent.futures.TimeoutError', LegacyTimeoutError), \
                     patch('mini_harness.webui.asyncio.run_coroutine_threadsafe', return_value=future):
                    with self.assertRaises(error_type) as caught:
                        ui._call_async(factory, timeout=0.01)
                self.assertIs(caught.exception, error)
                future.cancel.assert_called_once_with()
                future.result.assert_called_once_with(timeout=0.01)
