import asyncio
import json
import io
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from mini_harness.interaction import UserQuestions
from mini_harness.interrupt import InterruptService
from mini_harness.cli import _install_question_responder
from mini_harness.console_input import read_line


class QuestionLifecycleTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.ctx = SimpleNamespace(emit=AsyncMock())
        self.questions = UserQuestions(self.ctx, timeout=.05)
        self.questions.browser = True
        self.token = InterruptService()
        self.call = SimpleNamespace(session=None, cancellation=self.token)

    async def test_answer_wins_when_cancel_is_also_ready(self):
        async def emit(kind, session, message):
            if message['kind'] == 'question':
                self.questions.answer(message['id'], 'received')
                self.token.request('stop')
                await asyncio.sleep(0)
        self.ctx.emit.side_effect = emit
        result = await self.questions.ask({'question':'test'}, self.call)
        self.assertFalse(result.is_error)
        self.assertEqual(json.loads(result.content)['answer'], 'received')
        self.assertEqual(self.questions.pending, {})

    async def test_async_responder_timeout_cleans_up_reader(self):
        self.questions.browser = False
        closed = asyncio.Event()
        async def responder(message):
            try:
                await asyncio.Event().wait()
            finally:
                closed.set()
        self.questions.responder = responder
        result = await asyncio.wait_for(self.questions.ask({'question':'test'}, self.call), 1)
        self.assertEqual(json.loads(result.content)['reason'], '等待回答超时')
        self.assertTrue(closed.is_set())
        self.assertEqual(self.questions.pending, {})

    async def test_cli_input_is_cancelled_with_question(self):
        self.questions.browser = False
        self.questions.timeout = 10
        ctx = SimpleNamespace(get=lambda *args: self.questions, interrupt=self.token)
        entered, closed = asyncio.Event(), asyncio.Event()
        async def read(prompt):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                closed.set()
        with patch('mini_harness.cli._stdin_is_tty', return_value=True), patch('mini_harness.cli._write'), patch('mini_harness.cli.read_line', side_effect=read):
            _install_question_responder(ctx)
            task = asyncio.create_task(self.questions.ask({'question':'test'}, self.call))
            await asyncio.wait_for(entered.wait(), 1)
            self.token.request('stop')
            result = await asyncio.wait_for(task, 1)
        self.assertEqual(json.loads(result.content)['reason'], '用户已停止')
        self.assertTrue(closed.is_set())

    async def test_first_accepted_answer_wins_over_responder_return(self):
        self.questions.browser = False
        def responder(message):
            self.assertTrue(self.questions.answer(message['id'], 'first'))
            return 'late'
        self.questions.responder = responder
        result = await self.questions.ask({'question':'test'}, self.call)
        self.assertEqual(json.loads(result.content)['answer'], 'first')

    async def test_responder_failure_is_propagated_and_cleans_pending(self):
        self.questions.browser = False
        for responder_type in (Mock, AsyncMock):
            with self.subTest(responder=responder_type.__name__):
                failure = ValueError('input failed')
                self.questions.responder = responder_type(side_effect=failure)
                self.call.session = SimpleNamespace(append=Mock())
                with self.assertRaisesRegex(ValueError, 'input failed') as raised:
                    await self.questions.ask({'question':'test'}, self.call)
                self.assertIs(raised.exception, failure)
                self.assertEqual(self.questions.pending, {})
                recorded = self.call.session.append.call_args
                self.assertEqual(recorded.args, ('interaction/answer',))
                closed = recorded.kwargs
                self.assertEqual(closed['kind'], 'question-closed')
                self.assertEqual(closed['reason'], '应答界面出错：ValueError')
                self.assertTrue(closed['skipped'])
                self.assertEqual(closed['answer'], '')
                self.assertEqual(self.ctx.emit.await_args.args[2], closed)

    def test_timeout_must_be_positive_finite(self):
        for value in (0, -1, float('inf'), float('nan'), '60'):
            with self.assertRaises(ValueError):
                UserQuestions(self.ctx, timeout=value)


class ConsoleInputTests(unittest.IsolatedAsyncioTestCase):
    async def test_windows_input_editing_and_extended_keys(self):
        chars = iter(['你', 'a', '\b', '\xe0', 'K', '好', '\r'])
        keyboard = SimpleNamespace(kbhit=lambda: True, getwch=lambda: next(chars))
        with patch('mini_harness.console_input.os.name', 'nt'), patch.dict('sys.modules', {'msvcrt':keyboard}), patch('sys.stdout', io.StringIO()):
            self.assertEqual(await read_line('prompt'), '你好')

    async def test_windows_wait_cancels_without_background_reader(self):
        keyboard = SimpleNamespace(kbhit=lambda: False, getwch=Mock())
        with patch('mini_harness.console_input.os.name', 'nt'), patch.dict('sys.modules', {'msvcrt':keyboard}), patch('sys.stdout', io.StringIO()):
            task = asyncio.create_task(read_line('prompt'))
            await asyncio.sleep(.01)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            keyboard.getwch.assert_not_called()

    async def test_posix_wait_cancellation_flushes_partial_input(self):
        terminal = SimpleNamespace(tcflush=Mock(), TCIFLUSH=0)
        stdin = SimpleNamespace(fileno=lambda: 42, readline=Mock())
        with patch('mini_harness.console_input.os.name', 'posix'), patch.dict('sys.modules', {'termios':terminal}), patch('select.select', return_value=([], [], [])), patch('sys.stdin', stdin), patch('sys.stdout', io.StringIO()):
            task = asyncio.create_task(read_line('prompt'))
            await asyncio.sleep(.01)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        terminal.tcflush.assert_called_once_with(42, 0)
        stdin.readline.assert_not_called()
