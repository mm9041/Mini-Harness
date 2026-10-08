import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from mini_harness.cli import main
from mini_harness import cli
from mini_harness.llm import LLMError


class StartupFailureTests(unittest.TestCase):
    def test_os_error_uses_startup_failure_channel_in_every_mode(self):
        message = '无法读取厂商配置，已保留原文件，请检查文件权限或磁盘状态'
        for arguments in (['--mock', 'hello'], ['--repl'], ['--web'], ['--list-tools']):
            with self.subTest(arguments=arguments):
                stderr = io.StringIO()
                with patch('mini_harness.cli._build_config'), patch('mini_harness.cli.build_context', side_effect=OSError(message)) as build, redirect_stderr(stderr):
                    result = main(arguments)
                self.assertEqual(result, 2)
                self.assertEqual(stderr.getvalue(), f'启动失败: {message}\n')
                build.assert_called_once()


class PersistenceFailureTests(unittest.TestCase):
    def setUp(self):
        self.session = Mock(id='test', source_path=None)
        self.session.save.side_effect = OSError('disk full')
        self.args = SimpleNamespace(save_session='test.jsonl', dump_events=False, quiet=True, task='test')
        self.config = SimpleNamespace(streaming=False)

    def test_repl_final_save_cannot_mask_an_inflight_exception(self):
        primary = ValueError('primary failure')
        repl = Mock(session=self.session)
        repl.start.side_effect = primary
        stderr = io.StringIO()
        with patch('mini_harness.cli.ReplSession', return_value=repl), patch('mini_harness.cli._print_banner'), patch('mini_harness.cli._print_mode_line'), redirect_stderr(stderr):
            with self.assertRaises(ValueError) as raised:
                cli._run_repl(None, self.config, self.args, self.session)
        self.assertIs(raised.exception, primary)
        self.assertIn('会话保存失败: OSError: disk full', stderr.getvalue())

    def test_repl_success_with_failed_final_save_returns_nonzero(self):
        repl = Mock(session=self.session)
        repl.start.return_value = 0
        with patch('mini_harness.cli.ReplSession', return_value=repl), patch('mini_harness.cli._print_banner'), patch('mini_harness.cli._print_mode_line'), redirect_stderr(io.StringIO()):
            self.assertEqual(cli._run_repl(None, self.config, self.args, self.session), 2)

    def test_model_failure_keeps_exit_code_when_save_also_fails(self):
        ctx = Mock()
        ctx.agents.create.return_value.run = AsyncMock(side_effect=LLMError('model failed'))
        stderr = io.StringIO()
        with patch('mini_harness.cli.TracePrinter'), patch('mini_harness.cli.status_for', return_value=''), redirect_stdout(io.StringIO()), redirect_stderr(stderr):
            self.assertEqual(cli._run_once(ctx, self.config, self.args, self.session), 3)
        self.assertIn('model failed', stderr.getvalue())
        self.assertIn('disk full', stderr.getvalue())
        self.assertNotIn('Traceback', stderr.getvalue())

    def test_successful_task_with_failed_save_returns_two(self):
        self.session.events = []
        self.config.offline = False
        ctx = Mock()
        ctx.agents.create.return_value.run = AsyncMock(return_value=SimpleNamespace(
            text='completed answer', steps=1, stopped='final', duration_ms=1))
        printer = Mock(show_stream=False)
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch('mini_harness.cli.TracePrinter', return_value=printer), patch('mini_harness.cli.status_for', return_value=''), redirect_stdout(stdout), redirect_stderr(stderr):
            self.assertEqual(cli._run_once(ctx, self.config, self.args, self.session), 2)
        self.assertIn('completed answer', stdout.getvalue())
        self.assertIn('会话保存失败', stderr.getvalue())
        self.assertNotIn('[已保存]', stdout.getvalue())
