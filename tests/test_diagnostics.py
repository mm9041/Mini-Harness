from contextlib import redirect_stderr
import io
import logging
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mini_harness.session import Session
from mini_harness.spill import LocalSpillStore


class NonfatalDiagnosticTests(unittest.TestCase):
    def isolated_logger(self, name, level=logging.WARNING):
        logger = logging.getLogger(name)
        state = logger.handlers[:], logger.propagate, logger.level
        def restore():
            logger.handlers, logger.propagate, logger.level = state
            logger.setLevel(state[2])
        self.addCleanup(restore)
        logger.handlers = []
        logger.propagate = False
        logger.setLevel(level)
        return logger

    def broken(self, event):
        raise RuntimeError('display failed')

    def test_observer_last_resort_is_concise_without_handlers(self):
        self.isolated_logger('mini_harness.session')
        session = Session('test')
        session.observe(self.broken)
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            session.append('user/message', text='test')
        self.assertIn('RuntimeError: display failed', stderr.getvalue())
        self.assertNotIn('Traceback', stderr.getvalue())
        self.assertEqual(len(session.events), 1)

    def test_cleanup_last_resort_is_concise_without_handlers(self):
        self.isolated_logger('mini_harness.spill')
        with tempfile.TemporaryDirectory() as root:
            store = LocalSpillStore(Path(root))
            stderr = io.StringIO()
            with patch.object(store, 'cleanup', side_effect=OSError('disk unavailable')), redirect_stderr(stderr):
                store.cleanup_if_due()
            self.assertIn('OSError: disk unavailable', stderr.getvalue())
            self.assertNotIn('Traceback', stderr.getvalue())

    def test_debug_retains_traceback_for_diagnosis(self):
        self.isolated_logger('mini_harness.session', logging.DEBUG)
        session = Session('test')
        session.observe(self.broken)
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            session.append('user/message', text='test')
        self.assertIn('Traceback', stderr.getvalue())
