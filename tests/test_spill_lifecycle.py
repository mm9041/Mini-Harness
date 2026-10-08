import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from mini_harness.kernel import Context, MODE_WATERFALL, mount
from mini_harness.llm import ToolCall
from mini_harness import spill, tools
from mini_harness.spill import LocalSpillStore
from mini_harness.tools import Tool, ToolResult


class SpillStorageLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_creation_is_exclusive_and_never_overwrites_existing_snapshot(self):
        store = LocalSpillStore(self.root)
        with patch('mini_harness.spill.time.strftime', return_value='120000'), patch('mini_harness.spill.secrets.token_hex', return_value='same'):
            record = store.save('original', session_id='s')
            with self.assertRaises(FileExistsError):
                store.save('replacement', session_id='s')
        self.assertEqual(Path(record.locator).read_text(), 'original')

    def test_creation_requests_mode_0600_before_any_data_write(self):
        store = LocalSpillStore(self.root)
        real_open = os.open
        seen = []
        def opened(path, flags, mode=0o777, **kwargs):
            seen.append((flags, mode))
            return real_open(path, flags, mode, **kwargs)
        with patch('mini_harness.spill.os.open', side_effect=opened):
            record = store.save('private', session_id='s')
        self.assertEqual(seen[0][1], 0o600)
        self.assertTrue(seen[0][0] & os.O_EXCL)
        if os.name != 'nt':
            self.assertEqual(Path(record.locator).stat().st_mode & 0o777, 0o600)

    def test_quotas_reject_new_output_without_deleting_existing_files(self):
        for settings in ({'max_session_files':1}, {'max_session_bytes':5}):
            with self.subTest(settings=settings):
                store = LocalSpillStore(self.root / str(len(settings)) / next(iter(settings)), **settings)
                record = store.save('first', session_id='s')
                with self.assertRaisesRegex(OSError, '配额已满'):
                    store.save('second', session_id='s')
                self.assertEqual(Path(record.locator).read_text(), 'first')
                self.assertEqual(len(list(Path(record.locator).parent.iterdir())), 1)

    def test_later_save_reclaims_expired_sessions(self):
        with patch('mini_harness.spill.time.monotonic', return_value=0):
            store = LocalSpillStore(self.root)
            old = store.save('expired', session_id='old')
        stale = time.time() - 8 * 86400
        os.utime(Path(old.locator).parent, (stale, stale))
        with patch('mini_harness.spill.time.monotonic', return_value=3601):
            new = store.save('fresh', session_id='new')
        self.assertFalse(Path(old.locator).exists())
        self.assertTrue(Path(new.locator).exists())


class SpillPipelineLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.ctx = Context()
        mount(self.ctx, [tools.plugin(), spill.plugin(root=self.root, max_inline_chars=100, head_chars=20, tail_chars=20)])
        self.original = ToolResult('private-output-' * 1000, True, [{'test':'image'}])
        self.ctx.tools.register(Tool('large', 'test', handler=lambda a,c:self.original))

    async def asyncTearDown(self):
        self.ctx.dispose()
        self.temp.cleanup()

    async def test_spill_and_two_replacement_rewriters_all_compose(self):
        seen = []
        async def first(call, result, context, nxt):
            seen.append(result.content)
            return await nxt(call, ToolResult(result.content + '|first', result.is_error, result.images), context)
        async def second(call, result, context, nxt):
            return await nxt(call, ToolResult(result.content + '|second', result.is_error, result.images), context)
        self.ctx.on('tools/post-execute', first, MODE_WATERFALL)
        self.ctx.on('tools/post-execute', second, MODE_WATERFALL)
        result = await self.ctx.tools.execute(ToolCall('c', 'large'))
        self.assertIn('已保存到', seen[0])
        self.assertTrue(result.content.endswith('|first|second'))
        self.assertLess(len(result.content), len(self.original.content))
        self.assertTrue(result.is_error)
        self.assertEqual(result.images, self.original.images)
        self.assertEqual(next(self.root.glob('*/*.txt')).read_text(), self.original.content)

    async def test_storage_failure_forwards_truncation_instead_of_original(self):
        seen = []
        async def observer(call, result, context, nxt):
            seen.append(result.content)
            return await nxt()
        self.ctx.on('tools/post-execute', observer, MODE_WATERFALL)
        with patch.object(self.ctx.spillStore, 'save', side_effect=OSError('配额已满')):
            result = await self.ctx.tools.execute(ToolCall('c', 'large'))
        self.assertIn('配额已满', result.content)
        self.assertIn('不可恢复', result.content)
        self.assertLess(len(result.content), len(self.original.content))
        self.assertEqual(seen, [result.content])
        self.assertTrue(result.is_error)
        self.assertEqual(result.images, self.original.images)

    async def test_invalid_postprocessor_cannot_restore_unbounded_original(self):
        self.ctx.on('tools/post-execute', lambda *args: None, MODE_WATERFALL)
        result = await self.ctx.tools.execute(ToolCall('c', 'large'))
        self.assertTrue(result.is_error)
        self.assertIn('未返回有效', result.content)

    async def test_zero_preview_still_gives_locator(self):
        policy = spill.SpillPolicy(max_inline_chars=100, head_chars=0, tail_chars=0)
        result, record = policy.apply(self.original, store=self.ctx.spillStore, session_id='zero')
        self.assertIn(record.locator, result.content)
        self.assertIn('read', result.content)
