import asyncio
import unittest
from pathlib import Path

from mini_harness.adapters.mock import stream_from_result
from mini_harness.llm import GenerateResult, StreamEvent, ToolCall
from .support import adapter_plugin_for, build_context_with, make_temp_dir, remove_tree
from .test_webui import post_json


class Gates:
    def __init__(self):
        self.started = {key: asyncio.Event() for key in ('A', 'B')}
        self.release = {key: asyncio.Event() for key in ('A', 'B')}


class GatedAdapter:
    name = 'gated-test'
    model = 'test-model'
    reasoning_effort = 'none'

    def __init__(self, gates, escalation=False):
        self.gates, self.escalation = gates, escalation
        self.requests = []

    def clone_for_conversation(self):
        # Only test synchronization is shared; model settings and requests are independent.
        return GatedAdapter(self.gates, self.escalation)

    async def stream(self, request):
        self.requests.append(request)
        label = next(m.content for m in reversed(request.messages) if m.role == 'user')
        if request.messages[-1].role == 'tool':
            result = GenerateResult(text='finished-' + label, finish_reason='stop')
        else:
            yield StreamEvent('delta', reasoning='reason-' + label)
            yield StreamEvent('delta', text='working-' + label)
            self.gates.started[label].set()
            await self.gates.release[label].wait()
            args = {'file_path':'result.txt', 'content':label}
            if self.escalation:
                args.update(sandbox_permissions='workspace-write', justification='write test result')
            call = ToolCall('same-id-in-each-conversation', 'write', args)
            yield StreamEvent('tool_call', tool_call=call)
            result = GenerateResult(text='working-' + label, reasoning='reason-' + label,
                                    tool_calls=[call], finish_reason='tool_calls')
            yield StreamEvent('done', result=result)
            return
        async for frame in stream_from_result(result):
            yield frame

    async def list_models(self):
        return [self.model]


class ConcurrentConversationTests(unittest.IsolatedAsyncioTestCase):
    async def test_storage_maintenance_cleans_spill_without_another_write(self):
        import os
        import time
        from unittest.mock import patch
        store = self.ui.ctx.spillStore
        record = store.save('expired', session_id='stale-maintenance')
        directory = Path(record.locator).parent
        old = time.time() - 8 * 86400
        os.utime(directory, (old, old))
        later = time.monotonic() + 3601
        with patch('mini_harness.spill.time.monotonic', return_value=later):
            self.ui.cleanup_expired_storage()
        self.assertFalse(directory.exists())

    async def asyncSetUp(self):
        self.root = make_temp_dir().resolve()
        self.a_dir, self.b_dir = self.root/'a', self.root/'b'
        self.a_dir.mkdir(); self.b_dir.mkdir()
        self.gates = Gates()
        self.adapter = GatedAdapter(self.gates)
        self.ctx = build_context_with(adapter_plugin_for(self.adapter), self.a_dir, permission_preset='workspace-write',
                                      session_root=self.root/'sessions', compaction=False)
        self.ui = self.ctx.webui
        self.ui.attach(); self.ui.autosave = True
        self.ui.bind_loop(asyncio.get_running_loop())
        self.ui.port = 0
        _, port = self.ui.start()
        self.base = f'http://127.0.0.1:{port}'

    async def asyncTearDown(self):
        await self.ui.shutdown()
        self.ui.stop()
        self.ctx.dispose()
        await remove_tree(self.root)

    async def start_both(self):
        a = self.ui.target()
        a.send('A')
        await asyncio.wait_for(self.gates.started['A'].wait(), 3)
        self.ui.new_session(str(self.b_dir))
        b = self.ui.target()
        b.send('B')
        await asyncio.wait_for(self.gates.started['B'].wait(), 3)
        self.assertTrue(a._busy and b._busy)
        return a, b

    async def finish(self, worker):
        await asyncio.wait_for(asyncio.wrap_future(worker._turn_future), 5)

    async def wait_for(self, predicate):
        for _ in range(200):
            if predicate():
                return
            await asyncio.sleep(.01)
        self.fail('condition did not become true')

    async def test_two_turns_overlap_and_keep_workspaces_streams_and_history_separate(self):
        a, b = await self.start_both()
        for key in ('interrupt','agentLoop','tokenMeter','llm','tools','jobs','approval','permissions'):
            self.assertIsNot(a.ctx.get(key), b.ctx.get(key), key)
        self.assertIsNot(a.ctx.llm.active, b.ctx.llm.active)
        b.ctx.llm.active.model = 'b-only-model'
        self.assertEqual(a.ctx.llm.active.model, 'test-model')
        rows = self.ui.history_items()
        self.assertEqual(sum(row['running'] for row in rows), 2)
        self.ui.open_history(self.ui.history.key(a.session.source_path))
        self.assertEqual(self.ui.snapshot()['stream']['text'], 'working-A')
        self.assertEqual(self.ui.status()['cwd'], str(self.a_dir))
        events = self.ui.subscribe()
        self.gates.release['B'].set()
        await self.finish(b)
        pushed = []
        while not events.empty():
            pushed.append(events.get_nowait())
        self.assertFalse(any(e.get('kind') == 'assistant' and e.get('text') == 'finished-B' for e in pushed))
        self.assertTrue(self.ui.status()['busy'])
        self.assertEqual((self.b_dir/'result.txt').read_text(), 'B')
        self.assertFalse((self.a_dir/'result.txt').exists())
        self.gates.release['A'].set(); await self.finish(a)
        self.assertEqual((self.a_dir/'result.txt').read_text(), 'A')
        self.ui.open_history(self.ui.history.key(b.session.source_path))
        self.assertFalse(self.ui.snapshot()['busy'])
        self.assertIs(self.ui.target(), b)
        self.assertIn('finished-B', [e.data.get('text') for e in b.session.events_of('assistant/message')])
        self.assertNotIn('finished-A', [e.data.get('text') for e in b.session.events_of('assistant/message')])

    async def test_delayed_http_stop_targets_originating_conversation_not_selected_one(self):
        a, b = await self.start_both()
        code, _ = await asyncio.to_thread(post_json, self.base, '/api/message', {'session':a.session.id,'text':'duplicate'})
        self.assertEqual(code, 409)
        code, result = await asyncio.to_thread(post_json, self.base, '/api/interrupt', {'session':a.session.id})
        self.assertEqual(code, 200); self.assertTrue(result['ok'])
        await self.finish(a)
        self.assertTrue(b._busy)
        self.assertFalse(b.ctx.interrupt.cancelled)
        self.assertFalse((self.a_dir/'result.txt').exists())
        with self.assertRaisesRegex(RuntimeError, '正在运行'):
            self.ui.delete_history(self.ui.history.key(b.session.source_path))
        # An idle history can be deleted even while the selected task is running.
        self.ui.delete_history(self.ui.history.key(a.session.source_path))
        self.assertTrue(b._busy)
        self.gates.release['B'].set(); await self.finish(b)
        self.assertEqual((self.b_dir/'result.txt').read_text(), 'B')

    async def test_approvals_and_policy_changes_do_not_cross_conversations(self):
        self.adapter.escalation = True
        self.ui.switch_access('read-only')
        a = self.ui.target(); a.send('A')
        await asyncio.wait_for(self.gates.started['A'].wait(),3)
        self.ui.new_session(str(self.b_dir))
        self.ui.switch_access('read-only')
        b = self.ui.target(); b.send('B')
        await asyncio.wait_for(self.gates.started['B'].wait(),3)
        self.gates.release['A'].set(); self.gates.release['B'].set()
        await self.wait_for(lambda: a._pending and b._pending)
        self.assertEqual(sum(row['waiting'] for row in self.ui.history_items()),2)
        a_id, b_id = next(iter(a._pending)), next(iter(b._pending))
        self.assertNotEqual(a_id,b_id)
        self.assertFalse(a.resolve_approval(b_id,True))
        a.ctx.permissions.set('danger-full-access',a.session)
        self.assertEqual(b.ctx.permissions.current(b.session),'read-only')
        self.assertTrue(b.resolve_approval(b_id,True))
        await self.finish(a); await self.finish(b)
        self.assertFalse((self.a_dir/'result.txt').exists())
        self.assertEqual((self.b_dir/'result.txt').read_text(),'B')

    async def test_shutdown_does_not_cancel_unrelated_loop_tasks(self):
        a,b = await self.start_both()
        unrelated=asyncio.create_task(asyncio.sleep(30))
        try:
            await self.ui.shutdown()
            self.assertFalse(unrelated.cancelled())
            self.assertFalse(a._busy or b._busy)
            self.assertFalse((self.a_dir/'result.txt').exists())
            self.assertFalse((self.b_dir/'result.txt').exists())
        finally:
            unrelated.cancel()
            await asyncio.gather(unrelated,return_exceptions=True)

    async def test_archive_http_restore_and_permanent_delete(self):
        import json
        from urllib.request import urlopen
        path = self.root / 'sessions' / 'archivable.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'type':'user/message','data':{'text':'archive API'}}) + '\n', encoding='utf-8')
        key = self.ui.history.key(path)
        code, result = await asyncio.to_thread(post_json, self.base, '/api/history/delete', {'id':key})
        self.assertEqual(code, 200)
        token = result['undo_token']
        def listing():
            with urlopen(self.base + '/api/history/archives') as response:
                return json.load(response)
        data = await asyncio.to_thread(listing)
        self.assertEqual(data['items'][0]['token'], token)
        self.assertEqual(data['retention_days'], 7)
        self.ui.history.retention_seconds = 3600
        custom = await asyncio.to_thread(listing)
        self.assertAlmostEqual(custom['retention_days'], 1 / 24)
        self.assertEqual(custom['items'][0]['expires_at'], custom['items'][0]['archived_at'] + 3600)
        code, result = await asyncio.to_thread(post_json, self.base, '/api/history/restore', {'token':token})
        self.assertEqual(code, 200)
        self.assertTrue(path.is_file())
        self.assertEqual((await asyncio.to_thread(listing))['items'], [])
        _, result = await asyncio.to_thread(post_json, self.base, '/api/history/delete', {'id':key})
        code, _ = await asyncio.to_thread(post_json, self.base, '/api/history/purge', {'token':result['undo_token']})
        self.assertEqual(code, 200)
        self.assertEqual((await asyncio.to_thread(listing))['items'], [])
        self.assertFalse(path.exists())

    async def test_worker_initialization_failures_dispose_unregistered_context(self):
        from unittest.mock import patch
        for stage in ('copy', 'constructor', 'attach', 'adopt'):
            with self.subTest(stage=stage):
                fresh = self.ui._fresh_context(self.b_dir)
                disposed = []
                fresh.effect(lambda: disposed.append(True))
                before = dict(self.ui._workers)
                owned = list(self.ui._owned_contexts)
                target = {'copy': 'mini_harness.webui_sessions.copy.copy',
                          'constructor': 'mini_harness.webui.ConversationUi',
                          'attach': 'mini_harness.webui.ConversationUi.attach'}
                failure = (patch.object(self.ui, '_adopt', side_effect=RuntimeError('injected'))
                           if stage == 'adopt' else patch(target[stage], side_effect=RuntimeError('injected')))
                with patch.object(self.ui, '_fresh_context', return_value=fresh), failure:
                    with self.assertRaisesRegex(RuntimeError, 'injected'):
                        self.ui._create_worker(self.b_dir)
                self.assertEqual(disposed, [True])
                self.assertEqual(self.ui._workers, before)
                self.assertEqual(self.ui._owned_contexts, owned)

    async def test_failed_archive_rolls_back_replacement_worker(self):
        from unittest.mock import patch
        worker = self.ui.target()
        worker.session.append('user/message', text='keep me')
        worker._save_session()
        path = worker.session.source_path
        before = dict(self.ui._workers)
        owned = list(self.ui._owned_contexts)
        disposed = []
        original = self.ui._fresh_context
        def fresh(cwd):
            ctx = original(cwd)
            ctx.effect(lambda: disposed.append(True))
            return ctx
        with patch.object(self.ui, '_fresh_context', side_effect=fresh), \
             patch.object(self.ui.history, 'delete', side_effect=OSError('archive failed')):
            with self.assertRaisesRegex(OSError, 'archive failed'):
                self.ui.delete_history(self.ui.history.key(path))
        self.assertIs(self.ui.target(), worker)
        self.assertEqual(self.ui._workers, before)
        self.assertEqual(self.ui._owned_contexts, owned)
        self.assertEqual(disposed, [True])
        self.assertTrue(path.exists())

    async def test_stop_then_shutdown_drains_owned_context_cleanup(self):
        a, b = await self.start_both()
        disposed = []
        b.ctx.effect(lambda: disposed.append(True))
        self.ui.stop()
        self.assertTrue(self.ui._cleanup_futures)
        await asyncio.wait_for(self.ui.shutdown(), 5)
        self.assertEqual(disposed, [True])
        self.assertEqual(self.ui._cleanup_futures, [])
        self.assertEqual(self.ui._owned_contexts, [])
        self.ui.stop()
        self.assertEqual(disposed, [True])

    def test_uninitialized_facade_raises_attribute_error(self):
        from mini_harness.webui_sessions import WebUi
        ui = WebUi.__new__(WebUi)
        with self.assertRaises(AttributeError):
            _ = ui.missing


if __name__ == '__main__':
    unittest.main()
