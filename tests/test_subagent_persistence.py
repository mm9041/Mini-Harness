import asyncio
from pathlib import Path
import unittest
from unittest.mock import patch

from mini_harness.adapters.mock import ScriptedAdapter
from mini_harness.history import ConversationHistory
from mini_harness.kernel import MODE_EMIT
from mini_harness.llm import GenerateResult, LLMError, ToolCall
from mini_harness.session import SessionsService
from mini_harness.subagent import _DEPTH
from .support import adapter_plugin_for, build_context_with, make_temp_dir, remove_tree


class SubagentPersistenceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.root = make_temp_dir()
        self.adapter = ScriptedAdapter([{'text':'child result'}])
        self.ctx = build_context_with(adapter_plugin_for(self.adapter), self.root,
                                      session_root=self.root/'sessions', compaction=False,
                                      permission_preset='workspace-write')
        self.parent = self.ctx.sessions.create()

    async def asyncTearDown(self):
        self.ctx.dispose()
        await remove_tree(self.root)

    def saved_child(self, result):
        path = Path(result.session_path)
        self.assertTrue(path.is_file())
        return SessionsService(self.ctx.sessions.root).open(path)

    async def test_success_survives_restart_but_is_hidden_in_chat_history(self):
        result = await self.ctx.subagents.run('inspect files', parent_session=self.parent, title='Inspect')
        self.assertTrue(result.ok)
        child = self.saved_child(result)
        self.assertEqual(child.events_of('assistant/message')[-1].data['text'], 'child result')
        self.assertEqual(child.events_of('session/parent')[0].data['parent_session'], self.parent.id)
        history = ConversationHistory(self.ctx.sessions.root)
        self.assertEqual(history.list(), [])
        row, = history.list(include_children=True)
        self.assertEqual(history.list(), [])  # Cached entries must stay filtered.
        self.assertEqual(row['title'], 'inspect files')
        self.assertEqual(Path(row['cwd']), self.root)
        self.assertIn(Path(result.session_path), self.ctx.sessions.list_sessions())
        self.assertNotIn(result.session_id, self.ctx.sessions._live)
        self.assertIsNone(self.ctx.agents.get(result.session_id))
        self.assertIn(self.parent.id, self.ctx.sessions._live)
        self.assertTrue(self.parent.events_of('subagent/end')[-1].data['log_saved'])

    async def test_model_error_is_saved_and_still_returns_error(self):
        with patch.object(self.adapter, 'generate', side_effect=LLMError('invalid request', status=400)):
            result = await self.ctx.subagents.run('fail', parent_session=self.parent)
        self.assertFalse(result.ok)
        self.assertEqual(result.stopped, 'error')
        child = self.saved_child(result)
        self.assertEqual(child.events_of('turn/end')[-1].data['stopped'], 'error')
        self.assertEqual(len(child.events_of('turn/start')), len(child.events_of('turn/end')))

    async def test_hard_task_cancellation_preserves_audits_and_propagates(self):
        entered = asyncio.Event()
        async def blocked(request):
            entered.set()
            await asyncio.Event().wait()
        with patch.object(self.adapter, 'generate', side_effect=blocked):
            task = asyncio.create_task(self.ctx.subagents.run('wait', parent_session=self.parent))
            await asyncio.wait_for(entered.wait(), 2)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        result = self.ctx.subagents.runs[-1]
        self.assertEqual(result.stopped, 'cancelled')
        child = self.saved_child(result)
        self.assertEqual(len(child.events_of('step/start')), len(child.events_of('step/end')))
        self.assertEqual(len(child.events_of('turn/start')), len(child.events_of('turn/end')))
        self.assertEqual(len(self.parent.events_of('subagent/start')), len(self.parent.events_of('subagent/end')))
        self.assertIsNone(self.ctx.approval.parent_session(child))
        self.assertNotIn(child.id, self.ctx.sessions._live)
        self.assertEqual(self.ctx.subagents.depth, 0)

    async def test_save_failure_preserves_answer_warns_and_retains_live_session(self):
        with patch.object(self.ctx.sessions, 'save', side_effect=OSError('disk full')):
            result = await self.ctx.tools.execute(ToolCall('task', 'task', {'description':'Test','prompt':'run'}), self.parent, self.root)
        self.assertFalse(result.is_error)
        self.assertIn('child result', result.content)
        self.assertIn('日志保存失败', result.content)
        self.assertIn('disk full', result.content)
        self.assertIn('仅当前进程内存可用', result.content)
        self.assertNotIn('已保存至', result.content)
        run = self.ctx.subagents.runs[-1]
        self.assertEqual(run.stopped, 'final')  # Execution succeeded; persistence failed.
        self.assertTrue(run.ok)
        self.assertEqual(run.failure_reason, '')
        self.assertIn(run.session_id, self.ctx.sessions._live)
        self.assertIsNotNone(self.ctx.agents.get(run.session_id))
        self.assertFalse(self.parent.events_of('subagent/end')[-1].data['log_saved'])
        self.assertIn('disk full', self.parent.events_of('command/result')[-1].data['text'])

    async def test_save_warning_survives_bounded_answer_rendering(self):
        self.ctx.subagents.policy.result_max_chars = 10
        self.adapter.script = [GenerateResult(text='A' * 100)]
        with patch.object(self.ctx.sessions, 'save', side_effect=OSError('disk full')):
            result = await self.ctx.tools.execute(ToolCall('task', 'task', {'description':'Test','prompt':'run'}), self.parent, self.root)
        self.assertFalse(result.is_error)
        self.assertIn('A' * 10, result.content)
        self.assertNotIn('A' * 11, result.content)
        self.assertIn('原始 100 字符', result.content)
        self.assertIn('disk full', result.content)
        self.assertIn('日志未落盘', result.content)

    async def test_execution_failure_is_not_masked_by_save_failure(self):
        with patch.object(self.adapter, 'generate', side_effect=LLMError('model failed', status=400)), \
             patch.object(self.ctx.sessions, 'save', side_effect=OSError('disk full')):
            result = await self.ctx.tools.execute(ToolCall('task', 'task', {'description':'Test','prompt':'run'}), self.parent, self.root)
        self.assertTrue(result.is_error)
        self.assertIn('model failed', result.content)
        self.assertIn('disk full', result.content)

    async def test_refused_is_not_an_orphan_end_or_fictional_session(self):
        token = _DEPTH.set(self.ctx.subagents.policy.max_depth)
        try:
            result = await self.ctx.tools.execute(ToolCall('task', 'task', {'description':'Test','prompt':'nested'}), self.parent, self.root)
        finally:
            _DEPTH.reset(token)
        self.assertTrue(result.is_error)
        self.assertIn('未创建子会话', result.content)
        self.assertEqual(self.parent.events_of('subagent/start'), [])
        self.assertEqual(self.parent.events_of('subagent/end'), [])
        self.assertEqual(len(self.parent.events_of('subagent/refused')), 1)
        self.assertEqual(self.ctx.sessions.list_sessions(), [])

    async def test_start_notification_failure_still_closes_and_saves_child(self):
        def broken(*args):
            raise RuntimeError('trace failed')
        self.ctx.on('subagent/started', broken, MODE_EMIT)
        result = await self.ctx.subagents.run('never started', parent_session=self.parent)
        self.assertEqual(result.stopped, 'error')
        self.assertIn('trace failed', result.note)
        self.saved_child(result)
        self.assertEqual(len(self.parent.events_of('subagent/end')), 1)

    async def test_saved_policy_matches_parent_at_completion(self):
        self.ctx.permissions.set('danger-full-access', self.parent)
        def change_parent(*args):
            self.ctx.permissions.set('read-only', self.parent)
        self.ctx.on('subagent/started', change_parent, MODE_EMIT)
        result = await self.ctx.subagents.run('inspect', parent_session=self.parent)
        child = self.saved_child(result)
        self.assertEqual(self.ctx.permissions.current(child), 'read-only')
