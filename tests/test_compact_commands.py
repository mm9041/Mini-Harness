import asyncio
import unittest
from unittest.mock import patch
from mini_harness.adapters.mock import ScriptedAdapter
from mini_harness.webui import ui_message
from mini_harness.llm import GenerateResult
from tests.support import adapter_plugin_for, build_context_with, make_temp_dir, remove_tree

class CompactCommandTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.root = make_temp_dir()
        self.adapter = ScriptedAdapter([{'text':'short summary'}])
        self.ctx = build_context_with(adapter_plugin_for(self.adapter), self.root,
            session_root=self.root/'sessions', max_history_tokens=10**9, keep_recent_messages=2)
        self.ui = self.ctx.webui
        self.ui.attach()
        self.ui.bind_loop(asyncio.get_running_loop())

    async def asyncTearDown(self):
        self.ui.stop()
        self.ctx.dispose()
        await remove_tree(self.root)

    def seed(self):
        for i in range(10):
            self.ui.session.append('user/message' if i%2==0 else 'assistant/message', text='history '*100)

    async def test_web_compact_is_local_command_and_persists_summary(self):
        self.seed()
        count=len(self.ui.session.events_of('user/message'))
        self.ui.send('/compact')
        await asyncio.wrap_future(self.ui._turn_future)
        self.assertEqual(len(self.adapter.requests),1)
        self.assertEqual(self.adapter.requests[0].tools,self.ctx.tools.schemas())
        self.assertEqual(len(self.ui.session.events_of('user/message')),count)
        self.assertEqual(len(self.ui.session.compactions()),1)
        self.assertEqual(ui_message(self.ui.session.events_of('compaction')[0])['kind'],'command-result')
        projected=self.ctx.compaction.project(self.ui.session)
        self.assertIn('short summary',projected[0].content)

    async def test_local_commands_and_short_history_do_not_call_model(self):
        for command in ('/help','/tools','/status','/unknown','/compact'):
            self.ui.send(command)
            await asyncio.wrap_future(self.ui._turn_future)
        self.assertEqual(self.adapter.requests,[])
        self.assertEqual(self.ui.session.events_of('user/message'),[])
        self.assertEqual(len(self.ui.session.events_of('command/result')),5)

    async def test_eighty_percent_triggers_even_below_history_budget(self):
        self.seed()
        self.ctx.tokenMeter.note_capacities({self.ctx.tokenMeter.model:1000})
        with patch.object(self.ctx.agentLoop,'_estimated_tokens',return_value=799):
            self.assertIsNone(await self.ctx.compaction.maybe_condense(self.ui.session))
        with patch.object(self.ctx.agentLoop,'_estimated_tokens',return_value=800) as estimate:
            self.assertIsNotNone(await self.ctx.compaction.maybe_condense(self.ui.session))
            self.assertTrue(estimate.call_args.args[0])
            self.assertTrue(estimate.call_args.args[2])

    async def test_cancelled_compaction_does_not_commit(self):
        self.seed()
        started=asyncio.Event()
        async def summarize(messages):
            started.set()
            await asyncio.Event().wait()
        with patch.object(self.ctx.compaction,'_summarize',side_effect=summarize):
            task=asyncio.create_task(self.ctx.compaction.condense_now(self.ui.session))
            await started.wait()
            self.ctx.interrupt.request('cancel')
            self.assertIsNone(await asyncio.wait_for(task,1))
        self.assertEqual(self.ui.session.compactions(),[])

    async def test_final_reply_can_trigger_automatic_compaction(self):
        self.seed()
        # Leave room for the full system/tool envelope; pressure should start after the reply.
        self.ctx.tokenMeter.note_capacities({self.ctx.tokenMeter.model:20000})
        with patch.object(self.adapter,'generate',side_effect=[
            GenerateResult(text='answer '*4000),
            GenerateResult(text='summary')]):
            self.ctx.agentLoop.streaming=False
            await self.ui.agent.run('continue')
        self.assertTrue(self.ui.session.compactions())

    async def test_manual_busy_is_a_visible_command_result(self):
        self.seed()
        self.ctx.compaction._active.add(id(self.ui.session))
        try:
            await self.ui._run_command('/compact')
        finally:
            self.ctx.compaction._active.clear()
        notice = self.ui.session.events_of('command/result')[-1]
        self.assertEqual(notice.data['code'], 'busy')
        self.assertIn('正在压缩', notice.data['text'])
        self.assertEqual(ui_message(notice)['kind'], 'command-result')
        self.assertEqual(self.adapter.requests, [])

    async def test_manual_progress_explains_effective_retention(self):
        self.seed()
        for keep, expected in ((0, '最新一条'), (2, '至少 2 条')):
            self.ctx.compaction.policy.keep_recent_messages = keep
            with patch.object(self.ui, 'publish') as publish, patch.object(
                    self.ctx.compaction, 'condense_now', return_value=None):
                await self.ui._run_command('/compact')
            progress = [call.args[0]['text'] for call in publish.call_args_list
                        if call.args[0].get('kind') == 'command-progress']
            self.assertTrue(any(expected in text for text in progress))

    async def test_final_compaction_failure_does_not_duplicate_notice(self):
        self.seed()
        self.ctx.tokenMeter.note_capacities({self.ctx.tokenMeter.model:20000})
        with patch.object(self.adapter, 'generate', side_effect=[
                GenerateResult(text='answer '*4000),
                GenerateResult(text='summary '*10000)]):
            self.ctx.agentLoop.streaming = False
            result = await self.ui.agent.run('continue')
        self.assertEqual(result.stopped, 'final')
        notices = self.ui.session.events_of('command/result')
        self.assertEqual(len(notices), 1)
        self.assertIn('摘要未缩小', notices[0].data['text'])
