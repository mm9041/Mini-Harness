import unittest

from mini_harness.adapters.mock import ScriptedAdapter
from mini_harness.token_meter import TokenMeter, cache_usage
from .support import adapter_plugin_for, build_context_with, make_temp_dir, remove_tree
from .test_compaction import long_session


class CacheUsageTests(unittest.TestCase):
    def test_provider_fields_and_unknown_values(self):
        self.assertEqual(cache_usage({'prompt_tokens':1000, 'prompt_tokens_details':{'cached_tokens':600}}), (600,60))
        self.assertEqual(cache_usage({'prompt_tokens':1000, 'prompt_cache_hit_tokens':0}), (0,0))
        self.assertEqual(cache_usage({'prompt_cache_hit_tokens':600}), (600,None))
        for usage in (None, {}, {'prompt_tokens':1000}, {'prompt_tokens':10,'prompt_cache_hit_tokens':11},
                      {'prompt_cache_hit_tokens':True}, {'prompt_cache_hit_tokens':-1}):
            with self.subTest(usage=usage):
                self.assertEqual(cache_usage(usage), (None,None))

    def test_cache_fields_do_not_reduce_context_usage_or_leak_to_next_request(self):
        meter = TokenMeter('test')
        meter.note_request(1000)
        meter.note_response({'prompt_tokens':1000,'prompt_cache_hit_tokens':600})
        self.assertEqual(meter.snapshot().used, 1000)
        self.assertEqual(meter.snapshot().cache_hit_percent, 60)
        meter.note_request(1100)
        self.assertIsNone(meter.snapshot().cached_tokens)
        meter.note_response({'prompt_tokens':1100,'prompt_cache_hit_tokens':700})
        meter.set_model('different')
        self.assertIsNone(meter.snapshot().cached_tokens)


class CacheAuditTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.root = make_temp_dir()
        self.adapter = ScriptedAdapter([{'text':'摘要','usage':{'prompt_tokens':1000,'completion_tokens':25,'prompt_cache_hit_tokens':600}}])
        self.ctx = build_context_with(adapter_plugin_for(self.adapter), self.root, session_root=self.root/'sessions', max_history_tokens=10**9)
        self.session = self.ctx.sessions.create()
        self.ctx.webui.attach(self.session)

    async def asyncTearDown(self):
        self.ctx.dispose()
        await remove_tree(self.root)

    async def test_ui_reads_cache_from_persisted_reply(self):
        model = getattr(self.ctx.llm.active, 'model', '')
        self.session.append('assistant/message', text='reply', model=model,
                            usage={'prompt_tokens':1000,'prompt_tokens_details':{'cached_tokens':600}})
        path = self.ctx.sessions.save(self.session)
        self.ctx.webui.attach(self.ctx.sessions.open(path))
        context = self.ctx.webui.status()['context']
        self.assertEqual(context['last_cached_tokens'], 600)
        self.assertEqual(context['last_cache_hit_percent'], 60)

    async def test_summary_usage_is_separate_and_persisted_even_when_rejected(self):
        long_session(self.session)
        self.ctx.tokenMeter.note_request(700)
        self.ctx.tokenMeter.note_response({'prompt_tokens':700,'prompt_cache_hit_tokens':100})
        await self.ctx.compaction.condense_now(self.session)
        self.assertEqual(self.ctx.tokenMeter.snapshot().measured, 700)
        self.assertEqual(self.ctx.tokenMeter.snapshot().cached_tokens, 100)
        audit = self.session.events_of('compaction/usage')[-1]
        self.assertEqual(audit.data['usage']['completion_tokens'], 25)
        summary = self.ctx.webui.status()['context']['last_compaction']
        self.assertEqual(summary['status'], 'completed')
        self.assertEqual(summary['cache_hit_percent'], 60)
        # A paid response can fail validation; usage must still survive.
        from unittest.mock import patch
        from mini_harness.llm import GenerateResult
        long_session(self.session, turns=3)
        with patch.object(self.adapter, 'generate', return_value=GenerateResult(text='partial', finish_reason='length', usage={'prompt_tokens':200,'completion_tokens':5})):
            with self.assertRaises(RuntimeError):
                await self.ctx.compaction.condense_now(self.session)
        self.assertEqual(self.session.events_of('compaction/end')[-1].data['status'], 'error')
        self.assertEqual(self.session.events_of('compaction/usage')[-1].data['usage']['completion_tokens'], 5)
        self.assertEqual(len(self.session.compactions()), 1)

    async def test_start_without_end_is_visible_as_incomplete(self):
        self.session.append('compaction/start', operation_id='interrupted', force=True)
        path = self.ctx.sessions.save(self.session)
        self.ctx.webui.attach(self.ctx.sessions.open(path))
        summary = self.ctx.webui.status()['context']['last_compaction']
        self.assertEqual(summary['status'], 'incomplete')
        self.assertIsNone(summary['input_tokens'])

    async def test_new_summary_never_reuses_previous_summary_metrics(self):
        self.session.append('compaction/start', operation_id='old')
        self.session.append('compaction/usage', operation_id='old', usage={
            'prompt_tokens':100, 'prompt_cache_hit_tokens':80})
        self.session.append('compaction/end', operation_id='old', status='completed')
        model = getattr(self.ctx.llm.active, 'model', '')
        self.session.append('assistant/message', text='reply', model=model,
                            usage={'prompt_tokens':200, 'prompt_cache_hit_tokens':150})
        self.session.append('compaction/start', operation_id='new')
        status = self.ctx.webui.status()['context']
        self.assertEqual(status['last_cached_tokens'], 150)
        self.assertEqual(status['last_compaction']['status'], 'incomplete')
        self.assertIsNone(status['last_compaction']['cached_tokens'])
        self.session.append('compaction/usage', operation_id='new', usage={
            'prompt_tokens':50, 'prompt_cache_hit_tokens':10})
        self.session.append('compaction/end', operation_id='new', status='error')
        status = self.ctx.webui.status()['context']
        self.assertEqual(status['last_cached_tokens'], 150)
        self.assertEqual(status['last_compaction']['status'], 'error')
        self.assertEqual(status['last_compaction']['cached_tokens'], 10)
