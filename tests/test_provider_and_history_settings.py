import asyncio
import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from mini_harness.adapters.mock import ScriptedAdapter
from mini_harness.adapters.openai_compat import OpenAICompatAdapter
from mini_harness.llm import GenerateRequest, GenerateResult, Message, ToolCall
from mini_harness.history import ConversationHistory
from mini_harness.providers import PRESETS, ProviderStore, reasoning_options, seal, unseal
from mini_harness.token_meter import TokenMeter, window_for
from .support import adapter_plugin_for, build_context_with, make_temp_dir, remove_tree


class ReasoningSettingsTests(unittest.TestCase):
    def test_deepseek_all_requested_levels(self):
        for effort, expected in [('none', None), ('low', 'low'), ('medium', 'high'), ('high', 'high'), ('max', 'max')]:
            with self.subTest(effort=effort):
                payload, note = reasoning_options('deepseek', 'deepseek-v4-flash', effort)
                self.assertEqual(payload['thinking']['type'], 'disabled' if expected is None else 'enabled')
                self.assertEqual(payload.get('reasoning_effort'), expected)
                self.assertTrue(note)

    def test_qwen_budgets_are_ordered(self):
        budgets = [reasoning_options('qwen', 'qwen-plus', e)[0]['thinking_budget'] for e in ('low', 'medium', 'high', 'max')]
        self.assertEqual(budgets, sorted(set(budgets)))
        self.assertEqual(reasoning_options('qwen', 'qwen-plus', 'none')[0], {'enable_thinking': False})

    def test_qwen_max_preview_none_uses_required_thinking_in_wire_payload(self):
        for model in ('qwen3.7-max-preview', 'vendor/qwen3.7-max-preview'):
            adapter = OpenAICompatAdapter('test-key', model=model)
            adapter.reasoning_effort = 'none'
            payload = adapter._build_payload(GenerateRequest(system='test'), stream=True)
            self.assertTrue(payload['enable_thinking'])
            self.assertEqual(payload['thinking_budget'], 1024)
            self.assertIn('none→low', reasoning_options('auto', model, 'none')[1])

    def test_mandatory_thinking_models_do_not_receive_invalid_disabled(self):
        for protocol, model in [('zhipu', 'glm-5.3'), ('kimi', 'kimi-k3')]:
            options, note = reasoning_options(protocol, model, 'none')
            self.assertEqual(options['reasoning_effort'], 'low')
            self.assertNotIn('disabled', str(options))
            self.assertIn('none', note)

    def test_custom_none_does_not_break_legacy_endpoints(self):
        self.assertEqual(reasoning_options('openai', 'custom', 'none')[0], {})
        with self.assertRaises(ValueError):
            reasoning_options('openai', 'custom', 'invalid')

    def test_real_adapter_serializes_selected_reasoning(self):
        adapter = OpenAICompatAdapter('test-key', model='deepseek-v4-flash')
        adapter.reasoning_effort = 'medium'
        payload = adapter._build_payload(GenerateRequest(system='test'), stream=True)
        self.assertEqual(payload['reasoning_effort'], 'high')
        self.assertEqual(payload['thinking'], {'type': 'enabled'})
        self.assertNotIn('test-key', json.dumps(payload))

    def test_window_tiers_and_provider_hard_limit(self):
        self.assertEqual(window_for('deepseek-v4-flash')[0], 1_000_000)
        self.assertEqual(window_for('unknown')[0], 256_000)
        meter = TokenMeter('custom')
        meter.note_capacities({'custom': 2_000_000})
        self.assertEqual(meter.window, 2_000_000)
        self.assertFalse(meter.window_is_guess)
        meter.note_capacities({'custom': 128_000})
        self.assertEqual(meter.window, 128_000)

    def test_key_roundtrip_and_windows_encryption(self):
        key = 'test-secret-not-a-real-key'
        sealed = seal(key)
        self.assertEqual(unseal(sealed), key)
        if os.name == 'nt':
            self.assertNotIn(key, json.dumps(sealed))

    def test_thinking_tool_protocol_preserves_reasoning_across_turns(self):
        adapter = OpenAICompatAdapter('test-key', model='deepseek-v4-flash')
        adapter.reasoning_effort = 'high'
        request = GenerateRequest(system='', messages=[
            Message('assistant', tool_calls=[ToolCall('old', 'read')], reasoning='older reasoning'),
            Message('tool', content='old result', tool_call_id='old'),
            Message('user', content='next task'),
            Message('assistant', tool_calls=[ToolCall('new', 'read')], reasoning='current reasoning'),
            Message('tool', content='new result', tool_call_id='new'),
        ])
        wire = adapter._build_payload(request, stream=True)['messages']
        self.assertEqual(wire[0]['reasoning_content'], 'older reasoning')
        self.assertEqual(wire[3]['reasoning_content'], 'current reasoning')
        adapter.reasoning_effort = 'none'
        self.assertNotIn('reasoning_content', adapter._build_payload(request, stream=True)['messages'][0])
        self.assertNotIn('reasoning_content', adapter._build_payload(request, stream=True)['messages'][3])


class ProviderAndHistoryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.cwd = make_temp_dir()
        self.ctx = build_context_with(
            adapter_plugin_for(ScriptedAdapter([GenerateResult(text='done')])), self.cwd,
            session_root=self.cwd / 'sessions', compaction=False,
        )
        self.ui = self.ctx.webui
        self.ui.attach()
        self.ui.bind_loop(asyncio.get_running_loop())
        self.ui.autosave = True

    async def asyncTearDown(self):
        await asyncio.sleep(0)
        await remove_tree(self.cwd)

    async def test_delete_active_does_not_resurrect_and_can_restore_after_restart(self):
        await self.ui._run_turn('delete me')
        path = self.ui.session.source_path
        original = path.read_bytes()
        result = self.ui.delete_history(self.ui.history.list()[0]['id'])
        self.assertFalse(path.exists())
        self.assertEqual(self.ui.history.list(), [])
        self.ui._save_session()
        self.assertFalse(path.exists())
        self.ui.history = ConversationHistory(self.cwd / 'sessions')
        self.assertEqual(self.ui.history.list(), [])
        self.ui.restore_history(result['undo_token'])
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(self.ui.history.list()[0]['title'], 'delete me')

    async def test_restore_refuses_to_overwrite_existing_file(self):
        await self.ui._run_turn('archive')
        path = self.ui.session.source_path
        result = self.ui.delete_history(self.ui.history.list()[0]['id'])
        path.write_text('new unrelated data', encoding='utf-8')
        with self.assertRaises(ValueError):
            self.ui.restore_history(result['undo_token'])
        self.assertEqual(path.read_text(encoding='utf-8'), 'new unrelated data')

    async def test_delete_rejected_while_busy(self):
        await self.ui._run_turn('saved')
        self.ui._busy = True
        with self.assertRaises(RuntimeError):
            self.ui.delete_history(self.ui.history.list()[0]['id'])

    async def test_group_metadata_uses_actual_workspace(self):
        await self.ui._run_turn('workspace one')
        self.ui.new_session()
        second = self.cwd / 'second'
        second.mkdir()
        self.ui.set_cwd(str(second))
        await self.ui._run_turn('workspace two')
        entries = self.ui.history.list()
        self.assertEqual({e['cwd'] for e in entries}, {str(self.cwd), str(second)})
        self.assertEqual(len({e['workspace_key'] for e in entries}), 2)

    async def test_message_model_is_persisted(self):
        self.ctx.llm.active.model = 'test-model-v1'
        await self.ui._run_turn('name')
        self.ctx.llm.active.model = 'test-model-v2'
        event = self.ui.session.events_of('assistant/message')[0]
        self.assertEqual(event.data['model'], 'test-model-v1')

    async def test_provider_profile_persists_without_exposing_key(self):
        payload = dict(id='custom', name='Local provider', base_url='http://127.0.0.1:19100/v1', model='test-model', protocol='openai', api_key='private-test-key')
        result = self.ui.configure_provider(payload)
        self.assertNotIn('private-test-key', json.dumps(result))
        self.assertEqual(self.ctx.llm.active.model, 'test-model')
        store = ProviderStore(self.cwd / 'providers.json')
        profile, key = store.active()
        self.assertEqual(key, 'private-test-key')
        self.assertEqual(profile['base_url'], payload['base_url'])
        if os.name == 'nt':
            self.assertNotIn(key, store.path.read_text(encoding='utf-8'))

    async def test_blank_key_reuses_only_same_endpoint(self):
        store = self.ui.providers
        profile, _ = store.prepare(dict(PRESETS[0], api_key='test-key'))
        store.save(profile)
        reused, key = store.prepare(dict(PRESETS[0], api_key=''))
        self.assertEqual(key, 'test-key')
        with self.assertRaises(ValueError):
            store.prepare(dict(PRESETS[0], base_url='https://other.example/v1', api_key=''))

    async def test_invalid_provider_leaves_current_adapter(self):
        original = self.ctx.llm.active
        with self.assertRaises(ValueError):
            self.ui.configure_provider(dict(name='bad', model='m', base_url='file:///secret', api_key='key'))
        self.assertIs(self.ctx.llm.active, original)

    async def test_reasoning_change_persists_and_busy_change_is_rejected(self):
        self.ui.switch_reasoning('max')
        self.assertEqual(self.ctx.llm.active.reasoning_effort, 'max')
        self.assertEqual(ProviderStore(self.cwd / 'providers.json').data['reasoning'], 'max')
        self.ui._busy = True
        with self.assertRaises(RuntimeError):
            self.ui.switch_reasoning('low')

    async def test_catalog_capacities_reach_token_meter(self):
        self.ctx.llm.active.model_context_windows = {'remote': 1_048_576}
        await self.ctx.llm.list_models()
        self.ctx.tokenMeter.set_model('remote')
        self.assertEqual(self.ctx.tokenMeter.window, 1_048_576)
        self.assertFalse(self.ctx.tokenMeter.window_is_guess)
