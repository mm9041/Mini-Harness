import asyncio
import json
import os
import shutil
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mini_harness.adapters.mock import ScriptedAdapter
from mini_harness.llm import ToolCall
from mini_harness.system_prompt import SystemPromptService
from .support import adapter_plugin_for, build_context_with, make_temp_dir, remove_tree


class DynamicPromptTests(unittest.TestCase):
    def test_equal_priority_uses_registration_order_not_name(self):
        prompt = SystemPromptService('persona')
        prompt.add_section('zzz', 'registered first', order=50)
        prompt.add_section('aaa', 'registered second', order=50)
        prompt.add_section('last-registered', 'lower order', order=10)
        self.assertEqual(prompt.render(), 'persona\n\nlower order\n\nregistered first\n\nregistered second')

    def test_old_disposer_cannot_remove_replacement(self):
        prompt = SystemPromptService()
        old = prompt.add_section('runtime', 'first', order=50)
        prompt.add_section('peer', 'peer', order=50)
        state = {'text':'second'}
        new = prompt.add_section('runtime', lambda: state['text'], order=50)
        old()
        self.assertEqual(prompt.render(), 'peer\n\nsecond')
        state['text'] = 'updated'
        self.assertEqual(prompt.sections['runtime'], 'updated')
        new()
        old()
        new()
        self.assertEqual(prompt.render(), 'peer')

    def test_persona_update_is_not_removed_by_an_old_section_disposer(self):
        prompt = SystemPromptService()
        old = prompt.add_section('persona', 'old')
        self.assertIsNone(prompt.set_persona('current'))
        old()
        self.assertEqual(prompt.render(), 'current')
        prompt.set_persona('')
        self.assertEqual(prompt.render(), '')

    def test_runtime_section_uses_context_order_constant(self):
        from mini_harness import system_prompt
        from mini_harness.app import HarnessConfig, runtime_context_plugin
        from mini_harness.kernel import Context, mount
        ctx = Context()
        try:
            with patch.object(system_prompt, 'CONTEXT_ORDER', 120):
                mount(ctx, [system_prompt.plugin('persona'), runtime_context_plugin(HarnessConfig())])
            self.assertTrue(ctx.systemPrompt.render().endswith(ctx.systemPrompt.sections['runtime']))
        finally:
            ctx.dispose()

    def test_dynamic_section_reads_live_values_and_can_be_removed(self):
        prompt = SystemPromptService('persona')
        state = {'cwd': 'first'}
        remove = prompt.add_section('runtime', lambda: state['cwd'])
        self.assertEqual(prompt.render(), 'persona\n\nfirst')
        state['cwd'] = 'second'
        self.assertEqual(prompt.render(), 'persona\n\nsecond')
        self.assertEqual(prompt.sections['runtime'], 'second')
        remove()
        self.assertEqual(prompt.render(), 'persona')


class WorkspacePromptTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.root = make_temp_dir()
        self.launch = self.root / 'launch-project'
        self.workspace = self.root / 'selected-workspace'
        self.launch.mkdir()
        self.workspace.mkdir()
        self.adapter = ScriptedAdapter([{'text': 'ok'}])
        self.ctx = build_context_with(adapter_plugin_for(self.adapter), self.launch,
                                      session_root=self.root/'sessions', approval='allow', compaction=False)
        self.ui = self.ctx.webui
        # This intentionally differs from the config captured when plugins were mounted.
        self.ui.config = SimpleNamespace(task_cwd=self.launch, model='test', approval='allow')
        self.ui.attach()
        self.ui.bind_loop(asyncio.get_running_loop())

    async def asyncTearDown(self):
        await asyncio.sleep(0)
        self.ctx.dispose()
        await remove_tree(self.root)

    def assert_current_prompt(self):
        prompt = self.ctx.systemPrompt.render()
        self.assertIn(str(self.workspace.resolve()), prompt)
        self.assertNotIn(str(self.launch), prompt)
        self.assertNotIn(str(self.launch), str([s.to_wire() for s in self.ctx.tools.schemas()]))

    async def test_workspace_switch_is_visible_before_event_callback_runs(self):
        self.ui.set_cwd(str(self.workspace))
        # No await: the old queued workspace/changed callback cannot run yet.
        self.assert_current_prompt()
        await self.ui._run_turn('where am I?')
        self.assertIn(str(self.workspace), self.adapter.requests[-1].system)
        self.assertNotIn(str(self.launch), self.adapter.requests[-1].system)

    async def test_direct_driver_change_overrides_stale_config(self):
        self.ctx.agentLoop.cwd = self.workspace
        self.assertEqual(self.ui.config.task_cwd, self.launch)
        self.assert_current_prompt()

    async def test_real_shell_execution_matches_prompt_and_file_tools(self):
        self.ui.set_cwd(str(self.workspace))
        (self.workspace/'probe.txt').write_text('selected workspace', encoding='utf-8')
        session = self.ui.session
        result = await self.ctx.tools.execute(ToolCall('cwd','pwsh',{'command':'python -c "import os; print(os.getcwd())"'}), session=session, cwd=self.ctx.agentLoop.cwd)
        self.assertFalse(result.is_error, result.content)
        actual = Path(json.loads(result.content)['output'].strip()).resolve()
        self.assertEqual(os.path.normcase(str(actual)), os.path.normcase(str(self.workspace)))
        result = await self.ctx.tools.execute(ToolCall('read','read',{'file_path':'probe.txt'}), session=session, cwd=self.ctx.agentLoop.cwd)
        self.assertIn('selected workspace', result.content)
        self.assert_current_prompt()

    @unittest.skipUnless(shutil.which('pwsh') or shutil.which('powershell'), 'PowerShell not installed')
    async def test_pwsh_default_and_per_call_cwd_do_not_change_workspace(self):
        self.ui.set_cwd(str(self.workspace))
        child = self.workspace/'one-command-only'
        child.mkdir()
        for arguments, expected in [({'command':'(Get-Location).ProviderPath'},self.workspace),
                                    ({'command':'(Get-Location).ProviderPath','cwd':str(child)},child)]:
            result = await self.ctx.tools.execute(ToolCall('pwsh','pwsh',arguments), session=self.ui.session, cwd=self.ctx.agentLoop.cwd)
            self.assertFalse(result.is_error, result.content)
            self.assertEqual(Path(json.loads(result.content)['output'].strip()).resolve(), expected.resolve())
            self.assertEqual(self.ctx.agentLoop.cwd, self.workspace)
            self.assert_current_prompt()

    async def test_default_test_context_storage_is_isolated(self):
        ctx = build_context_with(adapter_plugin_for(ScriptedAdapter([])), self.workspace)
        try:
            for path in (ctx.sessions.root, ctx.webui.providers.path, ctx.jobs.root):
                self.assertTrue(Path(path).resolve().is_relative_to(self.workspace.resolve()))
        finally:
            ctx.dispose()
