import asyncio
import json
import os
import shutil
import unittest
from pathlib import Path

from mini_harness.adapters.mock import scripted_plugin
from mini_harness.app import HarnessConfig
from mini_harness.approval import ApprovalDecision, ApprovalRequest
from mini_harness.llm import ToolCall
from mini_harness.permissions import PRESETS, RANK, grant_private_temp
from mini_harness.tools import Tool
from .support import build_context_with, make_temp_dir, remove_tree


class PresetTests(unittest.IsolatedAsyncioTestCase):
    async def test_parent_policy_changes_are_inherited_until_detached(self):
        child = self.ctx.sessions.create()
        approval = self.ctx.approval
        self.assertIsNone(approval.parent_session(child))
        approval.set_parent_session(child, self.session)
        self.assertIs(approval.parent_session(child), self.session)
        self.ctx.permissions.set('read-only', self.session)
        approval.set_mode('deny', self.session)
        self.assertEqual(self.ctx.permissions.current(child), 'read-only')
        self.assertEqual(approval.mode_for(child), 'deny')
        self.ctx.permissions.set('danger-full-access', self.session)
        self.assertEqual(self.ctx.permissions.current(child), 'danger-full-access')
        approval.set_parent_session(child, None)
        self.assertIsNone(approval.parent_session(child))
        self.assertEqual(self.ctx.permissions.current(child), 'workspace-write')

    def test_permission_ranks_cover_presets_in_security_order(self):
        self.assertEqual(set(RANK), set(PRESETS))
        self.assertLess(RANK['read-only'], RANK['workspace-write'])
        self.assertLess(RANK['workspace-write'], RANK['danger-full-access'])

    async def asyncSetUp(self):
        self.root = make_temp_dir().resolve()
        grant_private_temp(self.root)
        self.workspace = self.root / 'workspace'
        self.workspace.mkdir()
        self.ctx = build_context_with(scripted_plugin([]), self.workspace,
                                      permission_preset='workspace-write')
        self.session = self.ctx.sessions.create()

    async def asyncTearDown(self):
        self.ctx.dispose()
        await remove_tree(self.root)

    async def call(self, name, **args):
        return await self.ctx.tools.execute(ToolCall('test', name, args), self.session, self.workspace)

    async def test_default_and_bundles(self):
        self.assertEqual(HarnessConfig().permission_preset, 'workspace-write')
        self.assertEqual(HarnessConfig.from_env({}, env_file=None).permission_preset, 'workspace-write')
        self.assertEqual(PRESETS['danger-full-access'].approval, 'never')
        result = await self.call('write', file_path='ok.txt', content='yes')
        self.assertFalse(result.is_error)
        self.assertEqual((self.workspace / 'ok.txt').read_text(), 'yes')
        self.assertEqual(self.ctx.approval.history, [])

    async def test_request_exposes_current_policy_without_changing_history(self):
        self.ctx.permissions.set('read-only', self.session)
        before = len(self.session.events)
        request = await self.ctx.agentLoop._build_request(self.session)
        self.assertIn('Current sandbox: read-only', request.system)
        self.assertIn('Approval policy: ask', request.system)
        self.assertEqual(len(self.session.events), before)
        self.ctx.permissions.set('danger-full-access', self.session)
        request = await self.ctx.agentLoop._build_request(self.session)
        self.assertIn('Approval policy: never', request.system)

    async def test_file_modes_and_one_call_escalation(self):
        outside = self.root / 'outside.txt'
        denied = await self.call('write', file_path=str(outside), content='bad')
        self.assertTrue(denied.is_error)
        self.assertFalse(outside.exists())
        self.ctx.approval.set_approver(lambda r: ApprovalDecision(True))
        allowed = await self.call('write', file_path=str(outside), content='allowed',
                                  sandbox_permissions='danger-full-access', justification='test outside write')
        self.assertFalse(allowed.is_error)
        self.assertEqual(outside.read_text(), 'allowed')
        denied = await self.call('write', file_path=str(outside), content='bad again')
        self.assertTrue(denied.is_error)
        self.assertEqual(outside.read_text(), 'allowed')
        self.assertEqual(self.ctx.permissions.current(self.session), 'workspace-write')

    async def test_read_only_and_full_access(self):
        self.ctx.permissions.set('read-only', self.session)
        self.assertTrue((await self.call('write', file_path='no.txt', content='no')).is_error)
        self.ctx.approval.set_approver(lambda r: ApprovalDecision(False))
        self.assertTrue((await self.call('write', file_path='no.txt', content='no',
                         sandbox_permissions='workspace-write', justification='test')).is_error)
        outside = self.root / 'outside.txt'
        self.ctx.permissions.set('danger-full-access', self.session)
        before = len(self.ctx.approval.history)
        self.assertFalse((await self.call('write', file_path=str(outside), content='ok')).is_error)
        self.assertEqual(len(self.ctx.approval.history), before)
        self.ctx.permissions.set('read-only', self.session)
        self.assertFalse((await self.call('read', file_path=str(outside))).is_error)

    async def test_never_means_reject_escalation(self):
        self.ctx.approval.set_approver(lambda r: self.fail('never must not prompt'))
        decision = await self.ctx.approval.decide(ApprovalRequest(ToolCall('x','write'), 'test',
                                                 session=self.session, policy='never'))
        self.assertFalse(decision.approved)

    async def test_unconfined_plugin_does_not_inherit_workspace_write(self):
        target = self.root / 'plugin.txt'
        self.ctx.tools.register(Tool('custom', 'test', handler=lambda a,c: target.write_text('bad'), permission='write'))
        self.assertTrue((await self.call('custom')).is_error)
        self.assertFalse(target.exists())

    async def test_policy_roundtrip_and_legacy_migration(self):
        self.ctx.permissions.set('read-only', self.session)
        path = self.ctx.sessions.save(self.session, self.root / 'session.jsonl')
        restored = self.ctx.sessions.open(path)
        self.assertEqual(self.ctx.permissions.current(restored), 'read-only')
        self.assertEqual(self.ctx.permissions.current(self.ctx.sessions.create()), 'workspace-write')
        legacy = self.ctx.sessions.create()
        legacy.append('approval/policy', mode='allow')
        self.assertEqual(self.ctx.permissions.current(legacy), 'workspace-write')
        legacy.append('approval/policy', mode='deny')
        self.assertEqual(self.ctx.permissions.current(legacy), 'read-only')

    async def test_switch_revokes_pending_escalation(self):
        started = asyncio.Event()
        async def approve(request):
            started.set()
            await asyncio.Event().wait()
        self.ctx.approval.set_approver(approve)
        task = asyncio.create_task(self.call('write', file_path=str(self.root/'no.txt'), content='bad',
                    sandbox_permissions='danger-full-access', justification='test'))
        await started.wait()
        self.ctx.permissions.set('read-only', self.session)
        self.assertTrue((await asyncio.wait_for(task, 1)).is_error)
        self.assertFalse((self.root/'no.txt').exists())

    async def test_missing_runner_fails_closed(self):
        self.ctx.permissions.runtime_root = self.root / 'missing'
        result = await self.call('pwsh', command="Set-Content no.txt bad", timeout=5)
        self.assertTrue(result.is_error)
        self.assertIn('SANDBOX_UNAVAILABLE', result.content)
        self.assertFalse((self.workspace/'no.txt').exists())

    async def test_runtime_inside_writable_workspace_is_rejected(self):
        self.ctx.permissions.runtime_root = self.workspace / 'runtime'
        runner = self.ctx.permissions.runtime_root / 'node_modules/@deepseek-ai/dsh-sandbox-windows-acl/lib/runner.js'
        runner.parent.mkdir(parents=True)
        runner.write_text("throw new Error('must not execute')")
        (self.ctx.permissions.runtime_root / 'sandbox_runner.cjs').write_text("throw new Error('must not execute')")
        result = await self.call('pwsh', command='echo no', timeout=5)
        self.assertTrue(result.is_error)
        self.assertIn('SANDBOX_UNAVAILABLE', result.content)

    async def test_web_command_switches_preset_and_reports_policy(self):
        ui = self.ctx.webui
        ui.attach()
        ui.bind_loop(asyncio.get_running_loop())
        await ui._run_command('/permission read-only')
        state = ui.status()
        self.assertEqual(state['access'], 'read-only')
        self.assertEqual(state['permissions']['approval'], 'ask')
        self.assertTrue(state['permission_scope']['shell_sandbox'])
        self.assertEqual(set(state['access_labels']), set(PRESETS))
        ui.stop()

    @unittest.skipUnless(os.name == 'nt' and shutil.which('node'), 'Windows native runner test')
    async def test_native_powershell_enforces_writes_and_read_only(self):
        outside = self.root / 'external.txt'
        outside.write_text('keep')
        quoted = str(outside).replace("'", "''")
        result = await self.call('pwsh', command=f"Write-Output '你好'; Set-Content inside.txt ok; Set-Content -LiteralPath (Join-Path $env:TEMP 'private.txt') ok; try {{ Set-Content -LiteralPath '{quoted}' bad }} catch {{ Write-Output 'DENIED' }}", timeout=20)
        self.assertFalse(result.is_error, result.content)
        self.assertEqual((self.workspace/'inside.txt').read_text().strip(), 'ok')
        self.assertEqual(outside.read_text(), 'keep')
        self.assertIn('DENIED', json.loads(result.content)['output'])
        self.assertIn('你好', json.loads(result.content)['output'])
        self.assertEqual(json.loads(result.content)['sandbox_mode'], 'workspace-write')
        self.ctx.permissions.set('read-only', self.session)
        result = await self.call('pwsh', command="Set-Content inside.txt changed", timeout=20)
        self.assertTrue(result.is_error)
        self.assertEqual((self.workspace/'inside.txt').read_text().strip(), 'ok')
        result = await self.call('pwsh', command="Write-Output '只读'", timeout=20)
        self.assertFalse(result.is_error, result.content)
        self.assertIn('只读', json.loads(result.content)['output'])

    @unittest.skipUnless(os.name == 'nt' and shutil.which('node'), 'Windows native runner test')
    async def test_native_child_and_alternate_cwd_do_not_escape_workspace(self):
        outside = self.root / 'external.txt'
        # Native cmd grandchild bypasses any hypothetical PowerShell text filtering.
        command = f'cmd.exe /d /c "echo CHILD_STARTED & echo escaped> {outside}"'
        result = await self.call('pwsh', command=command, cwd=str(self.root), timeout=20)
        self.assertIn('CHILD_STARTED', json.loads(result.content)['output'])
        self.assertFalse(outside.exists())


if __name__ == '__main__':
    unittest.main()
