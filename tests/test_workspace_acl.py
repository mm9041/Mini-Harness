import ctypes
import json
import os
from pathlib import Path
import shutil
import subprocess
import unittest
from unittest.mock import patch

from mini_harness.adapters.mock import scripted_plugin
from mini_harness.llm import ToolCall
from mini_harness.windows_acl import _WindowsAcl, READ_CONTROL, WRITE_DAC, WRITE_OWNER, prepare_workspace_acl
from .support import build_context_with, make_temp_dir, remove_tree


@unittest.skipUnless(os.name == 'nt' and shutil.which('node'), 'Windows DSH runner required')
class WorkspaceAclTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.root = make_temp_dir().resolve()
        self.ctx = build_context_with(scripted_plugin([]), self.root,
                                      permission_preset='workspace-write')
        self.session = self.ctx.sessions.create()
        self.backups = self.root / 'backups'
        redirect = patch('mini_harness.windows_acl.prepare_workspace_acl',
                         side_effect=lambda workspace, _: prepare_workspace_acl(workspace, self.backups))
        redirect.start()
        self.addCleanup(redirect.stop)

    async def asyncTearDown(self):
        self.ctx.dispose()
        await remove_tree(self.root)

    def workspace(self, name):
        path = self.root / name
        path.mkdir()
        # Match a caller-owned drive folder: Modify comes only from a group
        # that DSH drops; ownership supplies WRITE_DAC but not WRITE_OWNER.
        subprocess.run(['icacls', str(path), '/inheritance:r', '/grant',
                        '*S-1-5-11:(OI)(CI)M', '*S-1-5-32-545:(OI)(CI)RX', '/Q'],
                       capture_output=True, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
        self.assertFalse(_WindowsAcl().can_access(path, WRITE_OWNER))
        return path

    async def call(self, workspace, **args):
        return await self.ctx.tools.execute(ToolCall('acl-test', 'pwsh', args), self.session, workspace)

    async def test_repair_multiple_workspaces_and_preserve_confinement(self):
        outside = self.root / 'external.txt'
        outside.write_text('keep')
        first = self.workspace('工作区 one')
        second = self.workspace("workspace two's")
        for workspace in (first, second):
            existing = workspace / 'existing'
            existing.mkdir()
            (existing / 'before.txt').write_text('before')
            result = await self.call(workspace, command="Set-Content -LiteralPath 'existing/before.txt' ok; Write-Output '你好'", timeout=20)
            self.assertFalse(result.is_error, result.content)
            self.assertEqual((existing / 'before.txt').read_text().strip(), 'ok')
            self.assertEqual(json.loads(result.content)['sandbox_mode'], 'workspace-write')
            records = self.session.events_of('permission/workspace-acl-repaired')
            backup = Path(records[-1].data['backup_path'])
            self.addCleanup(backup.unlink, missing_ok=True)
            saved = json.loads(backup.read_text(encoding='utf-8'))
            self.assertEqual(saved['workspace'], str(workspace))
            self.assertIn('sddl_before', saved)
            self.assertEqual(len(saved['added_rights']), 2)
            # Checks are repeated, but an already repaired directory is not rewritten.
            self.assertIsNone(prepare_workspace_acl(workspace, self.backups))
        self.assertEqual(len(self.session.events_of('permission/workspace-acl-repaired')), 2)
        quoted = str(outside).replace("'", "''")
        result = await self.call(second, command=f"Set-Content -LiteralPath '{quoted}' escaped", timeout=20)
        self.assertTrue(result.is_error, result.content)
        self.assertEqual(outside.read_text(), 'keep')
        quoted = str(first / 'existing/before.txt').replace("'", "''")
        result = await self.call(second, command=f"Set-Content -LiteralPath '{quoted}' escaped", timeout=20)
        self.assertTrue(result.is_error, result.content)
        self.assertEqual((first / 'existing/before.txt').read_text().strip(), 'ok')
        # The added user rights do not let the restricted process edit an ACL.
        result = await self.call(second, command="icacls 'existing/before.txt' /grant '*S-1-1-0:F'", timeout=20)
        self.assertTrue(result.is_error, result.content)
        self.ctx.permissions.set('read-only', self.session)
        result = await self.call(second, command="Set-Content 'existing/before.txt' changed", timeout=20)
        self.assertTrue(result.is_error, result.content)
        self.assertEqual((second / 'existing/before.txt').read_text().strip(), 'ok')
        self.assertEqual(self.ctx.approval.history, [])

    async def test_read_only_and_full_access_do_not_repair(self):
        workspace = self.workspace('untouched')
        with patch('mini_harness.windows_acl.prepare_workspace_acl', side_effect=AssertionError('must not repair')):
            for mode in ('read-only', 'danger-full-access'):
                self.ctx.permissions.set(mode, self.session)
                result = await self.call(workspace, command="Write-Output 'ready'", timeout=20)
                self.assertFalse(result.is_error, result.content)
        self.assertFalse(_WindowsAcl().can_access(workspace, WRITE_OWNER))
        self.assertEqual(self.session.events_of('permission/workspace-acl-repaired'), [])

    async def test_backup_failure_prevents_changes_and_command_execution(self):
        workspace = self.workspace('backup-failure')
        def partial_dump(record, stream, **kwargs):
            stream.write('{"partial":')
            stream.flush()
            raise OSError('backup unavailable')
        with patch('mini_harness.windows_acl.json.dump', side_effect=partial_dump):
            result = await self.call(workspace, command="Set-Content 'must-not-run.txt' bad", timeout=20)
        self.assertTrue(result.is_error)
        self.assertIn('SANDBOX_UNAVAILABLE', result.content)
        self.assertIn(str(workspace), result.content)
        self.assertIn('backup unavailable', result.content)
        self.assertFalse((workspace / 'must-not-run.txt').exists())
        self.assertFalse(_WindowsAcl().can_access(workspace, WRITE_OWNER))
        self.assertEqual(self.ctx.jobs.list(self.session.id), [])
        self.assertEqual(list(self.backups.iterdir()), [])

    async def test_foreign_owner_does_not_trigger_ownership_or_acl_changes(self):
        workspace = self.workspace('foreign-owner')
        api = _WindowsAcl()
        # Simulate a different owner at the SID comparison seam; never take
        # ownership of a real foreign/system directory as a test fixture.
        with patch.object(api, 'equal', return_value=False), patch.object(api, 'set_security') as set_acl:
            with self.assertRaisesRegex(RuntimeError, '不属于当前用户'):
                api.prepare(workspace, self.backups)
            set_acl.assert_not_called()
        self.assertFalse(self.backups.exists())

    async def test_backup_cannot_be_written_inside_workspace(self):
        workspace = self.workspace('unsafe-backup')
        with self.assertRaisesRegex(RuntimeError, '备份目录必须位于工作区之外'):
            prepare_workspace_acl(workspace, workspace / 'backups')
        self.assertFalse(_WindowsAcl().can_access(workspace, WRITE_OWNER))
        self.assertFalse((workspace / 'backups').exists())

    async def test_explicit_deny_is_not_overridden(self):
        workspace = self.workspace('explicit-deny')
        api = _WindowsAcl()
        buffer, sid = api.current_user()
        user = api.string(api.to_sid, sid)
        # OWNER RIGHTS disables the otherwise implicit owner WRITE_DAC grant.
        subprocess.run(['icacls', str(workspace), '/deny', f'*{user}:(WDAC)', '*S-1-3-4:(WDAC)', '/Q'],
                       capture_output=True, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            result = await self.call(workspace, command="Set-Content no.txt bad", timeout=20)
            self.assertTrue(result.is_error)
            self.assertIn('WRITE_DAC', result.content)
            self.assertFalse((workspace / 'no.txt').exists())
            self.assertFalse(self.backups.exists())
        finally:
            # No actual file deletion needs WRITE_DAC; fixture cleanup remains possible.
            self.assertFalse(api.can_access(workspace, WRITE_OWNER))


@unittest.skipUnless(os.name == 'nt', 'Windows ACL APIs required')
class NullDaclTests(unittest.IsolatedAsyncioTestCase):
    async def test_null_dacl_is_rejected_without_tightening_permissions(self):
        root = make_temp_dir().resolve()
        workspace = root / 'null-dacl'
        workspace.mkdir()
        api = _WindowsAcl()
        handle = api.handle(workspace, READ_CONTROL | WRITE_DAC)
        original = ctypes.c_void_p()
        original_acl = ctypes.c_void_p()
        changed = False
        try:
            api.check_code(api.get_security(handle, 1, 4, None, None,
                                           ctypes.byref(original_acl), None, ctypes.byref(original)))
            api.check_code(api.set_security(handle, 1, 4, None, None, None, None))
            changed = True
            # The real NULL DACL must survive preparation, not merely a mock branch.
            for prepare in (False, True):
                if prepare:
                    with patch.object(api, 'set_security', wraps=api.set_security) as set_acl:
                        with self.assertRaisesRegex(RuntimeError, 'NULL DACL'):
                            api.prepare(workspace, root / 'backups')
                        set_acl.assert_not_called()
                descriptor, acl = ctypes.c_void_p(), ctypes.c_void_p()
                try:
                    api.check_code(api.get_security(handle, 1, 4, None, None,
                                                   ctypes.byref(acl), None, ctypes.byref(descriptor)))
                    self.assertIsNone(acl.value)
                finally:
                    if descriptor:
                        api.free(descriptor)
            self.assertFalse((root / 'backups').exists())
        finally:
            try:
                if changed:
                    api.check_code(api.set_security(handle, 1, 4, None, None, original_acl, None))
            finally:
                if original:
                    api.free(original)
                api.close(handle)
                await remove_tree(root)


if __name__ == '__main__':
    unittest.main()
