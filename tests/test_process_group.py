"""Real Windows processes: no target code before Job assignment, safe failures."""
import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from mini_harness.jobs import JobsService
from mini_harness.process_group import CREATE_SUSPENDED, WindowsProcessGroup


@unittest.skipUnless(os.name == 'nt', 'Windows Job Objects required')
class WindowsProcessGroupTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def suspended(self, marker):
        command = 'from pathlib import Path; import time; Path(' + repr(str(marker)) + ").write_text('started'); time.sleep(30)"
        process = subprocess.Popen([sys.executable, '-c', command], stdin=subprocess.DEVNULL,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   creationflags=CREATE_SUSPENDED | subprocess.CREATE_NO_WINDOW)
        def cleanup():
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
        self.addCleanup(cleanup)
        return process

    def wait_for_marker(self, marker):
        until = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < until:
            time.sleep(.01)
        self.assertTrue(marker.exists())

    def test_target_does_not_run_before_assignment_and_close_kills_it(self):
        marker = self.root / 'executed.txt'
        group = WindowsProcessGroup()
        self.addCleanup(group.close)
        process = self.suspended(marker)
        time.sleep(.25)  # Deliberately widen the formerly unsafe Popen -> assign interval.
        self.assertFalse(marker.exists())
        self.assertIsNone(process.poll())
        group.dll.IsProcessInJob.argtypes = [wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)]
        group.dll.IsProcessInJob.restype = wintypes.BOOL
        resume = group._resume_initial_thread
        def checked_resume(child):
            member = wintypes.BOOL()
            self.assertTrue(group.dll.IsProcessInJob(int(child._handle), group.handle, ctypes.byref(member)))
            self.assertTrue(member.value)
            self.assertFalse(marker.exists())
            resume(child)
        with patch.object(group, '_resume_initial_thread', side_effect=checked_resume):
            group.assign_suspended(process)
        self.wait_for_marker(marker)
        group.close()
        self.assertIsNotNone(process.wait(timeout=5))

    def test_nested_job_assignment(self):
        outer, inner = WindowsProcessGroup(), WindowsProcessGroup()
        self.addCleanup(outer.close)
        self.addCleanup(inner.close)
        marker = self.root / 'nested.txt'
        process = self.suspended(marker)
        outer.assign(process)
        inner.assign_suspended(process)
        self.wait_for_marker(marker)
        inner.close()
        self.assertIsNotNone(process.wait(timeout=5))

    @unittest.skipUnless(shutil.which('pwsh') or shutil.which('powershell'), 'PowerShell required')
    def test_failed_assignment_or_resume_leaves_no_live_job(self):
        real_popen = subprocess.Popen
        for stage in ('assign', '_resume_initial_thread'):
            with self.subTest(stage=stage):
                service = JobsService(self.root / stage)
                self.addCleanup(service.close)
                processes = []
                def popen(*args, **kwargs):
                    child = real_popen(*args, **kwargs)
                    processes.append(child)
                    return child
                with patch('mini_harness.jobs.subprocess.Popen', side_effect=popen), patch.object(WindowsProcessGroup, stage, side_effect=OSError('injected ' + stage)):
                    with self.assertRaisesRegex(OSError, 'injected'):
                        service.start("Write-Output 'must not run'", self.root, 'owner', 5)
                self.assertEqual(len(processes), 1)
                self.assertIsNotNone(processes[0].poll())
                self.assertEqual(service.jobs, {})
                self.assertTrue(all(file.read_bytes() == b'' for file in service.root.glob('*/output.log')))
