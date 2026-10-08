import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from mini_harness.jobs import Job, JobsService, _timeout


class JobsLifecycleTests(unittest.TestCase):
    def test_completed_records_are_bounded_without_losing_running_jobs_or_output(self):
        with tempfile.TemporaryDirectory() as folder:
            service = JobsService(Path(folder), max_completed=2)
            output = Path(folder) / 'output.log'
            output.write_text('saved output', encoding='utf-8')
            running = Job('running', 'owner', 'command', Path(folder), Mock(pid=1), output)
            service.jobs[running.id] = running
            finished = []
            for i in range(4):
                process = Mock(pid=i + 2, returncode=0)
                process.poll.return_value = 0
                job = Job(str(i), 'owner', 'command', Path(folder), process, output, group=Mock(), started=0)
                service.jobs[job.id] = job
                with patch('mini_harness.jobs.time.time', return_value=i + 1):
                    service._watch(job, 10)
                finished.append(job)
            self.assertEqual(set(service.jobs), {'running', '2', '3'})
            self.assertEqual(service.output(finished[0])['output'], 'saved output')
            with self.assertRaises(ValueError):
                service.get('0', 'owner')
            self.assertIs(service.get('running', 'owner'), running)
            with patch('mini_harness.jobs.time.time', return_value=300000):
                self.assertEqual(finished[-1].snapshot()['duration_ms'], 4000)

    def test_null_timeout_defaults_and_invalid_values_are_actionable(self):
        self.assertEqual(_timeout(None, 60), 60)
        self.assertEqual(_timeout(None, 30, waiting=True), 30)
        self.assertEqual(_timeout(0, 30, waiting=True), 0)
        for value in (True, [], {}, 'bad', float('nan'), float('inf'), -1):
            for waiting in (False, True):
                with self.subTest(value=value, waiting=waiting):
                    with self.assertRaisesRegex(ValueError, 'timeout 必须'):
                        _timeout(value, 30, waiting=waiting)
        with self.assertRaises(ValueError):
            _timeout(0, 60)
