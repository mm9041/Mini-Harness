import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mini_harness.session import Session, SessionsService
from mini_harness.adapters.mock import scripted_plugin
from .support import build_context_with


class SessionIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sessions = SessionsService(self.root)

    @unittest.skipUnless(os.name == 'nt', 'Windows sharing semantics')
    def test_save_recovers_after_concurrent_reader_releases_checkpoint(self):
        session = self.sessions.create()
        session.append('user/message', text='before')
        path = self.sessions.save(session)
        reader = path.open('r', encoding='utf-8')
        self.addCleanup(reader.close)
        session.append('assistant/message', text='after')
        with patch('mini_harness.session.time.sleep', side_effect=lambda _: reader.close()) as pause:
            self.sessions.save(session)
        pause.assert_called_once_with(0.01)
        self.assertIn('after', path.read_text(encoding='utf-8'))
        self.assertEqual(list(self.root.glob('.session-*.tmp')), [])

    @unittest.skipUnless(os.name == 'nt', 'Windows sharing semantics')
    def test_save_persistent_sharing_failure_preserves_previous_checkpoint(self):
        session = self.sessions.create()
        session.append('user/message', text='before')
        path = self.sessions.save(session)
        before = path.read_bytes()
        session.append('assistant/message', text='after')
        with path.open('r', encoding='utf-8'), patch('mini_harness.session.time.sleep') as pause:
            with self.assertRaises(OSError):
                self.sessions.save(session)
        self.assertEqual(pause.call_count, 5)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(list(self.root.glob('.session-*.tmp')), [])

    def damaged(self):
        path = self.root / 'damaged.jsonl'
        lines = [json.dumps({'seq':1,'type':'turn/start'}),
                 json.dumps({'seq':2,'type':'user/message','data':{'text':'hello'}}),
                 '{broken', json.dumps({'seq':4,'type':'assistant/message','data':{'text':'reply'}})]
        original = ('\r\n'.join(lines) + '\r\n').encode()
        path.write_bytes(original)
        return path, original

    def test_bom_preserves_workspace_and_append_uses_next_sequence(self):
        path = self.root / 'bom.jsonl'
        data = [{'seq':1,'type':'session/workspace','data':{'cwd':'/workspace'}},
                {'seq':2,'type':'user/message','data':{'text':'你好'}}]
        path.write_bytes(b'\xef\xbb\xbf' + ('\r\n'.join(json.dumps(e,ensure_ascii=False) for e in data) + '\r\n').encode())
        session = self.sessions.open(path)
        self.assertEqual(session.events[0].data['cwd'], '/workspace')
        self.assertEqual(session.repairs, [])
        self.assertEqual(session.append('assistant/message', text='reply').seq, 3)
        self.sessions.save(session)
        self.assertNotIn(b'\r\n', path.read_bytes())
        self.assertFalse(path.read_bytes().startswith(b'\xef\xbb\xbf'))
        self.assertEqual(self.sessions.open(path).events[0].type, 'session/workspace')

    def test_gaps_are_preserved_and_original_bad_line_is_backed_up_once(self):
        path, original = self.damaged()
        session = self.sessions.open(path)
        self.assertEqual([e.seq for e in session.events], [1,2,4,5])
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(session.append('user/message', text='next').seq, 6)
        self.sessions.save(session)
        self.assertEqual(session.recovery_backup.read_bytes(), original)
        session.append('assistant/message', text='answer')
        self.sessions.save(session)
        self.assertEqual(len(list(self.root.glob('*.bak'))), 1)
        self.assertEqual(session.recovery_backup.read_bytes(), original)
        self.assertEqual([e.seq for e in self.sessions.open(path).events], [1,2,4,5,6,7])

    def test_backup_failure_or_changed_source_prevents_overwrite(self):
        path, original = self.damaged()
        session = self.sessions.open(path)
        with patch('mini_harness.session.tempfile.mkstemp', side_effect=OSError('backup failed')):
            with self.assertRaises(OSError):
                session.save(path)
        self.assertEqual(path.read_bytes(), original)
        changed = original + b'new data\n'
        path.write_bytes(changed)
        with self.assertRaisesRegex(OSError, '已改变'):
            session.save(path)
        self.assertEqual(path.read_bytes(), changed)

    def test_saving_recovered_session_elsewhere_preserves_original(self):
        path, original = self.damaged()
        session = self.sessions.open(path)
        other = self.root / 'export.jsonl'
        self.sessions.save(session, other)
        session.append('user/message', text='new')
        self.sessions.save(session)
        self.assertEqual(path.read_bytes(), original)
        self.assertIn('new', other.read_text())

    def test_duplicate_or_decreasing_sequences_are_not_renumbered(self):
        for sequences in ([1,1], [3,2]):
            path = self.root / 'ambiguous.jsonl'
            raw = '\n'.join(json.dumps({'seq':seq,'type':'user/message','data':{'text':'hi'}}) for seq in sequences).encode()
            path.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, '序号重复或倒退'):
                self.sessions.open(path)
            self.assertEqual(path.read_bytes(), raw)

    def test_latest_skips_newer_empty_files(self):
        valid = self.root / 'older.jsonl'
        session = Session('older')
        session.append('user/message', text='keep')
        session.save(valid)
        os.utime(valid, (100,100))
        empty = self.root / 'newer.jsonl'
        empty.write_bytes(b'')
        os.utime(empty, (200,200))
        self.assertEqual(self.sessions.latest(), valid)
        valid.unlink()
        self.assertIsNone(self.sessions.latest())

    def test_web_autosave_does_not_create_blank_file(self):
        ctx = build_context_with(scripted_plugin([]), self.root, session_root=self.root/'sessions')
        self.addCleanup(ctx.dispose)
        ctx.webui.attach()
        ctx.webui.autosave = True
        ctx.webui._save_session()
        self.assertEqual(list((self.root/'sessions').glob('*.jsonl')), [])
        self.assertIsNone(ctx.webui.session.source_path)

    def test_incomplete_image_batch_does_not_bleed_into_next_batch(self):
        session = Session('images')
        session.append('assistant/message', tool_calls=[{'id':'old1','name':'read_image'}, {'id':'old2','name':'read_image'}])
        session.append('tool/result', call_id='old1', images=[{'data_url':'old'}])
        session.append('assistant/message', tool_calls=[{'id':'new','name':'read_image'}])
        session.append('tool/result', call_id='new', images=[{'data_url':'new'}])
        images = [image for message in session.derive_messages() for image in message.images]
        self.assertEqual(images, [{'data_url':'new'}])

    def test_observer_failure_does_not_revoke_event_or_skip_other_observers(self):
        session = Session('observers')
        def broken(event):
            raise RuntimeError('display failed')
        seen = []
        session.observe(broken)
        session.observe(seen.append)
        with self.assertLogs('mini_harness.session', level='ERROR'):
            event = session.append('user/message', text='committed')
        self.assertEqual(session.events, [event])
        self.assertEqual(seen, [event])
