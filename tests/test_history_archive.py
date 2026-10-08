"""Archive retention, recovery, and bounded permanent deletion."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mini_harness.history import ConversationHistory


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.history = ConversationHistory(self.root)
        self.path = self.root / 'session.jsonl'
        self.content = json.dumps({'type': 'user/message', 'data': {'text': 'Archived title'}, 'ts': 123}) + '\n'
        self.path.write_text(self.content, encoding='utf-8')

    def archive(self, now=1000):
        with patch('mini_harness.history.time.time', return_value=now):
            return self.history.delete(self.history.key(self.path))

    def test_list_and_restore_across_restart(self):
        token = self.archive()
        history = ConversationHistory(self.root)
        with patch('mini_harness.history.time.time', return_value=1001):
            item, = history.archives()
            self.assertEqual(item['token'], token)
            self.assertEqual(item['title'], 'Archived title')
            self.assertEqual(item['expires_at'], 1000 + 604800)
            self.assertEqual(history.list(), [])
            history.restore(token)
        self.assertEqual(self.path.read_text(encoding='utf-8'), self.content)
        self.assertFalse((self.root / '.trash' / token).exists())

    def test_cleanup_exact_boundary_preserves_recent_and_live(self):
        old = self.archive()
        self.path.write_text(self.content, encoding='utf-8')
        recent = self.archive(now=2000)
        self.path.write_text('live', encoding='utf-8')
        self.assertEqual(self.history.cleanup(now=605799), [])
        self.assertEqual(self.history.cleanup(now=605800), [old])
        self.assertTrue((self.root / '.trash' / recent).exists())
        self.assertEqual(self.path.read_text(), 'live')

    def test_legacy_archive_uses_metadata_time_not_conversation_time(self):
        token = self.archive()
        metadata = self.root / '.trash' / token / 'metadata.json'
        metadata.write_text(json.dumps({'original': str(self.path)}), encoding='utf-8')
        os.utime(metadata, (2000, 2000))
        with patch('mini_harness.history.time.time', return_value=2001):
            item, = self.history.archives()
        self.assertEqual(item['title'], 'Archived title')
        self.assertEqual(item['expires_at'], 606800)
        self.assertEqual(self.history.cleanup(now=606800), [token])

    def test_expired_restore_fails(self):
        token = self.archive()
        with patch('mini_harness.history.time.time', return_value=605800):
            with self.assertRaises(ValueError):
                self.history.restore(token)
        self.assertFalse(self.path.exists())

    def test_custom_retention_controls_expiry_and_error_message(self):
        class Short(ConversationHistory):
            retention_seconds = 3600

        self.history = Short(self.root)
        token = self.archive(now=1000)
        with patch('mini_harness.history.time.time', return_value=4599):
            item, = self.history.archives()
            self.assertEqual(item['expires_at'], 4600)
        with patch('mini_harness.history.time.time', return_value=4600):
            with self.assertRaises(ValueError) as raised:
                self.history.restore(token)
        self.assertEqual(str(raised.exception), '这条归档已超过保留期（0.0416667 天）')
        self.assertFalse(self.path.exists())

    def test_purge_is_final_and_rejects_traversal(self):
        token = self.archive()
        self.history.purge(token)
        with self.assertRaises(ValueError):
            self.history.restore(token)
        for token in ('../', '', '..' + 'a' * 30, 'A' * 32):
            with self.assertRaises(ValueError):
                self.history.purge(token)

    def test_restore_does_not_overwrite(self):
        token = self.archive()
        self.path.write_text('replacement', encoding='utf-8')
        with patch('mini_harness.history.time.time', return_value=1001):
            with self.assertRaises(ValueError):
                self.history.restore(token)
        self.assertEqual(self.path.read_text(), 'replacement')
        self.assertTrue((self.root / '.trash' / token / 'conversation.jsonl').is_file())

    def test_malformed_archive_does_not_block_other_cleanup(self):
        token = self.archive()
        broken = self.root / '.trash' / ('b' * 32)
        broken.mkdir()
        # Unreadable JSON must be skipped without blocking valid archives.
        (broken / 'metadata.json').write_text('{broken')
        self.assertEqual(self.history.cleanup(now=605800), [token])
        self.assertTrue(broken.exists())
