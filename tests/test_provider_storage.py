import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from mini_harness.providers import PRESETS, ProviderStore, _crypt, reasoning_options, seal, unseal


class ProviderStorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'providers.json'

    def test_bad_ports_and_other_urls_have_chinese_hint(self):
        store = ProviderStore(self.path)
        for url in ('https://example.com:abc', 'https://example.com:65536',
                    'https://example.com:-1', 'https://[broken', 'file:///tmp',
                    'https://user:password@example.com', 'https://example.com/?key=test'):
            with self.subTest(url=url):
                with self.assertRaisesRegex(ValueError, '^Base URL 需要完整 http'):
                    store.prepare({'base_url':url, 'model':'test', 'api_key':'fake-key'})
        self.assertFalse(self.path.exists())

    def test_crypt_failure_distinguishes_encryption_and_decryption(self):
        dll = SimpleNamespace(CryptProtectData=Mock(return_value=0), CryptUnprotectData=Mock(return_value=0))
        with patch('mini_harness.providers.ctypes.WinDLL', return_value=dll, create=True):
            with self.assertRaisesRegex(OSError, '加密保护'):
                _crypt(b'fake')
            with self.assertRaisesRegex(OSError, '解密已保存'):
                _crypt(b'fake', decrypt=True)

    def test_unknown_or_malformed_key_scheme_is_not_plaintext(self):
        for value in ({'scheme':'x', 'value':'fake'}, {'value':'fake'}, None,
                      {'scheme':'local-file', 'value':123}):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, '请重新填写'):
                    unseal(value)
        self.assertEqual(unseal({'scheme':'local-file', 'value':'fake'}), 'fake')
        self.assertEqual(unseal(seal('fake-test-key')), 'fake-test-key')

    @unittest.skipUnless(os.name == 'nt', 'Windows DPAPI required')
    def test_corrupt_dpapi_payload_reports_decryption_failure(self):
        with self.assertRaisesRegex(ValueError, '密文格式无效'):
            unseal({'scheme':'dpapi', 'value':'***'})
        with self.assertRaisesRegex(OSError, '解密已保存'):
            unseal({'scheme':'dpapi', 'value':'bm90LWEtZHBhcGktYmxvYg=='})

    def test_corrupt_bytes_are_preserved_after_reset_and_save(self):
        for raw in (b'{broken', b'\xff\xfe', b'[]', b'{"profiles":{"x":42}}'):
            with self.subTest(raw=raw):
                self.path.write_bytes(raw)
                store = ProviderStore(self.path)
                self.assertEqual(store.data['profiles'], {})
                self.assertIn('已备份', store.public()['warning'])
                backup = Path(store.warning.split('：', 1)[1])
                self.assertEqual(backup.read_bytes(), raw)
                store.set_effort('high')
                self.assertEqual(backup.read_bytes(), raw)
                self.assertEqual(ProviderStore(self.path).data['reasoning'], 'high')

    def test_read_or_backup_failure_preserves_original_and_raises(self):
        self.path.write_bytes(b'{broken')
        with patch.object(Path, 'read_text', side_effect=PermissionError('denied')):
            with self.assertRaisesRegex(OSError, '无法读取厂商配置'):
                ProviderStore(self.path)
        with patch.object(Path, 'rename', side_effect=PermissionError('denied')):
            with self.assertRaisesRegex(OSError, '无法备份'):
                ProviderStore(self.path)
        self.assertEqual(self.path.read_bytes(), b'{broken')
        self.assertEqual(list(self.path.parent.glob('*.bak')), [])

    def test_reusing_key_preserves_plaintext_and_public_response_hides_it(self):
        store = ProviderStore(self.path)
        profile, key = store.prepare(dict(PRESETS[0], api_key='fake-test-key'))
        store.save(profile)
        replacement, reused = store.prepare(dict(PRESETS[0], api_key=''))
        self.assertEqual(reused, key)
        self.assertEqual(unseal(replacement['api_key']), key)
        self.assertNotIn(key, json.dumps(store.public()))

    def test_default_preset_models_keep_distinct_reasoning_levels(self):
        for preset in PRESETS:
            with self.subTest(preset=preset['id']):
                low, _ = reasoning_options(preset['protocol'], preset['model'], 'low')
                high, note = reasoning_options(preset['protocol'], preset['model'], 'high')
                self.assertNotEqual(low, high)
                self.assertNotIn('仅适配思考开关', note)
                self.assertEqual(reasoning_options('auto', preset['model'], 'high')[0], high)
                if preset['id'] in ('zhipu', 'kimi'):
                    self.assertEqual(high['reasoning_effort'], 'high')
                    self.assertEqual(low['reasoning_effort'], 'low')
