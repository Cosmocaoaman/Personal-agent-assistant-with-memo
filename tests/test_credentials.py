"""Local Key persistence, endpoint binding and configuration transaction tests."""
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.providers import DEFAULTS
from server import App


class CredentialTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {
            'DEEPSEEK_API_KEY': '', 'OPENAI_API_KEY': '', 'AGENT_PROVIDER': 'deepseek',
            'DEEPSEEK_BASE_URL': DEFAULTS['deepseek']['base_url'],
            'OPENAI_BASE_URL': DEFAULTS['openai']['base_url'],
        })
        self.env.start()
        self.app = App(self.temp.name)

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def configure(self, name='deepseek', **extra):
        data = dict(provider=name, base_url=DEFAULTS[name]['base_url'], model='test-model')
        data.update(extra)
        return self.app.configure(data)

    def test_plain_database_key_survives_a_separate_process(self):
        key = 'test-only-local-persistence-key'
        self.configure(api_key=key)
        with self.app.store.connect() as db:
            value, kind = db.execute("SELECT api_key, typeof(api_key) FROM credentials WHERE provider='deepseek'").fetchone()
        self.assertEqual((value, kind), (key, 'text'))
        code = "from server import App; import hashlib, sys; a=App(sys.argv[1]); print(hashlib.sha256(a.provider().key.encode()).hexdigest())"
        result = subprocess.run([sys.executable, '-c', code, self.temp.name],
                                cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), hashlib.sha256(key.encode()).hexdigest())
        self.assertEqual(self.app.public_settings()['key_storage'], {'method': 'local_database', 'encrypted': False})

    def test_blank_save_preserves_key_and_replacement_persists(self):
        self.configure(api_key='first-test-key')
        self.configure(api_key='  ', model='new-model')
        self.assertEqual(App(self.temp.name).provider().key, 'first-test-key')
        self.configure(api_key=' second-test-key ')
        self.assertEqual(App(self.temp.name).provider().key, 'second-test-key')
        for data in (self.app.public_settings(), self.app.store.settings(), self.app.store.export()):
            self.assertNotIn('second-test-key', json.dumps(data))

    def test_clear_wins_and_survives_restart_despite_environment(self):
        self.configure(api_key='saved-test-key')
        with patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'environment-test-key'}):
            self.configure(clear_key=True, api_key='must-not-override-clear')
            restarted = App(self.temp.name)
        self.assertNotIn('deepseek', restarted.keys)
        self.assertEqual(restarted.store.credential('deepseek')['api_key'], '')
        with self.assertRaises(ValueError):
            restarted.provider()

    def test_endpoint_change_and_change_back_do_not_restore_old_key(self):
        self.configure(api_key='official-test-key')
        self.configure(base_url='https://gateway.example/v1')
        self.configure()
        self.assertNotIn('deepseek', App(self.temp.name).keys)
        self.configure(base_url='https://gateway.example/v1', api_key='gateway-test-key')
        restarted = App(self.temp.name)
        self.assertEqual(restarted.provider().base_url, 'https://gateway.example/v1')
        self.assertEqual(restarted.provider().key, 'gateway-test-key')

    def test_provider_keys_and_workspace_databases_are_independent(self):
        self.configure(api_key='deepseek-test-key')
        self.configure('openai', api_key='openai-test-key')
        self.configure(clear_key=True)
        restarted = App(self.temp.name)
        self.assertEqual(restarted.keys, {'openai': 'openai-test-key'})
        other = App(Path(self.temp.name) / 'separate-workspace')
        self.assertEqual(other.keys, {})

    def test_legacy_environment_import_keeps_endpoint_after_env_removed(self):
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'legacy-environment-test-key',
                                    'OPENAI_BASE_URL': 'https://gateway.example/v1', 'AGENT_PROVIDER': 'openai'}):
            imported = App(self.temp.name)
        restarted = App(self.temp.name)
        self.assertEqual(restarted.provider().base_url, 'https://gateway.example/v1')
        self.assertEqual(restarted.provider().key, 'legacy-environment-test-key')
        self.assertEqual(imported.keys, restarted.keys)

    def test_environment_key_is_never_imported_for_a_different_gateway(self):
        self.app.store.set_settings({'openai': {'base_url': 'https://gateway.example/v1', 'model': 'test-model'}})
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'official-environment-test-key'}):
            restarted = App(self.temp.name)
        self.assertNotIn('openai', restarted.keys)
        self.assertIsNone(restarted.store.credential('openai'))

    def test_database_failure_rolls_back_settings_key_and_runtime_cache(self):
        self.configure(api_key='original-test-key')
        before = self.app.store.settings()
        with self.app.store.connect() as db:
            db.execute("""CREATE TRIGGER reject_key BEFORE UPDATE ON credentials
                          BEGIN SELECT RAISE(ABORT, 'simulated database failure'); END""")
        with self.assertRaises(sqlite3.IntegrityError):
            self.configure(api_key='replacement-test-key', model='should-not-save', persona='should-not-save')
        self.assertEqual(self.app.store.settings(), before)
        self.assertEqual(self.app.settings['deepseek']['model'], 'test-model')
        self.assertEqual(self.app.keys['deepseek'], 'original-test-key')
        self.assertEqual(App(self.temp.name).provider().key, 'original-test-key')


if __name__ == '__main__':
    unittest.main()
