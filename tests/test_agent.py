import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

from agent.providers import Provider, ProviderError, validate_base
from agent.store import Store
from agent.trace import RunTrace
from server import App, make_server


class StoreFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.file = Path(self.temp.name) / 'memory.db'
        self.store = Store(self.file)

    def tearDown(self):
        self.temp.cleanup()


class StoreTests(StoreFixture):
    def test_memory_survives_restart_and_session_deletion(self):
        session = self.store.create_session()
        user = self.store.add_message(session['id'], 'user', '请记住我喜欢中文回答')
        self.store.save_memory('我喜欢中文回答', 'preference', 'language', source=user['id'])
        self.store.delete_session(session['id'])
        restarted = Store(self.file)
        self.assertEqual(restarted.search('中文')[0]['content'], '我喜欢中文回答')
        self.assertEqual(restarted.sessions(), [])

    def test_updates_dedupe_and_forgetting(self):
        first = self.store.save_memory('住在悉尼', 'profile', 'city')
        second = self.store.save_memory('住在墨尔本', 'profile', 'city')
        self.assertEqual(first['id'], second['id'])
        self.store.save_memory('住在墨尔本', 'profile')
        self.assertEqual(len(self.store.memories()), 1)
        self.store.delete_memory(first['id'])
        self.assertEqual(Store(self.file).search('墨尔本'), [])

    def test_retrieval_ranks_chinese_and_english(self):
        python = self.store.save_memory('正在学习 Python 异步编程', 'project')
        self.store.save_memory('喜欢喝咖啡', 'note')
        self.assertEqual(self.store.search('Python async')[0]['id'], python['id'])
        self.assertEqual(self.store.search('异步编程')[0]['id'], python['id'])
        self.assertEqual(self.store.search('完全无关词xyz'), [])

    def test_pinned_and_profile_context_without_keyword_overlap(self):
        pinned = self.store.save_memory('请称呼我小林', 'profile', pinned=True)
        found = self.store.search('Write code', context=True)
        self.assertEqual(found[0]['id'], pinned['id'])

    def test_edit_collision_does_not_overwrite_another_memory(self):
        a = self.store.save_memory('one', key='one')
        self.store.save_memory('two', key='two')
        with self.assertRaises(ValueError):
            self.store.save_memory('changed', key='two', memory_id=a['id'])
        self.assertEqual(len(self.store.memories()), 2)

    def test_task_persistence_and_validation(self):
        task = self.store.add_task('学习 Python')
        self.store.add_task('学习 Python')
        self.assertEqual(len(self.store.tasks()), 1)
        with self.assertRaises(ValueError): self.store.set_task(task['id'], 'false')
        self.store.set_task(task['id'], True)
        self.assertEqual(Store(self.file).tasks()[0]['done'], 1)

    def test_export_includes_messages_and_memory(self):
        session = self.store.create_session()
        self.store.add_message(session['id'], 'user', 'hello')
        self.store.save_memory('中文内容')
        dump = self.store.export()
        self.assertEqual(dump['sessions'][0]['messages'][0]['content'], 'hello')
        self.assertEqual(dump['memories'][0]['content'], '中文内容')


class HTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.env = patch.dict(os.environ, {'OPENAI_API_KEY': '', 'DEEPSEEK_API_KEY': ''})
        cls.env.start()
        cls.server = make_server(0, cls.temp.name)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = 'http://127.0.0.1:' + str(cls.server.server_address[1])
        cls.token = cls.server.app.csrf

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown(); cls.server.server_close(); cls.thread.join()
        cls.env.stop(); cls.temp.cleanup()

    def request(self, route, data=None, headers=None):
        h = {'Content-Type': 'application/json', 'X-Mori-Token': self.token}
        h.update(headers or {})
        req = urllib.request.Request(self.base + route, data=None if data is None else json.dumps(data).encode(), headers=h)
        try:
            with urllib.request.urlopen(req) as r: return r.status, r.read().decode(), r.headers
        except urllib.error.HTTPError as r:
            return r.code, r.read().decode(), r.headers

    def test_static_files_and_security_headers(self):
        status, html, headers = self.request('/')
        self.assertEqual(status, 200)
        self.assertIn('木记', html)
        self.assertIn("frame-ancestors 'none'", headers['Content-Security-Policy'])
        for secret in ['/.env', '/data/mori.sqlite3', '/data/mori.sqlite3-wal', '/../server.py', '/server.py']:
            self.assertEqual(self.request(secret)[0], 404)

    def test_csrf_and_origin_and_host_rejected(self):
        self.assertEqual(self.request('/api/sessions', {}, {'X-Mori-Token': ''})[0], 403)
        self.assertEqual(self.request('/api/state', headers={'Origin': 'https://evil.example'})[0], 403)
        self.assertEqual(self.request('/api/state', headers={'Host': 'evil.example'})[0], 403)

    def test_flow_assets_and_history_endpoints(self):
        for file in ('/flow.js', '/flow.css'):
            self.assertEqual(self.request(file)[0], 200)
        session = self.server.app.store.create_session()
        trace = RunTrace(self.server.app.store, session['id'], 'HTTP 流程测试', 'model', 'deepseek', lambda *_: None)
        trace.begin('prepare')
        status, body, _ = self.request('/api/runs?session=' + session['id'])
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)[0]['id'], trace.data['id'])
        status, body, _ = self.request('/api/runs/detail?id=' + trace.data['id'])
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)['nodes']['prepare']['status'], 'running')
        self.assertEqual(self.request('/api/runs/detail?id=missing')[0], 400)
        self.server.app.store.delete_session(session['id'])

    def test_mutations_and_export(self):
        status, content, _ = self.request('/api/memories/save', {'content': '我喜欢茶', 'category': 'preference', 'key': 'drink'})
        self.assertEqual(status, 200)
        memory = json.loads(content)
        self.assertIn('我喜欢茶', self.request('/api/export')[1])
        self.request('/api/memories/delete', {'id': memory['id']})
        self.assertNotIn('我喜欢茶', self.request('/api/memories')[1])

    def test_state_previews_personal_context_without_new_messages_or_model_calls(self):
        item = self.server.app.store.save_memory('默认使用简体中文', 'preference', 'preview.language')
        before = self.server.app.store.sessions()
        status, raw, _ = self.request('/api/state')
        self.assertEqual(status, 200)
        state = json.loads(raw)
        self.assertIn(item['id'], [m['id'] for m in state['personal_context']['memories']])
        self.assertEqual(self.server.app.store.sessions(), before)
        self.server.app.store.delete_memory(item['id'])

    def test_keys_persist_privately_and_endpoint_change_clears_key(self):
        config = dict(provider='openai', model='test-model', base_url='https://api.openai.com/v1', api_key='sentinel-secret')
        self.assertEqual(self.request('/api/settings', config)[0], 200)
        self.assertNotIn('sentinel-secret', self.request('/api/state')[1])
        self.assertNotIn('sentinel-secret', self.request('/api/export')[1])
        self.assertNotIn('sentinel-secret', json.dumps(self.server.app.store.settings()))
        self.assertEqual(App(self.temp.name).keys['openai'], 'sentinel-secret')
        config.update(base_url='https://gateway.example/v1', api_key='')
        self.request('/api/settings', config)
        self.assertNotIn('openai', self.server.app.keys)
        self.assertNotIn('openai', App(self.temp.name).keys)

    def test_missing_key_and_oversized_message(self):
        self.request('/api/settings', dict(provider='deepseek', model='test', base_url='https://api.deepseek.com', clear_key=True))
        session = json.loads(self.request('/api/sessions', {})[1])
        status, _, _ = self.request('/api/chat', dict(session_id=session['id'], message='hello'))
        self.assertEqual(status, 400)
        self.assertEqual(self.request('/api/chat', dict(session_id=session['id'], message='x' * 12001))[0], 400)
        self.assertEqual(self.server.app.store.messages(session['id']), [])

    def test_concurrent_mutation_rejected(self):
        self.server.app.lock.acquire()
        try: self.assertEqual(self.request('/api/tasks/add', {'title': 'should not save'})[0], 409)
        finally: self.server.app.lock.release()

    def test_endpoint_validation(self):
        for bad in ['http://evil.example', 'file:///etc/passwd', 'https://user:pass@example.com', 'https://api.example?a=b']:
            with self.assertRaises(ValueError): validate_base(bad)
        self.assertEqual(validate_base('http://127.0.0.1:1234/'), 'http://127.0.0.1:1234')


class LifecycleTests(unittest.TestCase):
    def test_cli_stops_only_matching_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            server = make_server(0, directory)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            port = str(server.server_address[1])
            command = [sys.executable, str(Path(__file__).resolve().parents[1] / 'server.py'), '--port', port, '--stop', '--data-dir']
            try:
                wrong = subprocess.run(command + [str(Path(directory) / 'different')], capture_output=True, timeout=10)
                self.assertEqual(wrong.returncode, 1)
                self.assertTrue(thread.is_alive())
                correct = subprocess.run(command + [directory], capture_output=True, timeout=10)
                self.assertEqual(correct.returncode, 0, correct.stdout)
                thread.join(timeout=5)
                self.assertFalse(thread.is_alive())
            finally:
                server.shutdown(); server.server_close(); thread.join(timeout=5)


if __name__ == '__main__':
    unittest.main()
