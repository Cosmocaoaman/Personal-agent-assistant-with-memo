import json
import tempfile
import unittest
from pathlib import Path
import httpx

from agent.config import model_options, runtime_options
from agent.core import Agent
from agent.providers import Provider, ProviderError
from agent.transport import PooledHTTP
from agent.store import Store
from model_fixture import ScriptedModel
from server import App


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.app = App(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def configure(self, provider='deepseek', **extra):
        return self.app.configure(dict(provider=provider, base_url='http://127.0.0.1:9876', model='test-model', **extra))

    def test_legacy_settings_preserve_previous_behavior(self):
        self.app.store.set_settings({'deepseek': {'base_url': 'https://api.deepseek.com', 'model': 'my-existing-model'}})
        reopened = App(self.temp.name)
        self.assertEqual(reopened.settings['deepseek']['model'], 'my-existing-model')
        self.assertEqual(reopened.settings['deepseek']['options']['reasoning_effort'], 'none')
        self.assertEqual(reopened.settings['openai']['options']['reasoning_effort'], 'auto')
        self.assertFalse(reopened.settings['openai']['options']['web_search'])

    def test_provider_options_runtime_and_keys_survive_restart(self):
        self.configure(options={'reasoning_effort': 'max', 'max_output_tokens': 8192}, api_key='secret-sentinel', runtime={'memory_steps': 3})
        self.configure('openai', options={'reasoning_effort': 'high', 'web_search': True})
        reopened = App(self.temp.name)
        self.assertEqual(reopened.settings['deepseek']['options']['reasoning_effort'], 'max')
        self.assertTrue(reopened.settings['openai']['options']['web_search'])
        self.assertEqual(reopened.settings['runtime']['memory_steps'], 3)
        self.assertEqual(reopened.keys['deepseek'], 'secret-sentinel')
        self.assertNotIn('secret-sentinel', json.dumps(reopened.store.settings()))
        self.assertNotIn('secret-sentinel', json.dumps(self.app.public_settings()))

    def test_invalid_configuration_does_not_change_settings_or_clear_key(self):
        self.configure(api_key='existing-key')
        before = json.dumps(self.app.settings, sort_keys=True)
        with self.assertRaises(ValueError):
            self.app.configure(dict(provider='deepseek', model='other', base_url='https://other.example', clear_key=True, options={'web_search': True}))
        self.assertEqual(json.dumps(self.app.settings, sort_keys=True), before)
        self.assertEqual(self.app.keys['deepseek'], 'existing-key')

    def test_numeric_enum_and_capability_validation(self):
        bad = [{'max_output_tokens': True}, {'max_output_tokens': 0}, {'timeout_seconds': 301},
               {'temperature': float('nan')}, {'top_p': 0}, {'web_search': 'true'},
               {'reasoning_effort': 'invalid'}, {'unknown_option': 1}]
        for options in bad:
            with self.subTest(options=options), self.assertRaises(ValueError): model_options('openai', options)
        for options in [{'reasoning_effort': 'medium'}, {'web_search': True}, {'verbosity': 'high'},
                        {'reasoning_effort': 'high', 'temperature': 1}, {'top_p': 0.8}]:
            with self.subTest(options=options), self.assertRaises(ValueError): model_options('deepseek', options)
        for limits in [{'history_messages': -1}, {'personal_items': 101}, {'memory_steps': 13}, {'tool_calls_per_step': 5}]:
            with self.subTest(limits=limits), self.assertRaises(ValueError): runtime_options(limits)


class AdapterTests(unittest.TestCase):
    def client(self, provider, model=None, **options):
        self.transport = model or ScriptedModel(operations=[])
        return Provider(provider, 'http://127.0.0.1:9876', 'test-model', 'test-key', self.transport, options=options)

    def test_deepseek_thinking_and_output_limit_reach_wire(self):
        client = self.client('deepseek', reasoning_effort='max', max_output_tokens=32768, top_p=0.98)
        client.complete(client.start('[MORI_MAIN]', [{'role': 'user', 'content': 'Hi'}]), [], final=True)
        payload = self.transport.calls[0]['payload']
        self.assertEqual(payload['thinking'], {'type': 'enabled'})
        self.assertEqual(payload['reasoning_effort'], 'max')
        self.assertEqual(payload['max_tokens'], 32768)
        self.assertEqual(payload['top_p'], 0.98)
        self.assertNotIn('temperature', payload)

    def test_deepseek_disabled_and_auto_modes_are_distinct(self):
        for effort in ('none', 'auto'):
            client = self.client('deepseek', reasoning_effort=effort)
            client.complete(client.start('[MORI_MAIN]', []), [], final=True)
            payload = self.transport.calls[0]['payload']
            if effort == 'auto': self.assertNotIn('thinking', payload)
            else: self.assertEqual(payload['thinking'], {'type': 'disabled'})

    def test_openai_parameters_and_web_are_main_only(self):
        client = self.client('openai', reasoning_effort='high', max_output_tokens=16000, verbosity='low', web_search=True, search_context_size='low')
        client.complete(client.start('[MORI_MAIN]', []), [], final=True)
        payload = self.transport.calls[-1]['payload']
        self.assertEqual(payload['reasoning'], {'effort': 'high'})
        self.assertEqual(payload['max_output_tokens'], 16000)
        self.assertEqual(payload['text'], {'verbosity': 'low'})
        self.assertEqual(payload['tools'], [{'type': 'web_search', 'search_context_size': 'low', 'external_web_access': True}])
        self.assertEqual(payload['tool_choice'], 'auto')
        client.complete(client.start('[MORI_MEMORY_CONTEXT]', []), [])
        self.assertEqual(self.transport.calls[-1]['payload']['tools'], [])
        self.assertNotIn('text', self.transport.calls[-1]['payload'])
        client.probe()
        self.assertEqual(self.transport.calls[-1]['payload']['tools'], [])
        self.assertEqual(self.transport.calls[-1]['payload']['tool_choice'], 'none')

    def test_timeout_is_used_by_http_transport(self):
        timeouts = []
        def handler(request):
            timeouts.append(request.extensions['timeout'])
            return httpx.Response(200, json={'choices': [{'message': {'content': 'OK'}}]})
        with httpx.Client(transport=httpx.MockTransport(handler)) as http:
            client = Provider('deepseek', 'http://127.0.0.1:9876', 'model', 'test-key',
                PooledHTTP(http), options={'timeout_seconds': 35})
            client.probe()
        self.assertEqual(timeouts[0]['read'], 35)
        self.assertEqual(timeouts[0]['connect'], 35)

    def test_reported_reasoning_usage_and_safe_sources(self):
        result = {'status': 'completed', 'usage': {'total_tokens': 100, 'output_tokens_details': {'reasoning_tokens': 60}},
                  'output': [{'type': 'web_search_call', 'id': 'search', 'status': 'completed'},
                             {'type': 'reasoning', 'encrypted_content': 'hidden-test-reasoning'},
                             {'type': 'message', 'content': [{'type': 'output_text', 'text': 'Answer', 'annotations': [
                                 {'type': 'url_citation', 'url': 'https://example.com/source', 'title': 'Source'},
                                 {'type': 'url_citation', 'url': 'javascript:alert(1)', 'title': 'Bad'}]}]}]}
        client = self.client('openai', model=lambda *_: result, web_search=True)
        text, calls, usage = client.complete(client.start('[MORI_MAIN]', []), [], final=True)
        self.assertEqual(text, 'Answer')
        self.assertEqual(calls, [])
        self.assertEqual(usage['reasoning_tokens'], 60)
        self.assertEqual(usage['sources'], [{'url': 'https://example.com/source', 'title': 'Source'}])
        self.assertEqual(len(usage['web_searches']), 1)

    def test_missing_reasoning_usage_stays_unknown(self):
        for provider in ('openai', 'deepseek'):
            client = self.client(provider)
            _, _, usage = client.complete(client.start('[MORI_MAIN]', []), [], final=True)
            self.assertIsNone(usage['reasoning_tokens'])


class RuntimeTests(unittest.TestCase):
    def test_runtime_limits_and_configuration_snapshot_are_effective(self):
        with tempfile.TemporaryDirectory() as root:
            store = Store(Path(root) / 'test.db')
            store.save_memory('用中文回答', 'preference', 'language')
            session = store.create_session()
            store.add_message(session['id'], 'user', '不应被载入的旧消息')
            model = ScriptedModel(operations=[], direct_context=True)
            provider = Provider('deepseek', 'http://127.0.0.1:9876', 'model', 'test-key', model, options={'reasoning_effort': 'high'})
            runtime = {'history_messages': 0, 'personal_items': 0, 'memory_steps': 1, 'tool_calls_per_step': 1}
            result = Agent(store, provider, runtime=runtime).run(session['id'], '本轮消息', lambda *_: None)
            self.assertEqual(result['context']['personal']['memory_ids'], [])
            for call in model.calls:
                self.assertNotIn('不应被载入的旧消息', json.dumps(call['payload'], ensure_ascii=False))
                self.assertEqual(call['payload']['reasoning_effort'], 'high')
            trace = store.run(result['answer']['meta']['run_id'])
            self.assertEqual(trace['configuration']['runtime']['memory_steps'], 1)
            self.assertEqual(trace['configuration']['model_options']['reasoning_effort'], 'high')
            self.assertEqual(trace['nodes']['prepare']['detail']['personal_count'], 0)

    def test_call_limit_stops_recall_and_preserves_other_branches(self):
        with tempfile.TemporaryDirectory() as root:
            store = Store(Path(root) / 'test.db')
            session = store.create_session()
            model = ScriptedModel(operations=[])
            provider = Provider('deepseek', 'http://127.0.0.1:9876', 'model', 'key', model)
            result = Agent(store, provider, runtime={'memory_steps': 1}).run(session['id'], '你好', lambda *_: None)
            self.assertFalse(result['context']['ok'])
            self.assertEqual(len([c for c in model.calls if c['stage'] == 'context']), 1)
            self.assertFalse(result['answer']['meta'].get('error'))
            self.assertEqual(result['memory_result']['status'], 'skipped')

    def test_main_search_trace_keeps_memory_child_offline_and_hides_reasoning(self):
        with tempfile.TemporaryDirectory() as root:
            store = Store(Path(root) / 'test.db')
            session = store.create_session()
            model = ScriptedModel(operations=[], direct_context=True)
            def transport(url, key, payload):
                data = model(url, key, payload)
                data['usage']['output_tokens_details'] = {'reasoning_tokens': 3}
                if '[MORI_MAIN]' in payload['input'][0]['content']:
                    data['output'].insert(0, {'type': 'reasoning', 'encrypted_content': 'opaque-hidden-reasoning'})
                    data['output'].insert(1, {'type': 'web_search_call', 'id': 'search', 'status': 'completed'})
                    data['output'][-1]['content'][0]['annotations'] = [{'type': 'url_citation', 'url': 'https://example.com', 'title': 'Source'}]
                return data
            provider = Provider('openai', 'http://127.0.0.1:9876', 'model', 'key', transport, options={'web_search': True})
            result = Agent(store, provider).run(session['id'], '查询资料', lambda *_: None)
            meta = result['answer']['meta']
            self.assertEqual(meta['sources'][0]['url'], 'https://example.com')
            self.assertEqual(meta['reasoning_usage'], {'answer': 3, 'recall': 3, 'memory': 3})
            self.assertEqual(meta['tools'][0]['name'], 'web_search')
            for call in model.calls:
                if call['stage'] != 'main': self.assertFalse(any(t['type'] == 'web_search' for t in call['payload']['tools']))
            trace = store.run(meta['run_id'])
            self.assertEqual(trace['nodes']['answer']['reasoning_tokens'], 3)
            self.assertEqual(trace['nodes']['answer']['tools'], 1)
            self.assertNotIn('opaque-hidden-reasoning', json.dumps(trace))


if __name__ == '__main__': unittest.main()
