import json
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx

from agent.core import Agent, recent_messages
from agent.providers import Provider, ProviderError
from agent.store import Store
from agent.streaming import StreamResult
from agent.transport import PooledHTTP
from model_fixture import ScriptedModel


def sse(event):
    return ('data: ' + json.dumps(event, ensure_ascii=False) + '\r\n\r\n').encode()


class BytesStream(httpx.SyncByteStream):
    def __init__(self, chunks): self.chunks = chunks
    def __iter__(self): yield from self.chunks


class StreamingTests(unittest.TestCase):
    def client(self, provider, chunks, headers=None):
        def handle(request):
            self.payload = json.loads(request.content)
            return httpx.Response(200, headers=headers or {'content-type': 'text/event-stream'}, stream=BytesStream(chunks))
        http = httpx.Client(transport=httpx.MockTransport(handle))
        self.addCleanup(http.close)
        return Provider(provider, 'http://127.0.0.1:9999', 'fixture', 'secret-test-only', PooledHTTP(http))

    def test_openai_deltas_arrive_before_complete_and_usage_is_preserved(self):
        seen, first = [], []
        response = {'status': 'completed', 'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': '你好世界'}]}],
                    'usage': {'total_tokens': 12, 'output_tokens_details': {'reasoning_tokens': 0}}}
        def chunks():
            # Splits inside UTF-8, comments, CRLF, and JSON across network chunks.
            wire = b': keepalive\r\n\r\n' + sse({'type': 'response.output_text.delta', 'delta': '你好'})
            for offset in range(0, len(wire), 3): yield wire[offset:offset + 3]
            self.assertEqual(seen, ['你好'])
            yield sse({'type': 'response.output_text.delta', 'delta': '世界'})
            yield sse({'type': 'response.completed', 'response': response})
        client = self.client('openai', chunks())
        text, calls, usage = client.complete(client.start('test', []), [], final=True,
            on_delta=seen.append, on_first_output=lambda: first.append(True))
        self.assertEqual(text, '你好世界')
        self.assertEqual(calls, [])
        self.assertEqual(first, [True])
        self.assertEqual(usage['reasoning_tokens'], 0)
        self.assertTrue(self.payload['stream'])
        self.assertTrue(usage['request_timing']['streaming'])
        self.assertIsNotNone(usage['request_timing']['first_output_ms'])

    def test_openai_tool_arguments_use_completed_response_only(self):
        call = {'type': 'function_call', 'call_id': 'call-1', 'name': 'memory_context', 'arguments': '{"summary":"","memory_ids":[]}'}
        hidden = {'type': 'reasoning', 'encrypted_content': 'opaque-secret-reasoning'}
        chunks = [sse({'type': 'response.function_call_arguments.delta', 'delta': '{"summary":'}),
                  sse({'type': 'response.completed', 'response': {'status': 'completed', 'output': [hidden, call]}})]
        client = self.client('openai', chunks)
        visible, first = [], []
        history = client.start('test', [])
        _, calls, usage = client.complete(history, [], on_delta=visible.append, on_first_output=lambda: first.append(True))
        self.assertEqual(calls[0]['arguments'], call['arguments'])
        self.assertEqual(visible, [])
        self.assertEqual(first, [True])
        self.assertIn(hidden, history)
        self.assertNotIn('opaque-secret-reasoning', json.dumps(usage))

    def test_chat_tool_fragments_and_hidden_reasoning_are_assembled(self):
        chunks = [sse({'choices': [{'index': 0, 'delta': {'reasoning_content': 'hidden-fixture', 'tool_calls': None}}]}),
                  sse({'choices': [{'index': 0, 'delta': {'tool_calls': [{'index': 0, 'id': 'call-1', 'function': {'name': 'memory_context', 'arguments': '{"summary":'}}]}}]}),
                  sse({'choices': [{'index': 0, 'delta': {'tool_calls': [{'index': 0, 'function': {'arguments': '"","memory_ids":[]}'}}]}, 'finish_reason': 'tool_calls'}]}),
                  sse({'choices': [], 'usage': {'total_tokens': 15}}), b'data: [DONE]\n\n']
        client = self.client('deepseek', chunks)
        visible = []
        history = client.start('test', [])
        _, calls, usage = client.complete(history, [], on_delta=visible.append)
        self.assertEqual(json.loads(calls[0]['arguments']), {'summary': '', 'memory_ids': []})
        self.assertEqual(visible, [])
        self.assertEqual(history[-1]['reasoning_content'], 'hidden-fixture')
        self.assertEqual(usage['total_tokens'], 15)
        self.assertEqual(self.payload['stream_options'], {'include_usage': True})

    def test_incomplete_streams_fail_instead_of_executing_partial_tools(self):
        cases = [('openai', [sse({'type': 'response.function_call_arguments.delta', 'delta': '{}'})]),
                 ('deepseek', [sse({'choices': [{'delta': {'content': 'partial'}, 'finish_reason': 'stop'}]})]),
                 ('openai', [sse({'type': 'error', 'message': 'secret-test-only'})])]
        for provider, chunks in cases:
            with self.subTest(provider=provider):
                client = self.client(provider, chunks)
                with self.assertRaises(ProviderError) as caught:
                    client.complete(client.start('test', []), [])
                self.assertNotIn('secret-test-only', str(caught.exception))
                self.assertIsNotNone(caught.exception.request_timing)

    def test_json_gateway_fallback_has_no_fake_first_token_metric(self):
        data = {'choices': [{'message': {'role': 'assistant', 'content': 'OK'}, 'finish_reason': 'stop'}]}
        client = self.client('deepseek', [json.dumps(data).encode()], {'content-type': 'application/json'})
        text, _, usage = client.complete(client.start('test', []), [], final=True)
        self.assertEqual(text, 'OK')
        self.assertFalse(usage['request_timing']['streaming'])
        self.assertIsNone(usage['request_timing']['first_output_ms'])

    def test_redirect_and_error_bodies_never_leak_keys(self):
        for status in (302, 401, 500):
            requests = []
            def handler(request):
                requests.append(request)
                return httpx.Response(status, headers={'location': 'http://elsewhere.invalid/'}, text='secret-test-only')
            with httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True) as http:
                with self.assertRaises(ProviderError) as caught:
                    PooledHTTP(http)('http://127.0.0.1:9999/', 'secret-test-only', {})
                self.assertEqual(len(requests), 1)
                self.assertNotIn('secret-test-only', str(caught.exception))

    def test_real_http_connection_reuse_and_request_specific_keys(self):
        requests = []
        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'
            def log_message(self, *_): pass
            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length']))
                requests.append((self.client_address, self.headers['Authorization']))
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', '11')
                self.end_headers()
                self.wfile.write(b'{"ok":true}')
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        pool = PooledHTTP()
        try:
            url = f'http://127.0.0.1:{server.server_port}'
            first = pool(url, 'one', {})['_request_timing']
            second = pool(url, 'two', {})['_request_timing']
            self.assertEqual(requests[0][0], requests[1][0])
            self.assertEqual([r[1] for r in requests], ['Bearer one', 'Bearer two'])
            self.assertFalse(first['connection_reused'])
            self.assertTrue(second['connection_reused'])
            self.assertEqual(second['connection_ms'], 0)
        finally:
            pool.close(); server.shutdown(); server.server_close(); thread.join()

    def test_graph_streaming_order_failure_persistence_and_parallel_review(self):
        with tempfile.TemporaryDirectory() as root:
            store = Store(Path(root) / 'test.sqlite3'); session = store.create_session()
            brain = ScriptedModel(operations=[], direct_context=True)
            events, review_started = [], threading.Event()
            def handler(request):
                payload = json.loads(request.content)
                result = brain(str(request.url), 'fake', payload)
                if '[MORI_MAIN]' not in payload['input'][0]['content']:
                    if '[MORI_MEMORY_REVIEW]' in payload['input'][0]['content']: review_started.set()
                    return httpx.Response(200, json=result)
                def chunks():
                    self.assertTrue(review_started.wait(2))
                    yield sse({'type': 'response.output_text.delta', 'delta': '部分回答'})
                    self.assertTrue(any(name == 'answer_delta' for name, _ in events))
                    yield sse({'type': 'response.failed', 'response': {'error': {'message': 'fake-secret'}}})
                return httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=BytesStream(chunks()))
            with httpx.Client(transport=httpx.MockTransport(handler)) as http:
                provider = Provider('openai', 'http://127.0.0.1:9999', 'fixture', 'fake', PooledHTTP(http))
                result = Agent(store, provider).run(session['id'], 'Hi', lambda n, p: events.append((n, p)))
            message = result['answer']
            self.assertTrue(message['meta']['error'])
            self.assertTrue(message['meta']['partial'])
            self.assertIn('部分回答', store.messages(session['id'])[-1]['content'])
            self.assertNotIn('fake-secret', json.dumps(events))
            self.assertEqual(len(recent_messages(store.messages(session['id']), 'next')), 2)
            trace = store.run(message['meta']['run_id'])
            self.assertTrue(trace['nodes']['answer']['requests'][0]['streaming'])
            names = [n for n, _ in events]
            self.assertLess(names.index('answer_start'), names.index('answer_delta'))
            self.assertLess(names.index('answer_delta'), names.index('answer'))
            self.assertEqual(names[-1], 'done')


if __name__ == '__main__': unittest.main()
