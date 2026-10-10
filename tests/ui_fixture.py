"""Isolated browser fixture. Scripted model responses, no real API requests."""
import json
import hashlib
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server import make_server
from model_fixture import ScriptedModel, create_op
from test_recall_modes import ModeModel


def scripted_plan(messages):
    # Explicit fixture cases, never loaded by the production application.
    user = next(m['content'] for m in reversed(messages) if m.get('role') == 'user')
    if user.startswith('测试偏好：'):
        return [create_op()]
    return []


def delay_review(stage, payload):
    if stage == 'review':
        time.sleep(4)


brain = ScriptedModel(operations=scripted_plan, reason='测试模拟决定：根据本轮含义选择维护或不作改动。', hook=delay_review)
direct_brain = ModeModel(hook=lambda phase: time.sleep(4) if phase == 'review' else None)
retrieval_brain = ModeModel(extra=True, hook=lambda phase: time.sleep(4) if phase == 'review' else None)


class MockAPI(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    def log_message(self, *_): pass

    def do_POST(self):
        data = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if self.path.endswith('/embeddings'):
            dimensions = data.get('dimensions', 8)
            vectors = []
            for index, text in enumerate(data['input']):
                digest = hashlib.sha256(text.encode()).digest()
                vectors.append(dict(index=index, embedding=[(digest[i % 32]-128)/128 for i in range(dimensions)]))
            result = dict(data=vectors, model=data['model'], usage={'total_tokens': len(vectors)*5})
        else:
            model = brain
            if data['model'] == 'recall-modes-fixture':
                messages = data.get('input', data.get('messages'))
                last_user = next(m['content'] for m in reversed(messages) if m.get('role') == 'user')
                model = retrieval_brain if last_user.startswith('测试按需检索：') else direct_brain
            result = model('http://127.0.0.1:9876' + self.path, 'fixture-key', data)
        # Synthetic usage and hosted-search events for UI assertions, never a real search.
        if self.path.endswith('/responses'):
            result['usage']['output_tokens_details'] = {'reasoning_tokens': 3}
            if any(t['type'] == 'web_search' for t in data.get('tools', [])):
                result['output'].insert(0, {'type': 'web_search_call', 'id': 'test-search', 'status': 'completed'})
                result['output'][-1]['content'][0]['annotations'] = [{'type': 'url_citation', 'url': 'https://example.com/mori-test', 'title': '测试模拟搜索来源'}]
        elif not self.path.endswith('/embeddings'):
            result['usage']['completion_tokens_details'] = {'reasoning_tokens': 0 if data.get('thinking', {}).get('type') == 'disabled' else 3}
        chunks = []
        if data.get('stream'):
            def event(value):
                return ('data: ' + json.dumps(value, ensure_ascii=False) + '\n\n').encode()
            if self.path.endswith('/responses'):
                for item in result['output']:
                    if item['type'] == 'function_call':
                        chunks.append(event({'type': 'response.function_call_arguments.delta', 'delta': item['arguments']}))
                    elif item['type'] == 'message':
                        text = item['content'][0]['text']
                        split = max(1, len(text) // 2)
                        chunks.extend(event({'type': 'response.output_text.delta', 'delta': part}) for part in (text[:split], text[split:]))
                chunks.append(event({'type': 'response.completed', 'response': result}))
            else:
                message = result['choices'][0]['message']
                if message.get('tool_calls'):
                    if message.get('content'):
                        chunks.append(event({'choices': [{'index': 0, 'delta': {'content': message['content']}}]}))
                    tools = [dict(index=i, **t) for i, t in enumerate(message['tool_calls'])]
                    chunks.append(event({'choices': [{'index': 0, 'delta': {'tool_calls': tools}, 'finish_reason': 'tool_calls'}]}))
                else:
                    text = message['content']; split = max(1, len(text) // 2)
                    chunks.extend(event({'choices': [{'index': 0, 'delta': {'content': part}}]}) for part in (text[:split], text[split:]))
                    chunks.append(event({'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]}))
                chunks.extend([event({'choices': [], 'usage': result['usage']}), b'data: [DONE]\n\n'])
        body = b''.join(chunks) if chunks else json.dumps(result, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream' if chunks else 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        if chunks:
            main = '[MORI_MAIN]' in data.get('input', data.get('messages'))[0]['content']
            for chunk in chunks:
                self.wfile.write(chunk); self.wfile.flush()
                if main: time.sleep(0.5)
        else:
            self.wfile.write(body)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'artifacts' / 'ui-data')
    args = parser.parse_args()
    mock = ThreadingHTTPServer(('127.0.0.1', 9876), MockAPI)
    threading.Thread(target=mock.serve_forever, daemon=True).start()
    server = make_server(8766, args.data_dir)
    print('UI fixture: http://127.0.0.1:8766 | scripted API: http://127.0.0.1:9876', flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        mock.shutdown()
        mock.server_close()
