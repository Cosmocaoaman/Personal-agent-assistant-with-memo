"""Scripted wire responses used ONLY by tests; not a model-quality evaluation."""
import json
import threading

from agent.providers import ProviderError


def create_op(content='偏好简洁的中文回复', key='preference.style'):
    return dict(operation='create', id='', key=key, content=content, category='preference', evidence='用户表达了希望简短回答的偏好')


class ScriptedModel:
    def __init__(self, operations=None, reason='这是一项可跨会话使用的稳定偏好。', fail_stage=None, hook=None, direct_context=False):
        self.operations = operations if operations is not None else [create_op()]
        self.reason = reason
        self.fail_stage = fail_stage
        self.hook = hook
        self.direct_context = direct_context
        self.calls = []
        self.lock = threading.Lock()

    def __call__(self, url, key, payload):
        responses = url.endswith('/responses')
        messages = payload['input' if responses else 'messages']
        system = messages[0]['content']
        stage = 'context' if '[MORI_MEMORY_CONTEXT]' in system else 'review' if '[MORI_MEMORY_REVIEW]' in system else 'main'
        with self.lock:
            self.calls.append({'stage': stage, 'payload': json.loads(json.dumps(payload)), 'thread': threading.get_ident()})
        if self.hook:
            self.hook(stage, payload)
        if stage == self.fail_stage:
            raise ProviderError('测试模拟：该分支 API 暂不可用')
        call = None
        if stage == 'context' and self.direct_context:
            call = ('memory_context', {'summary': '', 'memory_ids': []})
        elif stage == 'context':
            results = [m for m in messages if m.get('role') == 'tool' or m.get('type') == 'function_call_output']
            if not results:
                call = ('memory_list', {'offset': 0})
            else:
                last = results[-1]
                rows = json.loads(last.get('output') or last.get('content')).get('memories', [])
                call = ('memory_context', {'summary': '；'.join(m['content'] for m in rows[:3]), 'memory_ids': [m['id'] for m in rows[:3]]})
        elif stage == 'review':
            plan = self.operations(messages) if callable(self.operations) else self.operations
            call = ('memory_decide', {'reason': self.reason, 'operations': plan})
        if call:
            name, args = call
            raw = json.dumps(args, ensure_ascii=False)
            if responses:
                return {'status': 'completed', 'output': [{'type': 'function_call', 'call_id': f'{stage}-call', 'name': name, 'arguments': raw}], 'usage': {'total_tokens': 10}}
            return {'choices': [{'message': {'role': 'assistant', 'content': None, 'reasoning_content': 'opaque-fixture', 'tool_calls': [
                {'id': f'{stage}-call', 'type': 'function', 'function': {'name': name, 'arguments': raw}}]}}], 'usage': {'total_tokens': 10}}
        answer = '【测试模拟响应】已收到，我会简洁地回答。'
        if any('你记得' in m.get('content', '') for m in messages if m.get('role') == 'user'):
            answer = '【测试模拟响应】你的偏好是简洁的中文回复。' if '偏好简洁' in system else '【测试模拟响应】没有可用的偏好。'
        if responses:
            return {'status': 'completed', 'output': [{'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': answer}]}], 'usage': {'total_tokens': 10}}
        return {'choices': [{'message': {'role': 'assistant', 'content': answer}}], 'usage': {'total_tokens': 10}}
