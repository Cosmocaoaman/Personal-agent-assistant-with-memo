"""Scripted wire probes for context capacity, not an AI-quality evaluator."""
from __future__ import annotations

import json
import tempfile
import threading
import time
from pathlib import Path

from agent.core import Agent
from agent.providers import Provider
from agent.store import Store, context_memory
from tools.memory_lab_data import dataset, seed


class ProbeModel:
    def __init__(self, mode='search', query='Python aiohttp 异步爬虫', operations=None):
        self.mode, self.query = mode, query
        self.operations = operations or []
        self.calls = []
        self.lock = threading.Lock()

    def __call__(self, url, key, payload):
        is_openai = url.endswith('/responses')
        messages = payload['input' if is_openai else 'messages']
        system = messages[0]['content']
        stage = 'recall' if '[MORI_MEMORY_CONTEXT]' in system else 'review' if '[MORI_MEMORY_REVIEW]' in system else 'answer'
        with self.lock:
            self.calls.append(dict(stage=stage, input_json_chars=len(json.dumps(messages, ensure_ascii=False)),
                                   prompt=system, tool_count=len(payload.get('tools', []))))
        if stage == 'answer':
            content = '【模拟测试】只验证真实执行编排和上下文容量，不评估模型回答。'
            if is_openai:
                return dict(status='completed', output=[dict(type='message', role='assistant', content=[dict(type='output_text', text=content)])], usage={})
            return dict(choices=[dict(message=dict(role='assistant', content=content))], usage={})
        outputs = [json.loads(m.get('output') or m.get('content')) for m in messages
                   if m.get('role') == 'tool' or m.get('type') == 'function_call_output']
        read_calls = []
        if self.mode == 'browse' and len(outputs) < 20:
            # Intentionally keep browsing even if a capacity warning is given.
            # The guard must hold independently of a cooperative model.
            read_calls = [('memory_list', {'offset': (len(outputs) + i) * 20}) for i in range(4)]
        elif self.mode == 'search' and not outputs:
            read_calls = [('memory_search', {'query': self.query})]
        if read_calls:
            calls = read_calls
        elif stage == 'recall':
            read = [r for output in outputs for r in output.get('memories', [])]
            selected = list({r['id']: r for r in read}.values())[-20:]
            calls = [('memory_context', dict(summary='【模拟交接】真实模型尚未评测。', memory_ids=[r['id'] for r in selected]))]
        else:
            operations = self.operations(outputs) if callable(self.operations) else self.operations
            calls = [('memory_decide', dict(reason='【模拟决定】仅验证事务及分支执行。', operations=operations))]
        if is_openai:
            return dict(status='completed', output=[dict(type='function_call', call_id=f'{stage}-{len(outputs)}-{i}', name=name,
                arguments=json.dumps(args, ensure_ascii=False)) for i, (name, args) in enumerate(calls)], usage={})
        return dict(choices=[dict(message=dict(role='assistant', content=None, tool_calls=[
            dict(id=f'{stage}-{len(outputs)}-{i}', type='function', function=dict(name=name, arguments=json.dumps(args, ensure_ascii=False)))
            for i, (name, args) in enumerate(calls)]))], usage={})


def probe(size=500, mode='browse'):
    with tempfile.TemporaryDirectory(prefix='mori-probe-') as directory:
        store = Store(Path(directory) / 'memory.sqlite3')
        seed(store, dataset(size))
        model = ProbeModel(mode)
        provider = Provider('deepseek', 'http://127.0.0.1:9876', 'scripted-capacity-probe', 'not-a-real-key', model)
        session = store.create_session('虚构容量测试')
        start = time.perf_counter()
        result = Agent(store, provider).run(session['id'], '这是独立的虚构容量测试。', lambda *_: None)
        elapsed = (time.perf_counter() - start) * 1000
        trace = store.run(result['answer']['meta']['run_id'])
        context = result['context']
        selected = [context_memory(context['seen'][mid]) for mid in context['memory_ids']]
        phases = {}
        for stage in ('recall', 'review', 'answer'):
            calls = [c for c in model.calls if c['stage'] == stage]
            phases[stage] = dict(calls=len(calls), max_input_json_chars=max(c['input_json_chars'] for c in calls),
                                total_input_json_chars=sum(c['input_json_chars'] for c in calls))
        return dict(size=size, mode=mode, scripted=True, model_quality_evaluated=False, real_api_calls=0,
            local_wall_ms=round(elapsed, 3), phases=phases, seen_memories=len(context['seen']),
            handoff_memories=len(context['memory_ids']), handoff_memory_json_chars=len(json.dumps(selected, ensure_ascii=False)),
            memory_status=result['memory_result']['status'], context_ok=context['ok'],
            context_read_chars=context.get('read_chars'), review_read_chars=result['memory_result'].get('read_chars'),
            context_handoff_omitted=context.get('handoff_omitted', 0), trace=trace)
