"""Thread-safe execution records, independent of model prompts and hidden reasoning."""
from __future__ import annotations

import copy
import threading
import time

from .store import now, uid


NODES = {
    'prepare': ('接收消息', 'system'),
    'recall': ('整理上下文', 'memory'),
    'answer': ('生成回答', 'main'),
    'consolidate': ('判断与维护记忆', 'memory'),
    'finish': ('汇总结果', 'system'),
}
EDGES = [('prepare', 'recall'), ('recall', 'answer'), ('recall', 'consolidate'),
         ('answer', 'finish'), ('consolidate', 'finish')]


class RunTrace:
    def __init__(self, store, session_id, preview, model, provider, emit, configuration=None):
        self.store, self.emit = store, emit
        self.clock = time.monotonic()
        self.lock = threading.RLock()
        self.data = dict(id=uid(), session_id=session_id, preview=preview[:90], model=model, provider=provider,
                         started_at=now(), status='running', elapsed_ms=0, events=[], edges=EDGES, configuration=configuration or {},
                         nodes={key: dict(id=key, label=label, agent=agent, status='waiting',
                            started_ms=None, ended_ms=None, duration_ms=None, tokens=0, reasoning_tokens=None, model_calls=0, tools=0, requests=[], detail={})
                            for key, (label, agent) in NODES.items()})
        if (configuration or {}).get('runtime', {}).get('recall_mode') == 'main':
            self.data['nodes']['recall'].update(label='载入基础上下文', agent='system')

    def _record(self, kind, node=None, patch=None, **extra):
        with self.lock:
            elapsed = round((time.monotonic() - self.clock) * 1000)
            self.data['elapsed_ms'] = elapsed
            event = dict(seq=len(self.data['events']) + 1, kind=kind, node=node, elapsed_ms=elapsed, **extra)
            if patch is not None:
                self.data['nodes'][node].update(patch)
                event['patch'] = copy.deepcopy(patch)
            self.data['events'].append(event)
            snapshot = copy.deepcopy(self.data)
            self.store.save_run(snapshot)
            self.emit('flow', {'trace': snapshot})

    def begin(self, node):
        with self.lock:
            self._record('node_start', node, {'status': 'running', 'started_ms': round((time.monotonic() - self.clock) * 1000)},
                         message=self.data['nodes'][node]['label'] + '开始')

    def end(self, node, success=True, detail=None, tokens=None):
        with self.lock:
            item = self.data['nodes'][node]
            elapsed = round((time.monotonic() - self.clock) * 1000)
            patch = dict(status='succeeded' if success else 'failed', ended_ms=elapsed,
                         duration_ms=max(0, elapsed - item['started_ms']), detail=detail or {})
            if tokens is not None:
                patch['tokens'] = tokens
            self._record('node_end', node, patch, message=item['label'] + ('完成' if success else '未完成'))

    def model(self, node, phase, success=True, tokens=0, reasoning_tokens=None, request_timing=None):
        with self.lock:
            item = self.data['nodes'][node]
            patch = {'model_calls': item['model_calls'] + 1} if phase == 'start' else {'tokens': item['tokens'] + tokens}
            if phase == 'end' and reasoning_tokens is not None:
                patch['reasoning_tokens'] = (item.get('reasoning_tokens') or 0) + reasoning_tokens
            if phase == 'end' and request_timing:
                patch['requests'] = item['requests'] + [dict(request_timing, ok=success)]
            self._record('model_' + phase, node, patch, ok=success, message='调用模型' if phase == 'start' else ('模型返回' if success else '模型调用失败'))

    def first_output(self, node):
        with self.lock:
            item = self.data['nodes'][node]
            patch = {} if 'first_output_ms' in item else {'first_output_ms': round((time.monotonic() - self.clock) * 1000)}
            self._record('model_first_output', node, patch, message='收到首个输出片段')

    def tool(self, node, name, success):
        with self.lock:
            self._record('tool', node, {'tools': self.data['nodes'][node]['tools'] + 1},
                         name=name, ok=success, message=name + (' 成功' if success else ' 未成功'))

    def complete(self):
        with self.lock:
            self.data['status'] = 'partial' if any(n['status'] == 'failed' for n in self.data['nodes'].values()) else 'completed'
            self._record('run_end', message='本轮运行结束', status=self.data['status'])

    def fail(self):
        with self.lock:
            elapsed = round((time.monotonic() - self.clock) * 1000)
            for node, item in self.data['nodes'].items():
                if item['status'] == 'running':
                    self._record('node_end', node, dict(status='failed', ended_ms=elapsed,
                        duration_ms=max(0, elapsed - item['started_ms']), detail={'error': '流程意外终止'}), message='节点意外终止')
                elif item['status'] == 'waiting':
                    self._record('node_cancelled', node, {'status': 'cancelled'}, message='节点未执行')
            self.data['status'] = 'failed'
            self._record('run_end', status='failed', message='本轮运行异常结束')
