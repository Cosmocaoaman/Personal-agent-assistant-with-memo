"""Behavioral contracts for one-pass handoff and model-owned optional retrieval."""
import json
import tempfile
import threading
import unittest
from pathlib import Path

from agent.config import runtime_options
from agent.core import Agent
from agent.providers import Provider
from agent.store import Store


class ModeModel:
    def __init__(self, extra=False, main_tool='memory_search', summary='', hook=None):
        self.extra, self.main_tool, self.summary, self.hook = extra, main_tool, summary, hook
        self.calls = []

    def __call__(self, url, key, payload):
        oa = url.endswith('/responses')
        messages = payload['input' if oa else 'messages']
        system = messages[0]['content']
        phase = 'recall' if '[MORI_MEMORY_CONTEXT]' in system else 'review' if '[MORI_MEMORY_REVIEW]' in system else 'main'
        self.calls.append((phase, json.loads(json.dumps(payload))))
        if self.hook: self.hook(phase)
        results = [json.loads(m.get('output') or m.get('content')) for m in messages if m.get('type') == 'function_call_output' or m.get('role') == 'tool']
        call, content = None, '【模拟回答】已完成。'
        if phase == 'review':
            call = ('memory_decide', {'reason': '【模拟判断】无须改动。', 'operations': []})
        elif phase == 'recall':
            candidates = json.loads(system.split('<prefetched_candidates>')[1].split('</prefetched_candidates>')[0])['memories']
            rows = candidates or [r for output in results for r in output.get('memories', [])]
            if self.extra and not rows and not results:
                call = ('memory_search', {'query': 'Aurora'})
            else:
                call = ('memory_context', {'summary': self.summary, 'memory_ids': [r['id'] for r in rows[:1]] if self.extra else []})
        elif self.extra and not results and payload.get('tools'):
            call = (self.main_tool, {'query': 'Aurora'} if self.main_tool == 'memory_search' else {'reason': '不允许', 'operations': []})
            content = '【模拟临时说明】准备查询。'
        elif self.extra and results:
            content = '【模拟回答】' + json.dumps(results[-1], ensure_ascii=False)
        usage = {'total_tokens': 10}
        if oa:
            output = [dict(type='function_call', call_id=phase + str(len(results)), name=call[0], arguments=json.dumps(call[1], ensure_ascii=False))] if call else [dict(type='message', role='assistant', content=[dict(type='output_text', text=content)])]
            if call and phase == 'main':
                output.insert(0, dict(type='message', role='assistant', content=[dict(type='output_text', text=content)]))
            return dict(status='completed', output=output, usage=usage)
        message = dict(role='assistant', content=content)
        if call: message['tool_calls'] = [dict(id=phase + str(len(results)), type='function', function=dict(name=call[0], arguments=json.dumps(call[1], ensure_ascii=False)))]
        return dict(choices=[dict(message=message)], usage=usage)


class RecallModeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.temp.name) / 'test.db')
        self.base = self.store.save_memory('默认使用中文。', 'preference', 'language')
        self.target = self.store.save_memory('Aurora 使用 PostgreSQL 和 Redis。', 'project', 'aurora')

    def tearDown(self): self.temp.cleanup()

    def turn(self, model, mode='specialist', provider='openai', **runtime):
        events = []
        client = Provider(provider, 'http://127.0.0.1:9876', 'fixture', 'fake', model)
        result = Agent(self.store, client, auto_memory=False, runtime=dict(recall_mode=mode, **runtime)).run(
            self.store.create_session()['id'], 'Aurora 用什么数据库？', lambda name, data: events.append((name, data)),
            {'route': 'daily', 'context': {'home_note': '不预设设备'}})
        return result, self.store.run(result['answer']['meta']['run_id']), events

    def test_prefetched_facts_are_selected_in_one_call_without_repeated_summary(self):
        model = ModeModel(extra=True)
        result, trace, _ = self.turn(model)
        self.assertEqual(trace['nodes']['recall']['model_calls'], 1)
        self.assertIn(self.target['id'], result['context']['memory_ids'])
        self.assertEqual(result['context']['summary'], '')
        self.assertEqual(result['context']['retrievals'][0]['name'], 'memory_prefetch')
        main = next(payload for phase, payload in model.calls if phase == 'main')
        self.assertIn('PostgreSQL', main['input'][0]['content'])
        self.assertEqual(main['tool_choice'], 'none')

    def test_unselected_candidates_do_not_leak_into_main_context(self):
        result, _, _ = self.turn(ModeModel(extra=False))
        self.assertIn(self.target['id'], result['context']['seen'])
        self.assertNotIn(self.target['id'], result['context']['memory_ids'])
        self.assertNotIn(self.target['id'], [r['id'] for r in result['answer']['meta']['recalled']])

    def test_candidates_can_be_disabled_and_specialist_can_supplement(self):
        result, trace, _ = self.turn(ModeModel(extra=True), prefetch_candidates=False)
        self.assertEqual(trace['nodes']['recall']['model_calls'], 2)
        self.assertIn(self.target['id'], result['context']['memory_ids'])
        self.assertEqual(result['context']['retrievals'][0]['name'], 'memory_search')

    def test_long_summary_is_rejected_and_baseline_survives(self):
        result, trace, _ = self.turn(ModeModel(extra=True, summary='长' * 65), handoff_summary_chars=64, memory_steps=1)
        self.assertFalse(result['context']['ok'])
        self.assertEqual(result['context']['memory_ids'], [self.base['id']])
        self.assertEqual(trace['nodes']['recall']['model_calls'], 1)

    def test_main_direct_answer_skips_recall_model_and_search_for_both_providers(self):
        for provider in ('openai', 'deepseek'):
            model = ModeModel()
            result, trace, _ = self.turn(model, mode='main', provider=provider)
            self.assertEqual([phase for phase, _ in model.calls].count('recall'), 0)
            self.assertEqual(trace['nodes']['recall']['model_calls'], 0)
            self.assertEqual(trace['nodes']['recall']['agent'], 'system')
            self.assertEqual(trace['nodes']['answer']['model_calls'], 1)
            self.assertFalse(trace['nodes']['answer']['detail']['extra_memory'])
            self.assertEqual(result['answer']['meta']['memory_tools'], [])
            payload = next(p for phase, p in model.calls if phase == 'main')
            self.assertEqual(payload['tool_choice'], 'auto')
            tools = payload['tools']
            names = [t.get('name') or t['function']['name'] for t in tools]
            self.assertEqual(names, ['memory_search', 'memory_list'])
            self.assertIn('默认使用中文', json.dumps(payload, ensure_ascii=False))
            self.assertIn('不预设设备', json.dumps(payload, ensure_ascii=False))

    def test_main_search_reads_facts_and_final_answer_has_no_tool_preamble(self):
        for provider in ('openai', 'deepseek'):
            result, trace, events = self.turn(ModeModel(extra=True), mode='main', provider=provider)
            self.assertEqual(trace['nodes']['answer']['model_calls'], 2)
            self.assertIn('PostgreSQL', result['answer']['content'])
            self.assertNotIn('模拟临时说明', result['answer']['content'])
            self.assertEqual(trace['nodes']['answer']['detail']['retrievals'][0]['name'], 'memory_search')
            self.assertEqual(result['answer']['meta']['memory_tools'][0]['agent'], 'main')
            self.assertIn('answer_reset', [name for name, _ in events])
            self.assertEqual(result['context']['memory_ids'], [self.base['id']])

    def test_main_cannot_write_even_if_model_requests_write_tool(self):
        before = self.store.memories()
        result, trace, _ = self.turn(ModeModel(extra=True, main_tool='memory_decide'), mode='main')
        self.assertEqual(self.store.memories(), before)
        self.assertFalse(result['answer']['meta']['memory_tools'][0]['ok'])
        self.assertIn('只读记忆工具', result['answer']['content'])
        self.assertEqual(trace['nodes']['answer']['model_calls'], 2)

    def test_main_answer_and_independent_maintenance_still_overlap(self):
        barrier = threading.Barrier(2, timeout=4)
        result, _, _ = self.turn(ModeModel(hook=lambda phase: barrier.wait() if phase in ('main', 'review') else None), mode='main')
        self.assertFalse(result['answer']['meta'].get('error', False))
        self.assertEqual(result['memory_result']['status'], 'readonly')

    def test_modes_and_bool_and_summary_limits_validate(self):
        for config in ({'recall_mode': 'unknown'}, {'prefetch_candidates': 'yes'}, {'handoff_summary_chars': 63}, {'handoff_summary_chars': True}):
            with self.assertRaises(ValueError): runtime_options(config)
