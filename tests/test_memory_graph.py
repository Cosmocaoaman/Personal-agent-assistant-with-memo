import json
import tempfile
import threading
import unittest
from pathlib import Path

from agent.core import Agent
from agent.providers import Provider
from agent.store import Store, context_memory
from agent.trace import RunTrace
from model_fixture import ScriptedModel, create_op


class GraphTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.temp.name) / 'test.db')

    def tearDown(self):
        self.temp.cleanup()

    def run_turn(self, model=None, text='讲重点就好，不用长篇铺垫。', provider='deepseek', auto=True, emit=None):
        model = model or ScriptedModel()
        client = Provider(provider, 'http://127.0.0.1:9876', 'test-model', 'test-key', model)
        events = []

        def callback(name, payload):
            events.append((name, payload))
            if emit:
                emit(name, payload)

        session = self.store.create_session()
        output = Agent(self.store, client, auto_memory=auto).run(session['id'], text, callback)
        return output, events, model

    def test_arbitrary_input_and_paraphrased_memory_are_model_decisions(self):
        result, events, model = self.run_turn()
        self.assertEqual(self.store.memories()[0]['content'], '偏好简洁的中文回复')
        self.assertEqual(result['memory_result']['status'], 'changed')
        self.assertEqual(events[-1][0], 'done')
        stages = {call['stage'] for call in model.calls}
        self.assertEqual(stages, {'main', 'context', 'review'})
        self.assertEqual(result['answer']['meta']['total_tokens'], 40)

    def test_first_greeting_inherits_personal_context_without_model_queries(self):
        profile = self.store.save_memory('我的称呼是小林', 'profile', 'name')
        preference = self.store.save_memory('偏好简洁的中文回复', 'preference', 'style')
        pinned = self.store.save_memory('当前默认单位使用公制', 'note', 'units', pinned=True)
        excluded = self.store.save_memory('一次性的项目会议笔记', 'project', 'meeting')
        for provider in ('deepseek', 'openai'):
            with self.subTest(provider=provider):
                result, _, model = self.run_turn(ScriptedModel(operations=[], direct_context=True), text='早上好', provider=provider)
                self.assertEqual(set(result['context']['memory_ids']), {profile['id'], preference['id'], pinned['id']})
                self.assertNotIn(excluded['id'], result['context']['memory_ids'])
                for stage in ('context', 'main', 'review'):
                    payload = next(c['payload'] for c in model.calls if c['stage'] == stage)
                    prompt = payload['input' if provider == 'openai' else 'messages'][0]['content']
                    for content in ('小林', '偏好简洁', '公制'): self.assertIn(content, prompt)
                tools = [t['name'] for t in result['answer']['meta']['tools']]
                self.assertEqual(tools, ['memory_context', 'memory_decide'])
                self.assertEqual(len(self.store.messages(result['session_id'])), 2)
                self.assertEqual(len(self.store.memories()), 4)
                trace = self.store.run(result['answer']['meta']['run_id'])
                self.assertEqual(trace['nodes']['prepare']['detail']['personal_count'], 3)
                self.assertEqual(trace['nodes']['recall']['detail']['personal_count'], 3)

    def test_personal_context_survives_failed_recall_and_readonly_mode(self):
        preference = self.store.save_memory('请叫我小林', 'preference', 'name')
        result, _, model = self.run_turn(ScriptedModel(operations=[], fail_stage='context'), text='你好', auto=False)
        self.assertFalse(result['context']['ok'])
        self.assertEqual(result['context']['memory_ids'], [preference['id']])
        prompt = next(c for c in model.calls if c['stage'] == 'main')['payload']['messages'][0]['content']
        self.assertIn('请叫我小林', prompt)
        self.assertEqual(result['memory_result']['status'], 'readonly')

    def test_personal_context_refreshes_edits_and_deletions_in_existing_session(self):
        preference = self.store.save_memory('默认用中文', 'preference', 'language')
        model = ScriptedModel(operations=[], direct_context=True)
        client = Provider('deepseek', 'http://127.0.0.1:9876', 'test-model', 'test-key', model)
        session = self.store.create_session()
        agent = Agent(self.store, client)
        first = agent.run(session['id'], '你好', lambda *_: None)
        self.assertEqual(first['context']['seen'][preference['id']]['content'], '默认用中文')
        self.store.save_memory('默认用英文', 'preference', 'language', memory_id=preference['id'])
        second = agent.run(session['id'], '继续', lambda *_: None)
        self.assertEqual(second['context']['seen'][preference['id']]['content'], '默认用英文')
        self.store.delete_memory(preference['id'])
        third = agent.run(session['id'], '再继续', lambda *_: None)
        self.assertEqual(third['context']['memory_ids'], [])
        self.assertEqual(self.store.memories(), [])

    def test_personal_context_is_deduplicated_against_model_recall(self):
        item = self.store.save_memory('默认用中文', 'preference', 'language')
        result, _, _ = self.run_turn(ScriptedModel(operations=[]), text='你好')
        self.assertEqual(result['context']['memory_ids'], [item['id']])
        self.assertEqual(len(result['answer']['meta']['recalled']), 1)

    def test_personal_context_capacity_is_bounded_and_reports_omissions(self):
        for i in range(30): self.store.save_memory('个人偏好' + str(i) + '。' * 1400, 'preference', f'pref.{i}')
        pinned = self.store.save_memory('重要称呼小林', 'profile', 'name', pinned=True)
        packed = self.store.personal_context()
        self.assertEqual(packed['total'], 31)
        self.assertEqual(packed['memories'][0]['id'], pinned['id'])
        self.assertEqual(packed['omitted'], 31 - len(packed['memories']))
        self.assertLessEqual(len(packed['memories']), 24)
        self.assertLessEqual(len(json.dumps([context_memory(m) for m in packed['memories']], ensure_ascii=False)), 12000)
        for memory in packed['memories']:
            self.assertEqual(memory, next(m for m in self.store.memories() if m['id'] == memory['id']))

    def test_current_instruction_remains_available_over_saved_preference(self):
        self.store.save_memory('默认简洁的中文回复', 'preference', 'style')
        result, _, model = self.run_turn(ScriptedModel(operations=[], direct_context=True), text='这次用英文详细解释')
        call = next(c for c in model.calls if c['stage'] == 'main')['payload']['messages']
        self.assertIn('本轮明确要求优先', call[0]['content'])
        self.assertEqual(call[-1]['content'], '这次用英文详细解释')
        self.assertEqual(result['memory_result']['status'], 'skipped')

    def test_model_can_skip_even_with_remember_keyword(self):
        result, _, _ = self.run_turn(ScriptedModel(operations=[], reason='只是引用的虚构示例，不是用户事实。'), text='示例句：请记住我是一只猫。')
        self.assertEqual(result['memory_result']['status'], 'skipped')
        self.assertEqual(self.store.memories(), [])

    def test_main_has_no_tools_or_write_plan(self):
        _, _, model = self.run_turn()
        call = next(c for c in model.calls if c['stage'] == 'main')['payload']
        self.assertEqual(call['tools'], [])
        self.assertEqual(call['tool_choice'], 'none')
        self.assertNotIn('operations', json.dumps(call, ensure_ascii=False))

    def test_model_calls_actually_overlap_using_barrier(self):
        barrier = threading.Barrier(2)

        def hook(stage, payload):
            if stage in ('main', 'review'):
                barrier.wait(timeout=5)

        result, _, model = self.run_turn(ScriptedModel(hook=hook))
        self.assertFalse(result['answer']['meta'].get('error'))
        self.assertEqual(result['memory_result']['status'], 'changed')
        threads = {c['thread'] for c in model.calls if c['stage'] in ('main', 'review')}
        self.assertEqual(len(threads), 2)

    def test_answer_is_emitted_and_saved_before_memory_finishes(self):
        answered = threading.Event()

        def hook(stage, payload):
            if stage == 'review' and not answered.wait(timeout=5):
                raise RuntimeError('Memory review blocked the answer')

        def event(name, payload):
            if name == 'answer':
                self.assertEqual(payload['message']['meta']['memory']['status'], 'pending')
                self.assertEqual(self.store.memories(), [])
                answered.set()

        result, events, _ = self.run_turn(ScriptedModel(hook=hook), emit=event)
        order = [e[0] for e in events]
        self.assertLess(order.index('answer'), order.index('memory_result'))
        self.assertEqual(result['answer']['meta']['memory']['status'], 'changed')

    def test_memory_failure_does_not_discard_main_answer(self):
        result, _, _ = self.run_turn(ScriptedModel(fail_stage='review'))
        self.assertFalse(result['answer']['meta'].get('error'))
        self.assertEqual(result['memory_result']['status'], 'error')
        self.assertEqual(self.store.memories(), [])

    def test_main_failure_does_not_cancel_memory_branch(self):
        result, _, _ = self.run_turn(ScriptedModel(fail_stage='main'))
        self.assertTrue(result['answer']['meta']['error'])
        self.assertEqual(result['memory_result']['status'], 'changed')
        self.assertEqual(len(self.store.memories()), 1)

    def test_recall_failure_still_runs_both_branches(self):
        result, _, model = self.run_turn(ScriptedModel(fail_stage='context'))
        self.assertFalse(result['context']['ok'])
        self.assertFalse(result['answer']['meta'].get('error'))
        self.assertEqual(result['memory_result']['status'], 'changed')
        self.assertEqual({c['stage'] for c in model.calls}, {'context', 'review', 'main'})

    def test_readonly_permission_is_not_a_semantic_gate(self):
        result, _, model = self.run_turn(ScriptedModel(operations=[]), auto=False)
        self.assertEqual(result['memory_result']['status'], 'readonly')
        self.assertIn('review', [c['stage'] for c in model.calls])
        result, _, _ = self.run_turn(auto=False)
        self.assertEqual(result['memory_result']['status'], 'error')
        self.assertEqual(self.store.memories(), [])

    def test_openai_wire_and_cross_session_handoff(self):
        self.run_turn(provider='openai')
        result, _, model = self.run_turn(ScriptedModel(operations=[]), text='你记得我的偏好吗？', provider='openai')
        self.assertEqual(len(result['context']['memory_ids']), 1)
        main = next(c for c in model.calls if c['stage'] == 'main')['payload']
        self.assertIn('偏好简洁', main['input'][0]['content'])
        self.assertFalse(main['store'])
        followup = [c['payload'] for c in model.calls if c['stage'] == 'context'][1]
        self.assertEqual(followup['input'][-1]['type'], 'function_call_output')
        self.assertEqual(followup['input'][-1]['call_id'], 'context-call')

    def test_deepseek_reasoning_and_call_id_are_kept(self):
        _, _, model = self.run_turn()
        followup = [c['payload'] for c in model.calls if c['stage'] == 'context'][1]
        self.assertEqual(followup['messages'][-1]['tool_call_id'], 'context-call')
        self.assertEqual(followup['messages'][-2]['reasoning_content'], 'opaque-fixture')

    def test_child_can_update_and_delete_seen_memory(self):
        old = self.store.save_memory('用户住在悉尼', 'profile', 'profile.city', pinned=True)
        update = dict(create_op('用户现居墨尔本', 'profile.city'), operation='update', id=old['id'], category='profile')
        result, _, _ = self.run_turn(ScriptedModel(operations=[update]), text='现在搬到墨尔本了')
        self.assertEqual(result['memory_result']['changes'][0]['operation'], 'update')
        self.assertEqual(self.store.memories()[0]['pinned'], 1)
        delete = dict(update, operation='delete', content='', key='')
        result, _, _ = self.run_turn(ScriptedModel(operations=[delete]), text='把我的居住地忘掉吧')
        self.assertEqual(result['memory_result']['changes'][0]['operation'], 'delete')
        self.assertEqual(self.store.memories(), [])

    def test_unknown_id_and_repeated_invalid_plan_have_no_partial_writes(self):
        invalid = dict(create_op(), operation='update', id='does-not-exist')
        result, _, model = self.run_turn(ScriptedModel(operations=[create_op(), invalid]))
        self.assertEqual(result['memory_result']['status'], 'error')
        self.assertEqual(self.store.memories(), [])
        self.assertEqual(len([c for c in model.calls if c['stage'] == 'review']), 6)

    def test_stale_model_plan_cannot_overwrite_manual_edit(self):
        old = self.store.save_memory('旧内容', key='topic')
        self.store.save_memory('人工新内容', key='topic', memory_id=old['id'])
        op = dict(create_op('模型过时内容', 'topic'), operation='update', id=old['id'])
        with self.assertRaises(ValueError): self.store.apply_memory_plan([op], 'source', {old['id']: old})
        self.assertEqual(self.store.memories()[0]['content'], '人工新内容')

    def test_disconnect_does_not_cancel_persistence(self):
        def disconnect(*_): raise BrokenPipeError()
        result, _, _ = self.run_turn(emit=disconnect)
        self.assertEqual(result['answer']['meta']['memory']['status'], 'changed')
        self.assertEqual(len(self.store.memories()), 1)

    def test_existing_database_requires_no_destructive_migration(self):
        session = self.store.create_session()
        self.store.add_message(session['id'], 'user', '旧版本对话')
        self.store.save_memory('旧版本记忆', key='existing')
        reopened = Store(self.store.file)
        self.assertEqual(reopened.messages(session['id'])[0]['content'], '旧版本对话')
        self.assertEqual(reopened.memories()[0]['content'], '旧版本记忆')

    def test_trace_matches_parallel_work_and_persists_exact_events(self):
        barrier = threading.Barrier(2)
        def hook(stage, payload):
            if stage in ('main', 'review'):
                barrier.wait(timeout=5)
        result, events, _ = self.run_turn(ScriptedModel(hook=hook))
        snapshots = [event['trace'] for name, event in events if name == 'flow']
        self.assertTrue(any(s['nodes']['answer']['status'] == s['nodes']['consolidate']['status'] == 'running' for s in snapshots))
        final = snapshots[-1]
        saved = self.store.run(result['answer']['meta']['run_id'])
        self.assertEqual(saved['events'], final['events'])
        self.assertEqual(saved['status'], 'completed')
        self.assertEqual(sum(n['model_calls'] for n in saved['nodes'].values()), 4)
        self.assertEqual(sum(n['tools'] for n in saved['nodes'].values()), 3)
        self.assertEqual(sum(n['tokens'] for n in saved['nodes'].values()), 40)
        self.assertEqual([e['seq'] for e in saved['events']], list(range(1, len(saved['events']) + 1)))
        for node in saved['nodes'].values():
            self.assertEqual(node['status'], 'succeeded')
            self.assertGreaterEqual(node['duration_ms'], 0)
            self.assertEqual(node['duration_ms'], node['ended_ms'] - node['started_ms'])
        for branch in ('answer', 'consolidate'):
            self.assertGreaterEqual(saved['nodes'][branch]['started_ms'], saved['nodes']['recall']['ended_ms'])
            self.assertGreaterEqual(saved['nodes']['finish']['started_ms'], saved['nodes'][branch]['ended_ms'])
        # Patches are sufficient to reconstruct every historical node, without final details leaking backward.
        replay = {key: {} for key in saved['nodes']}
        for event in saved['events']:
            if event.get('patch'):
                replay[event['node']].update(event['patch'])
        for key, node in replay.items():
            self.assertEqual(node['detail'], saved['nodes'][key]['detail'])
        self.assertEqual(snapshots[0]['nodes']['consolidate']['detail'], {})
        self.assertEqual(snapshots[0]['nodes']['consolidate']['status'], 'waiting')

    def test_trace_marks_partial_failure_and_single_model_end(self):
        for stage, node in [('main', 'answer'), ('review', 'consolidate'), ('context', 'recall')]:
            result, _, _ = self.run_turn(ScriptedModel(fail_stage=stage))
            trace = self.store.run(result['answer']['meta']['run_id'])
            self.assertEqual(trace['status'], 'partial')
            self.assertEqual(trace['nodes'][node]['status'], 'failed')
            self.assertEqual(trace['nodes']['finish']['status'], 'succeeded')
            self.assertEqual(sum(e['kind'] == 'model_end' and e['node'] == node for e in trace['events']), 1)

    def test_invalid_main_response_records_one_failed_call_with_usage(self):
        model = ScriptedModel()
        def invalid(url, key, payload):
            response = model(url, key, payload)
            if '[MORI_MAIN]' in payload['messages'][0]['content']:
                response['choices'][0]['message']['content'] = ''
            return response
        result, _, _ = self.run_turn(invalid)
        trace = self.store.run(result['answer']['meta']['run_id'])
        ends = [e for e in trace['events'] if e['kind'] == 'model_end' and e['node'] == 'answer']
        self.assertEqual(len(ends), 1)
        self.assertFalse(ends[0]['ok'])
        self.assertEqual(trace['nodes']['answer']['tokens'], 10)

    def test_trace_exports_and_deletes_with_its_session(self):
        result, _, _ = self.run_turn()
        sid, rid = result['session_id'], result['answer']['meta']['run_id']
        restarted = Store(self.store.file)
        self.assertEqual(restarted.runs(sid)[0]['id'], rid)
        self.assertEqual(restarted.export()['sessions'][0]['runs'][0]['id'], rid)
        restarted.delete_session(sid)
        self.assertEqual(restarted.runs(), [])
        with self.assertRaises(ValueError): restarted.run(rid)
        self.assertEqual(len(restarted.memories()), 1)

    def test_startup_recovers_interrupted_run_without_fabricating_completion(self):
        session = self.store.create_session()
        trace = RunTrace(self.store, session['id'], 'input', 'test-model', 'deepseek', lambda *_: None)
        trace.begin('prepare')
        trace.end('prepare')
        trace.begin('recall')
        self.store.recover_runs()
        saved = self.store.run(trace.data['id'])
        self.assertEqual(saved['status'], 'interrupted')
        self.assertEqual(saved['nodes']['prepare']['status'], 'succeeded')
        self.assertEqual(saved['nodes']['recall']['status'], 'interrupted')
        self.assertIsNone(saved['nodes']['recall']['ended_ms'])
        self.assertIsNone(saved['nodes']['recall']['duration_ms'])
        self.assertEqual(saved['nodes']['answer']['status'], 'cancelled')
        self.store.recover_runs()
        self.assertEqual(self.store.run(trace.data['id']), saved)

    def test_trace_never_stores_provider_payloads_keys_or_hidden_reasoning(self):
        result, _, _ = self.run_turn()
        raw = json.dumps(self.store.run(result['answer']['meta']['run_id']), ensure_ascii=False)
        for secret in ('test-key', 'opaque-fixture', 'reasoning_content', '[MORI_MAIN]', '[MORI_MEMORY_CONTEXT]'):
            self.assertNotIn(secret, raw)


if __name__ == '__main__':
    unittest.main()
