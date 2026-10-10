"""Large-corpus, cache freshness and context capacity acceptance checks.

Model responses are scripted. These tests do not measure AI decision quality.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.config import runtime_options
from agent.core import Agent
from agent.memory_agent import MemoryAgent
from agent.providers import Provider
from agent.store import Store, context_memory, pack_memories
from tools.memory_lab import legacy_search
from tools.memory_lab_data import SOURCE, applicable_cases, dataset, seed
from tools.memory_lab_probe import ProbeModel, probe


class ScaleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.file = Path(self.temp.name) / 'scale.sqlite3'
        self.store = Store(self.file)

    def tearDown(self):
        self.temp.cleanup()

    def run_model(self, model, runtime=None):
        client = Provider('deepseek', 'http://127.0.0.1:9876', 'scripted-test', 'not-a-key', model)
        session = self.store.create_session('容量测试')
        return Agent(self.store, client, runtime=runtime).run(session['id'], '这是虚构测试消息。', lambda *_: None)

    def test_500_domains_use_four_categories_and_seed_never_replaces_edits(self):
        rows = dataset(500)
        self.assertEqual(len(rows), 500)
        self.assertGreaterEqual(len({r['domain'] for r in rows}), 15)
        self.assertEqual({r['category'] for r in rows}, {'profile', 'preference', 'project', 'note'})
        self.assertTrue(seed(self.store, rows))
        old = self.store.memories()[0]
        self.store.save_memory('在测试界面手动修改的事实', old['category'], old['key'], memory_id=old['id'])
        self.assertFalse(seed(self.store, rows))
        self.assertEqual(len(self.store.memories()), 500)
        self.assertTrue(any(r['content'] == '在测试界面手动修改的事实' for r in self.store.memories()))

    def test_fts5_recalls_labelled_topics_without_reading_full_memory_snapshot(self):
        rows = dataset(500)
        seed(self.store, rows)
        with patch.object(self.store, 'memories', side_effect=AssertionError('full snapshot read')):
            for case in applicable_cases(rows):
                if case['kind'] not in ('keyword', 'multi_domain', 'temporal'):
                    continue
                with self.subTest(query=case['query']):
                    found = {r['key'] for r in self.store.search(case['query'], 12)}
                    self.assertTrue(set(case['expected_keys']) <= found)

    def test_warm_index_is_reused_and_fresh_after_other_store_edits(self):
        seed(self.store, dataset(100))
        self.store.search('Python')
        with self.store.connect() as db:
            previous = db.execute('SELECT rowid,terms FROM memory_fts ORDER BY rowid').fetchall()
        reopened = Store(self.file)
        reopened.search('京都')
        with reopened.connect() as db:
            self.assertEqual([tuple(r) for r in previous], [tuple(r) for r in db.execute('SELECT rowid,terms FROM memory_fts ORDER BY rowid')])
            self.assertEqual(db.execute('SELECT COUNT(*) FROM search_changes').fetchone()[0], 0)
        other = Store(self.file)
        old = next(r for r in other.memories() if r['key'] == 'dev.python')
        other.save_memory('改为 Scrapy 同步采集', 'project', old['key'], memory_id=old['id'])
        self.assertEqual(self.store.search('Scrapy')[0]['id'], old['id'])
        self.assertFalse(any(r['id'] == old['id'] for r in self.store.search('aiohttp')))
        other.delete_memory(old['id'])
        self.assertFalse(self.store.search('Scrapy'))

    def test_direct_sql_edit_with_same_timestamp_invalidates_index(self):
        item = self.store.save_memory('Python 采集', 'project', 'collector')
        self.store.search('Python')
        with self.store.connect() as db:
            db.execute('UPDATE memories SET content=? WHERE id=?', ('Rust 采集', item['id']))
        self.assertEqual(self.store.search('Rust')[0]['content'], 'Rust 采集')
        self.assertEqual(self.store.search('Python'), [])

    def test_callers_cannot_mutate_cached_results(self):
        self.store.save_memory('独立事实 Python', 'project', 'python')
        result = self.store.search('Python')
        result[0]['content'] = '被调用方改坏'
        self.assertEqual(self.store.search('Python')[0]['content'], '独立事实 Python')

    def test_fts_query_operators_are_data_and_punctuation_still_matches(self):
        item = self.store.save_memory('Python 和 🙂 以及 C++', 'note', 'syntax')
        self.assertEqual(self.store.search('🙂')[0]['id'], item['id'])
        for query in ['" OR NOT : * ( )', 'NEAR(foo bar)', 'x AND y', "' OR 1=1 --"]:
            self.store.search(query)

    def test_hundreds_of_preferences_do_not_all_preload(self):
        seed(self.store, dataset(500))
        personal = self.store.personal_context()
        self.assertEqual(len(personal['memories']), 24)
        self.assertGreater(personal['omitted'], 100)
        self.assertTrue({r['id'] for r in dataset(500) if r['pinned']} <= {r['id'] for r in personal['memories']})
        self.assertLessEqual(len(json.dumps([context_memory(r) for r in personal['memories']], ensure_ascii=False)), 12000)

    def test_tool_returns_compact_rows_but_keeps_versions_in_python(self):
        item = self.store.save_memory('低冲击运动 游泳', 'preference', 'exercise', evidence='证据内容只用于事务验证')
        child = MemoryAgent(self.store, None, lambda *_: None)
        seen = {}
        result = child._read('memory_search', {'query': '游泳'}, seen)
        self.assertEqual(result['memories'][0], context_memory(item))
        self.assertNotIn('source', result['memories'][0])
        self.assertNotIn('evidence', result['memories'][0])
        self.assertEqual(seen[item['id']], item)

    def test_capacity_returns_only_whole_rows_and_does_not_mark_omitted_ids_seen(self):
        rows = [self.store.save_memory('内容' + str(i) + '。' * 1300, 'note', f'long.{i}') for i in range(5)]
        child = MemoryAgent(self.store, None, lambda *_: None)
        seen = {}
        result = child._read('memory_list', {'offset': 0}, seen, remaining=2000)
        self.assertEqual(len(result['memories']), 1)
        self.assertEqual(result['next_offset'], 1)
        self.assertEqual(result['omitted_due_to_budget'], 4)
        self.assertEqual(len(seen), 1)
        returned = result['memories'][0]
        self.assertEqual(returned['content'], seen[returned['id']]['content'])
        self.assertLessEqual(result['memory_json_chars'], 2000)
        second = child._read('memory_list', {'offset': 1}, seen, remaining=2000)
        self.assertNotEqual(second['memories'][0]['id'], returned['id'])
        exhausted = child._read('memory_list', {'offset': 2}, seen, remaining=1)
        self.assertEqual(exhausted['memories'], [])
        self.assertEqual(len(seen), 2)

    def test_continuous_browsing_stays_bounded_even_if_model_ignores_notice(self):
        result = probe(500, 'browse')
        self.assertTrue(result['context_ok'])
        self.assertLessEqual(result['context_read_chars'], 24000)
        self.assertLessEqual(result['review_read_chars'], 24000)
        self.assertLessEqual(result['handoff_memory_json_chars'], 16000)
        self.assertLess(result['phases']['recall']['max_input_json_chars'], 50000)
        self.assertLess(result['phases']['review']['max_input_json_chars'], 50000)
        self.assertLess(result['seen_memories'], 150)

    def test_main_handoff_budget_does_not_slice_long_facts(self):
        for i in range(16):
            self.store.save_memory('sharedword 条目 ' + str(i) + '。' * 1100, 'note', f'long.{i}')
        result = self.run_model(ProbeModel(query='sharedword'), runtime={'personal_items': 0, 'handoff_chars': 2000})
        context = result['context']
        self.assertTrue(context['ok'])
        self.assertEqual(len(context['memory_ids']), 1)
        self.assertEqual(context['handoff_omitted'], 11)
        self.assertLessEqual(context['handoff_chars'], 2000)
        actual = context['seen'][context['memory_ids'][0]]
        self.assertGreater(len(actual['content']), 1100)
        self.assertEqual(result['answer']['meta']['recalled'][0]['id'], actual['id'])

    def test_small_read_budget_also_limits_initial_personal_context(self):
        for i in range(30):
            self.store.save_memory('个人偏好' + str(i) + '。' * 1100, 'preference', f'pref.{i}')
        result = self.run_model(ProbeModel(query='个人偏好'), runtime={'memory_read_chars': 2000, 'handoff_chars': 1000})
        self.assertLessEqual(result['context']['read_chars'], 2000)
        self.assertLessEqual(result['context']['handoff_chars'], 1000)
        self.assertGreater(result['context']['personal']['omitted'], 0)
        self.assertEqual(result['context']['personal']['memory_ids'], [])

    def test_capacity_settings_validate_and_reach_trace_and_both_phases(self):
        for supplied in ({'memory_read_chars': 1999}, {'handoff_chars': 999}, {'memory_read_chars': True}, {'handoff_chars': 50001}):
            with self.subTest(supplied=supplied), self.assertRaises(ValueError):
                runtime_options(supplied)
        seed(self.store, dataset(100))
        result = self.run_model(ProbeModel(), runtime={'memory_read_chars': 4000, 'handoff_chars': 2000})
        trace = self.store.run(result['answer']['meta']['run_id'])
        self.assertEqual(trace['configuration']['runtime']['memory_read_chars'], 4000)
        self.assertEqual(trace['configuration']['runtime']['handoff_chars'], 2000)
        self.assertLessEqual(trace['nodes']['recall']['detail']['read_chars'], 4000)
        self.assertLessEqual(trace['nodes']['consolidate']['detail']['read_chars'], 4000)
        self.assertGreater(len(trace['nodes']['recall']['detail']['retrievals']), 0)

    def test_large_corpus_updates_creates_deletes_and_skips_without_changing_other_facts(self):
        seed(self.store, dataset(500))
        before = {r['id']: r for r in self.store.memories()}
        old = next(r for r in before.values() if r['key'] == 'food.coffee')
        op = dict(operation='update', id=old['id'], key=old['key'], content='小林下午两点以后不喝咖啡。',
                  category='preference', evidence='模拟用户的新表述')
        result = self.run_model(ProbeModel(query='咖啡', operations=[op]))
        self.assertEqual(result['memory_result']['changes'][0]['id'], old['id'])
        self.assertEqual(len(self.store.memories()), 500)
        op = dict(operation='create', id='', key='study.spanish', content='小林开始学习西班牙语。', category='project', evidence='模拟表述')
        result = self.run_model(ProbeModel(query='西班牙语', operations=[op]))
        self.assertEqual(len(self.store.memories()), 501)
        result = self.run_model(ProbeModel(query='咖啡', operations=[]))
        self.assertEqual(result['memory_result']['status'], 'skipped')
        op = dict(operation='delete', id=old['id'], key='', content='', category='preference', evidence='模拟忘记请求')
        self.run_model(ProbeModel(query='咖啡', operations=[op]))
        after = {r['id']: r for r in self.store.memories()}
        self.assertEqual(len(after), 500)
        self.assertNotIn(old['id'], after)
        for mid in before.keys() - {old['id']}:
            self.assertEqual(after[mid], before[mid])


if __name__ == '__main__':
    unittest.main()
