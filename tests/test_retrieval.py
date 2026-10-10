"""Retrieval contracts using deterministic vectors, not model-quality scores."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from agent.core import Agent
from agent.providers import Provider, ProviderError
from agent.routing import router_context
from agent.semantic import EmbeddingProvider, HybridRetriever, embedding_options, fingerprint
from agent.store import Store
from model_fixture import ScriptedModel
from server import App
from tools.memory_lab_probe import ProbeModel


class VectorFixture:
    def __init__(self):
        self.calls = []
        self.fail = False
        self.hook = None

    def __call__(self, url, key, payload):
        self.calls.append(payload)
        if self.fail:
            raise ProviderError('模拟 Embedding 不可用')
        if self.hook:
            self.hook()
        vectors = []
        for index, text in enumerate(payload['input']):
            # Fixture meaning labels test fusion, never production rules.
            vector = [1, 0, 0] if 'coffee' in text or 'stimulant' in text else [0, 1, 0] if 'swim' in text else [0, 0, 1]
            vectors.append(dict(index=index, embedding=vector))
        return dict(data=list(reversed(vectors)), usage={'total_tokens': len(vectors)*5})


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.temp.name)/'mori.sqlite3')
        self.config = embedding_options(dict(enabled=True, dimensions=3, batch_size=2))
        self.fixture = VectorFixture()
        self.provider = EmbeddingProvider(self.config, 'fixture-only-key', self.fixture)
        self.retriever = HybridRetriever(self.store, lambda: EmbeddingProvider(self.config, 'fixture-only-key', self.fixture), lambda: self.config)

    def tearDown(self):
        if self.retriever._worker:
            self.retriever._worker.join(3)
        self.temp.cleanup()

    def index(self):
        self.retriever.schedule(force=True)
        self.retriever._worker.join(3)
        self.assertEqual(self.retriever.status()['state'], 'complete')

    def test_semantic_only_candidate_and_repeated_query_cache(self):
        coffee = self.store.save_memory('coffee before noon', 'note', 'drink')
        self.store.save_memory('swim on Wednesday', 'note', 'exercise')
        self.index()
        self.assertFalse(self.store.search('stimulant cutoff'))
        rows, detail = self.retriever.search('stimulant cutoff')
        self.assertEqual(rows[0]['id'], coffee['id'])
        self.assertEqual(detail['mode'], 'hybrid')
        self.assertEqual(detail['lexical_count'], 0)
        self.assertEqual(detail['semantic_count'], 1)
        self.assertEqual(detail['embedding_tokens'], 5)
        count = len(self.fixture.calls)
        again, second = self.retriever.search('stimulant cutoff')
        self.assertEqual(again, rows)
        self.assertTrue(second['query_cache_hit'])
        self.assertEqual(len(self.fixture.calls), count)
        self.assertEqual(second['embedding_tokens'], 0)

    def test_index_survives_restart_and_unchanged_documents_are_not_reembedded(self):
        for i in range(5):
            self.store.save_memory('coffee '+str(i), 'note', 'drink.'+str(i))
        self.index()
        self.assertEqual(len(self.fixture.calls), 3)
        reopened = Store(self.store.file)
        service = HybridRetriever(reopened, lambda: self.provider, lambda: self.config)
        service.schedule(); service._worker.join(3)
        self.assertEqual(len(self.fixture.calls), 3)
        self.assertEqual(service.search('stimulant')[1]['indexed'], 5)

    def test_direct_edit_invalidates_stale_vector_and_delete_cascades(self):
        item = self.store.save_memory('coffee daily', 'note', 'drink')
        self.index()
        with self.store.connect() as db:
            db.execute('UPDATE memories SET content=? WHERE id=?', ('swim daily', item['id']))
        self.assertEqual(self.store.vector_status(fingerprint(self.config))['pending'], 1)
        self.assertEqual(self.retriever._snapshot(fingerprint(self.config))[0], [])
        self.index()
        self.assertEqual(len(self.fixture.calls), 2)
        self.assertFalse(self.retriever.search('stimulant')[0])
        self.store.delete_memory(item['id'])
        self.assertEqual(self.retriever.search('swim')[0], [])
        with self.store.connect() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM memory_vectors').fetchone()[0], 0)

    def test_edit_during_api_request_does_not_attach_old_vector(self):
        item = self.store.save_memory('coffee daily', 'note', 'drink')
        docs = self.store.pending_vectors(fingerprint(self.config))
        vectors, _ = self.provider.embed([d['document'] for d in docs])
        self.store.save_memory('swim daily', 'note', 'drink', memory_id=item['id'])
        self.store.save_vectors(fingerprint(self.config), docs, vectors)
        self.assertEqual(self.store.vector_status(fingerprint(self.config))['indexed'], 0)

    def test_query_failure_keeps_keyword_results_and_records_fallback(self):
        self.store.save_memory('coffee daily', 'note', 'drink')
        self.index()
        self.fixture.fail = True
        rows, detail = self.retriever.search('coffee')
        self.assertEqual(rows[0]['key'], 'drink')
        self.assertEqual(detail['mode'], 'keyword')
        self.assertIn('模拟', detail['fallback'])

    def test_child_hybrid_search_is_handed_to_main_and_recorded_in_both_branches(self):
        coffee = self.store.save_memory('coffee before noon', 'note', 'drink')
        self.index()
        model = ProbeModel(query='stimulant cutoff')
        chat = Provider('deepseek', 'http://127.0.0.1:9876', 'fixture', 'fake', model)
        session = self.store.create_session()
        result = Agent(self.store, chat, retriever=self.retriever).run(session['id'], 'A fixture question', lambda *_: None)
        self.assertIn(coffee['id'], result['context']['memory_ids'])
        trace = self.store.run(result['answer']['meta']['run_id'])
        for node in ['recall', 'consolidate']:
            read = next(r for r in trace['nodes'][node]['detail']['retrievals'] if r['name'] == 'memory_search')
            detail = read['retrieval']
            self.assertEqual(detail['mode'], 'hybrid')
            self.assertEqual(detail['semantic_count'], 1)
        self.assertTrue(trace['nodes']['consolidate']['detail']['retrievals'][0]['retrieval']['query_cache_hit'])
        prompt = next(c['prompt'] for c in model.calls if c['stage'] == 'answer')
        self.assertIn('coffee before noon', prompt)

    def test_disabled_embedding_never_makes_api_calls(self):
        self.config['enabled'] = False
        self.store.save_memory('coffee daily', 'note', 'drink')
        self.retriever.schedule()
        rows, detail = self.retriever.search('coffee')
        self.assertEqual(detail['mode'], 'keyword')
        self.assertEqual(len(rows), 1)
        self.assertFalse(self.fixture.calls)

    def test_new_model_or_dimensions_require_new_document_vectors(self):
        self.store.save_memory('coffee daily', 'note', 'drink')
        self.index()
        old_identity = fingerprint(self.config)
        self.config['model'] = 'other-embedding-model'
        self.assertNotEqual(fingerprint(self.config), old_identity)
        self.assertEqual(self.retriever.status()['indexed'], 0)
        self.index()
        self.assertEqual(len(self.fixture.calls), 2)
        self.config['dimensions'] = 4
        self.assertEqual(self.retriever.status()['pending'], 1)

    def test_config_change_between_document_selection_and_adapter_does_not_mislabel_vectors(self):
        self.store.save_memory('coffee daily', 'note', 'drink')
        changed = False
        def provider_factory():
            nonlocal changed
            if not changed:
                self.config['model'] = 'new-model'
                changed = True
            return EmbeddingProvider(self.config, 'fake', self.fixture)
        self.retriever.provider_factory = provider_factory
        self.index()
        self.assertEqual(len(self.fixture.calls), 1)
        self.assertEqual(self.fixture.calls[0]['model'], 'new-model')
        self.assertEqual(self.store.vector_status(fingerprint(self.config))['indexed'], 1)

    def test_bounded_rrf_returns_full_rows_without_ranking_metadata(self):
        for i in range(40):
            self.store.save_memory('coffee item '+str(i), 'note', 'drink.'+str(i))
        self.index()
        rows, detail = self.retriever.search('coffee', 12)
        self.assertEqual(len(rows), 12)
        self.assertEqual(detail['semantic_count'], 24)
        self.assertEqual(detail['fusion'], 'RRF')
        current = self.store.memories_by_ids([r['id'] for r in rows])
        self.assertTrue(all(current[r['id']] == r for r in rows))

    def test_embedding_adapter_validates_indices_values_and_dimensions(self):
        vectors, usage = self.provider.embed(['coffee', 'swim'])
        self.assertEqual(vectors[0].tolist(), [1,0,0])
        self.assertEqual(usage, 10)
        self.assertEqual(self.fixture.calls[0]['encoding_format'], 'float')
        self.assertEqual(self.fixture.calls[0]['dimensions'], 3)
        for data in [dict(data=[]), dict(data=[dict(index=0, embedding=[0,0,0])]),
                     dict(data=[dict(index=0, embedding=[1,2])]), dict(data=[dict(index=0, embedding=[1,float('nan'),0])]),
                     dict(data=[dict(index=1, embedding=[1,0,0])]), dict(data=[dict(index=0, embedding=[True,0,0])])]:
            with self.subTest(data=data), self.assertRaises(ProviderError):
                EmbeddingProvider(self.config, 'fake', lambda *_: data).embed(['coffee'])
        with self.assertRaises(ValueError):
            self.provider.embed(['字'*2500])


class RoutingAndConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.app = App(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def configure(self, **extra):
        return self.app.configure(dict(provider='openai', base_url='http://127.0.0.1:9876', model='test-model', **extra))

    def test_reused_key_requires_same_endpoint_and_separate_key_survives_restart(self):
        self.configure(api_key='chat-private')
        self.configure(embedding=dict(base_url='http://127.0.0.1:9876', key_source='openai'))
        self.assertEqual(self.app.embedding_key(), 'chat-private')
        before = dict(self.app.settings)
        with self.assertRaises(ValueError):
            self.configure(embedding=dict(base_url='https://other.example/v1', key_source='openai'))
        self.assertEqual(self.app.settings, before)
        self.configure(embedding=dict(base_url='https://other.example/v1', key_source='separate', api_key='embedding-private'))
        reopened = App(self.temp.name)
        self.assertEqual(reopened.embedding_key(), 'embedding-private')
        self.assertNotIn('embedding-private', json.dumps(reopened.public_settings()))
        self.assertNotIn('embedding-private', json.dumps(reopened.store.export()))
        self.configure(embedding=dict(clear_key=True))
        self.assertFalse(App(self.temp.name).embedding_key())

    def test_invalid_embedding_settings_preserve_chat_key_and_settings(self):
        self.configure(api_key='original-private')
        before = json.dumps(self.app.settings, sort_keys=True)
        for cfg in [dict(dimensions=True), dict(min_similarity=float('nan')), dict(batch_size=100), dict(extra=1)]:
            with self.subTest(config=cfg), self.assertRaises(ValueError):
                self.configure(api_key='replacement-private', embedding=cfg)
            self.assertEqual(json.dumps(self.app.settings, sort_keys=True), before)
            self.assertEqual(self.app.keys['openai'], 'original-private')

    def test_daily_route_preserves_home_context_through_child_handoff(self):
        self.app.store.save_memory('I prefer quiet evenings at home', 'note', 'home.quiet')
        model = ScriptedModel(operations=[])
        provider = Provider('deepseek', 'http://127.0.0.1:9876', 'fixture', 'fake', model)
        session = self.app.store.create_session()
        supplied = dict(route='daily', context={'home': {'source': 'router-test', 'note': 'test data only'}})
        result = Agent(self.app.store, provider).run(session['id'], 'Plan my evening', lambda *_: None, supplied)
        self.assertEqual(result['context']['router_context'], supplied)
        self.assertTrue(any(r['key'] == 'home.quiet' for r in result['context']['seen'].values()))
        for stage in ['context', 'review', 'main']:
            system = next(c['payload']['messages'][0]['content'] for c in model.calls if c['stage'] == stage)
            self.assertIn('router-test', system)
        trace = self.app.store.run(result['answer']['meta']['run_id'])
        self.assertEqual(trace['nodes']['prepare']['detail']['router_context'], supplied)
        self.assertEqual(self.app.store.messages(session['id'])[0]['meta']['router_context'], supplied)

    def test_invalid_router_data_is_rejected_before_saving_message(self):
        session = self.app.store.create_session()
        agent = Agent(self.app.store, None)
        for data in [dict(route='unknown'), dict(context=[]), dict(context={'long':'x'*8000}), dict(context={'x':float('nan')})]:
            with self.subTest(data=data), self.assertRaises(ValueError):
                agent.run(session['id'], 'hello', lambda *_: None, data)
        self.assertEqual(self.app.store.messages(session['id']), [])
        self.assertEqual(router_context(), dict(route='daily', context={}))
