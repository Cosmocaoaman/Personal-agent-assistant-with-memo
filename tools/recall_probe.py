"""Explicit real-API comparison using 500 synthetic memories, never source writes."""
import argparse
import json
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.core import Agent
from agent.providers import Provider
from agent.semantic import EmbeddingProvider, HybridRetriever, embedding_options, fingerprint
from agent.store import Store
from agent.transport import PooledHTTP
from tools.memory_lab_data import dataset, seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--real-api', action='store_true')
    parser.add_argument('--key-data-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.real_api: parser.error('Real API access requires --real-api.')
    db = (args.key_data_dir / 'mori.sqlite3').resolve()
    source = sqlite3.connect(db.as_uri() + '?mode=ro', uri=True)
    source.row_factory = sqlite3.Row
    settings = {r['key']: json.loads(r['value']) for r in source.execute('SELECT key,value FROM settings')}
    config = settings['openai']
    credential = source.execute("SELECT base_url,api_key FROM credentials WHERE provider='openai'").fetchone()
    if not credential or credential['base_url'] != config['base_url'] or not credential['api_key']:
        raise SystemExit('No matching saved OpenAI credential.')
    cfg = embedding_options(settings.get('embedding'))
    if cfg['key_source'] != 'openai' or cfg['base_url'] != config['base_url']:
        raise SystemExit('Probe requires Embedding configured to reuse the saved OpenAI endpoint.')
    vectors = [dict(r) for r in source.execute('SELECT * FROM memory_vectors WHERE fingerprint=?', (fingerprint(cfg),))]
    source.close()
    # Source conversations, settings, keys and modified memories are never copied.
    rows = dataset(500)
    documents = {r['id']: r['key'] + '\n' + r['content'] for r in rows}
    vectors = [v for v in vectors if documents.get(v['memory_id']) == v['document']]
    cases = [
        ('general', '用两句话说明 Python 列表与元组的区别。', [], []),
        ('project', '我之前的 Aurora 项目后端、数据库和缓存分别是什么？最多两句话。', ['work.aurora'], ['FastAPI', 'PostgreSQL', 'Redis']),
        ('daily_home', '我家的路由器放在哪里，书房电脑如何接网？最多两句话。', ['home.router'], ['书架', '有线']),
    ]
    modes = [('specialist_no_prefetch', {'recall_mode': 'specialist', 'prefetch_candidates': False}),
             ('specialist_prefetch', {'recall_mode': 'specialist', 'prefetch_candidates': True}),
             ('main', {'recall_mode': 'main'})]
    report = dict(model=config['model'], synthetic=True, count=500, indexed=len(vectors),
        limitation='Three short questions, one run per mode; not a statistical speed/quality guarantee. The no-prefetch control uses the same shortened handoff prompt. Read-only maintenance still makes independent model calls.', measurements=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pool = PooledHTTP()
    try:
        with tempfile.TemporaryDirectory(prefix='mori-recall-probe-') as root:
            store = Store(Path(root) / 'probe.sqlite3')
            seed(store, rows)
            with store.connect() as target:
                target.executemany('INSERT INTO memory_vectors VALUES (:memory_id,:fingerprint,:document,:dimensions,:vector)', vectors)
            refreshed, index_tokens = 0, 0
            embedding = EmbeddingProvider(cfg, credential['api_key'], pool)
            while pending := store.pending_vectors(fingerprint(cfg), cfg['batch_size']):
                new_vectors, tokens = embedding.embed([d['document'] for d in pending])
                store.save_vectors(fingerprint(cfg), pending, new_vectors)
                refreshed += len(pending)
                index_tokens += tokens
            report.update(indexed=store.vector_status(fingerprint(cfg))['indexed'], refreshed_synthetic_vectors=refreshed, index_tokens=index_tokens)
            client = Provider('openai', config['base_url'], config['model'], credential['api_key'], pool,
                              options=dict(config.get('options', {}), max_output_tokens=1024, web_search=False))
            for mode, runtime in modes:
                retriever = HybridRetriever(store, lambda: EmbeddingProvider(cfg, credential['api_key'], pool), lambda: cfg)
                for name, question, expected_keys, markers in cases:
                    events = []
                    result = Agent(store, client, auto_memory=False, runtime=runtime, retriever=retriever).run(
                        store.create_session('虚构检索测试')['id'], question,
                        lambda event, data: events.append(dict(event=event, elapsed_ms=data['elapsed_ms'])),
                        {'route': 'daily', 'context': {}})
                    trace = store.run(result['answer']['meta']['run_id'])
                    last_reset = max((i for i, e in enumerate(events) if e['event'] == 'answer_reset'), default=-1)
                    first = next((e['elapsed_ms'] for e in events if e['event'] == 'answer_delta'), None)
                    final_first = next((e['elapsed_ms'] for e in events[last_reset + 1:] if e['event'] == 'answer_delta'), None)
                    received_keys = [r['key'] for r in result['answer']['meta']['recalled']]
                    row = dict(mode=mode, case=name, question=question, status=trace['status'],
                        first_visible_ms=first, first_answer_ms=final_first,
                        answer_ms=next((e['elapsed_ms'] for e in events if e['event'] == 'answer'), None),
                        done_ms=trace['elapsed_ms'], answer=result['answer']['content'],
                        expected_keys=expected_keys, target_ids_received=all(k in received_keys for k in expected_keys),
                        answer_markers_present=all(m.lower() in result['answer']['content'].lower() for m in markers),
                        received_count=len(received_keys), summary_chars=len(result['context']['summary']),
                        nodes={name: {key: node.get(key) for key in ('duration_ms', 'model_calls', 'tokens', 'first_output_ms', 'requests', 'detail')}
                               for name, node in trace['nodes'].items() if name in ('recall', 'answer', 'consolidate')})
                    report['measurements'].append(row)
                    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
                    print(json.dumps({k: row[k] for k in ('mode','case','status','first_answer_ms','answer_ms','target_ids_received','answer_markers_present')}, ensure_ascii=True), flush=True)
    finally:
        pool.close()


if __name__ == '__main__': main()
