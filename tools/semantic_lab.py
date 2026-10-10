"""Opt-in real Embedding evaluation on synthetic data in a temporary DB.

Reads a saved key without displaying it. Does not modify the source workspace,
its memories, credentials, or model settings. No chat-model calls are made.
"""
import argparse
import json
import sqlite3
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.semantic import EmbeddingProvider, HybridRetriever, embedding_options
from agent.store import Store, now
from tools.memory_lab import benchmark, ms, quality, stats
from tools.memory_lab_data import applicable_cases, dataset, seed


def evaluate(size, credential_dir):
    database = credential_dir / 'mori.sqlite3'
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True) as db:
        row = db.execute("SELECT base_url,api_key FROM credentials WHERE provider='openai'").fetchone()
    if not row or not row[1]:
        raise ValueError('该工作空间没有已保存的 OpenAI Key')
    cfg = embedding_options(dict(enabled=True, base_url=row[0]))
    provider = EmbeddingProvider(cfg, row[1])
    rows = dataset(size)
    cases = applicable_cases(rows)
    with tempfile.TemporaryDirectory(prefix='mori-semantic-lab-') as directory:
        store = Store(Path(directory)/'synthetic.sqlite3')
        seed(store, rows)
        service = HybridRetriever(store, lambda: provider, lambda: cfg)
        started = time.perf_counter()
        service.schedule(force=True)
        while service._worker.is_alive():
            service._worker.join(1)
            status = service.status()
            if status['processed'] and status['processed'] % 64 == 0:
                print(json.dumps(dict(indexed=status['indexed'], total=status['total'])), flush=True)
        index_ms = round((time.perf_counter()-started)*1000, 1)
        status = service.status()
        if status['state'] != 'complete' or status['indexed'] != size:
            raise ValueError(status['error'] or '索引未完成')
        results, cold, warm, api = [], [], [], []
        for c in cases:
            (found, detail), elapsed = ms(lambda: service.search(c['query']))
            (cached, second), cached_ms = ms(lambda: service.search(c['query']))
            if detail['mode'] != 'hybrid' or not second['query_cache_hit']:
                raise ValueError(detail['fallback'] or '缓存或混合召回未生效')
            cold.append(elapsed); warm.append(cached_ms); api.append(detail['embedding_ms'])
            expected = set(c['expected_keys'])
            returned = [r['key'] for r in found]
            rank = next((i+1 for i,k in enumerate(returned) if k in expected), None)
            results.append(dict(c, returned_keys=returned, retrieval=detail,
                recall_at_5=len(expected.intersection(returned[:5]))/len(expected) if expected else None,
                recall_at_12=len(expected.intersection(returned))/len(expected) if expected else None,
                reciprocal_rank=1/rank if rank else 0))
            print(json.dumps(dict(case=c['name'], hits=len(expected.intersection(returned)),
                                 expected=len(expected), embedding_ms=detail['embedding_ms']), ensure_ascii=False), flush=True)
        return dict(size=size, config=cfg, index_ms=index_ms, index_tokens=status['tokens'],
            query_tokens=sum(c['retrieval']['embedding_tokens'] for c in results),
            cold_query=stats(cold), cached_query=stats(warm), embedding_api=stats(api),
            quality=quality(results, cases), cases=results,
            real_embedding_api=True, chat_model_quality_evaluated=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--real-api', action='store_true', help='Send synthetic facts and queries to the saved key endpoint')
    parser.add_argument('--key-data-dir', type=Path, default=Path('artifacts/memory-lab/sandbox'))
    parser.add_argument('--size', type=int, default=500)
    parser.add_argument('--output', type=Path, default=Path('artifacts/retrieval-v2.6'))
    args = parser.parse_args()
    if not args.real_api or not 10 <= args.size <= 10000:
        parser.error('需要 --real-api；size 为 10–10000')
    local = [benchmark(n, 4) for n in (500, 2000)]
    real = evaluate(args.size, args.key_data_dir)
    report = dict(created_at=now(), synthetic=True, source_database_modified=False, local=local, semantic=real)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    text = ['# Mori v2.6 检索实测\n\n所有事实、人物和问题均为虚构。临时数据库与个人/现有测试数据库隔离。只调用真实 Embedding API，不调用聊天模型。\n\n',
            '| 记忆量 | v2.5 快照缓存中位数 / P95 | v2.6 FTS5 中位数 / P95 | 已有索引重启后首次查询 | 首次索引同步 |\n|---|---|---|---|---|\n']
    for b in local:
        p, c = b['previous_cached'], b['current']
        text.append(f"| {b['size']} | {p['median_ms']} / {p['p95_ms']} ms | {c['median_ms']} / {c['p95_ms']} ms | {b['cold_search_ms']} ms | {b['initial_sync_ms']} ms |\n")
    text.append(f"\n真实模型 `{real['config']['model']}`，{real['config']['dimensions']} 维，{args.size} 条事实建立向量索引耗时 {real['index_ms']/1000:.1f} 秒，输入 {real['index_tokens']:,} Token。{len(real['cases'])} 个查询消耗 {real['query_tokens']:,} Token。\n\n")
    for label, key in [('首次查询（含联网）','cold_query'),('重复查询（缓存）','cached_query'),('单次 Embedding API','embedding_api')]:
        s=real[key]; text.append(f"- {label}：中位数 {s['median_ms']} ms，P95 {s['p95_ms']} ms。\n")
    text.append('\n| 问题类型 | FTS5 Recall@12 | 混合 Recall@12 |\n|---|---|---|\n')
    for kind, value in real['quality']['by_kind'].items():
        old=local[0]['quality']['by_kind'][kind]
        text.append(f"| {kind} | {old['recall_at_12']:.1%} | {value['recall_at_12']:.1%} |\n")
    text.append('\nRecall@12 是人工标注目标在候选中的命中率，不是最终回答准确率或记忆决策质量。改写题仅三题，不能代表所有表述。无相关记忆也可能返回向量候选，仍须子 Agent 筛选。API 耗时受网络与服务商影响；缓存只在当前进程有效。\n\n| 问题 | 标注目标 | 前五个候选 |\n|---|---|---|\n')
    for c in real['cases']:
        text.append(f"| {c['query']} | {', '.join(c['expected_keys']) or '无'} | {', '.join(c['returned_keys'][:5]) or '无'} |\n")
    (args.output/'report.md').write_text(''.join(text), encoding='utf-8')
    print(json.dumps(dict(report=str(args.output.resolve()), quality=real['quality'], cold=real['cold_query'], warm=real['cached_query']), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
