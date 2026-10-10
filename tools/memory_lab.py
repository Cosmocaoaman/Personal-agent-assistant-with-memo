"""Isolated retrieval benchmark. Usage: python tools/memory_lab.py --benchmark.

Offline results evaluate SQLite/lexical retrieval, NOT model decision quality.
The 500-record sandbox can also be used with the normal web app on port 8767.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import platform
import statistics
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.store import Store, context_memory, normalized, tokens
from agent.retrieval import LexicalIndex
from tools.memory_lab_data import applicable_cases, dataset, manifest, seed

LAB_ROOT = ROOT / 'artifacts' / 'memory-lab'


def legacy_search(store, query, limit=12, context=False):
    """Frozen pre-optimization baseline, with exactly the original ranking."""
    items = store.memories()
    if not items:
        return []
    if not query.strip():
        return items[:limit]
    docs = [Counter(tokens(r['content'] + ' ' + r['key'])) for r in items]
    q = set(tokens(query))
    avg = sum(sum(d.values()) for d in docs) / len(docs) or 1
    df = Counter(t for d in docs for t in d)
    scored = []
    for item, doc in zip(items, docs):
        length = sum(doc.values())
        score = sum(math.log(1 + (len(docs) - df[t] + .5) / (df[t] + .5)) *
                    (doc[t] * 2.2) / (doc[t] + 1.2 * (.25 + .75 * length / avg))
                    for t in q if doc[t])
        if normalized(query) in normalized(item['content']):
            score += 8
        if context:
            score += 8 if item['pinned'] else (2 if item['category'] in ('profile', 'preference') else 0)
        if score > 0:
            scored.append((score, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:limit]]


def ms(fn):
    started = time.perf_counter_ns()
    value = fn()
    return value, (time.perf_counter_ns() - started) / 1e6


def stats(values):
    values = sorted(values)
    return dict(samples=len(values), median_ms=round(statistics.median(values), 3),
        p95_ms=round(values[math.ceil(len(values) * .95) - 1], 3),
        min_ms=round(values[0], 3), max_ms=round(values[-1], 3))


def quality(rows, cases):
    positive = [r for r in rows if r['expected_keys']]
    negative = [r for r in rows if not r['expected_keys']]
    groups = {}
    for kind in {r['kind'] for r in positive}:
        selected = [r for r in positive if r['kind'] == kind]
        groups[kind] = dict(cases=len(selected), recall_at_5=round(statistics.mean(r['recall_at_5'] for r in selected), 3),
                            recall_at_12=round(statistics.mean(r['recall_at_12'] for r in selected), 3))
    return dict(positive_cases=len(positive), negative_cases=len(negative),
        recall_at_5=round(statistics.mean(r['recall_at_5'] for r in positive), 3),
        recall_at_12=round(statistics.mean(r['recall_at_12'] for r in positive), 3),
        mrr=round(statistics.mean(r['reciprocal_rank'] for r in positive), 3),
        negative_queries_with_hits=sum(bool(r['returned_keys']) for r in negative),
        by_kind=groups)


def benchmark(size, repeats):
    rows = dataset(size)
    cases = applicable_cases(rows)
    # All timing comparisons use the identical database and questions. Test
    # databases are temporary, with no access to data/mori.sqlite3.
    with tempfile.TemporaryDirectory(prefix='mori-lab-') as directory:
        store = Store(Path(directory) / 'memory.sqlite3')
        seed(store, rows)
        _, initial_sync = ms(lambda: store.search(cases[0]['query'], 12))
        reopened = Store(store.file)
        _, cold = ms(lambda: reopened.search(cases[0]['query'], 12))
        previous_snapshot, previous_index = None, None

        def previous_search(query):
            nonlocal previous_snapshot, previous_index
            snapshot = store.memories()
            if snapshot != previous_snapshot:
                previous_snapshot, previous_index = snapshot, LexicalIndex(snapshot)
            return previous_index.search(query, 12)

        previous_search(cases[0]['query'])
        timings = {'baseline': [], 'previous_cached': [], 'current': []}
        results = []
        for c in cases:
            baseline = legacy_search(store, c['query'])
            current = store.search(c['query'], 12)
            expected = set(c['expected_keys'])
            returned = [r['key'] for r in current]
            rank = next((i + 1 for i, k in enumerate(returned) if k in expected), None)
            results.append(dict(c, returned_keys=returned,
                ranking_matches_baseline=returned == [r['key'] for r in baseline],
                recall_at_5=len(expected.intersection(returned[:5])) / len(expected) if expected else None,
                recall_at_12=len(expected.intersection(returned)) / len(expected) if expected else None,
                reciprocal_rank=1 / rank if rank else 0))
            # Alternate timing order to reduce systematic warm filesystem bias.
            for repeat in range(repeats):
                pairs = [('baseline', lambda: legacy_search(store, c['query'])),
                         ('previous_cached', lambda: previous_search(c['query'])),
                         ('current', lambda: store.search(c['query'], 12))]
                for name, fn in pairs[::1 if repeat % 2 == 0 else -1]:
                    _, elapsed = ms(fn)
                    timings[name].append(elapsed)
        personal, personal_time = ms(store.personal_context)
        compact = json.dumps([context_memory(r) for r in personal['memories']], ensure_ascii=False)
        return dict(size=size, manifest=manifest(rows), database_bytes=store.file.stat().st_size,
            content_chars=dict(min=min(len(r['content']) for r in rows), max=max(len(r['content']) for r in rows),
                               average=round(statistics.mean(len(r['content']) for r in rows), 1)),
            cold_search_ms=round(cold, 3),
            initial_sync_ms=round(initial_sync, 3), previous_cached=stats(timings['previous_cached']),
            baseline=stats(timings['baseline']), current=stats(timings['current']),
            quality=quality(results, cases), cases=results,
            personal_context=dict(total=personal['total'], loaded=len(personal['memories']),
                omitted=personal['omitted'], memory_json_chars=len(compact), elapsed_ms=round(personal_time, 3)),
            model_quality_evaluated=False, real_api_calls=0)


def write_report(report, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    body = ['# 跨领域记忆测试\n\n', '所有人物与记忆均为虚构，正式数据库未用于测试。\n\n',
            '本报告是本地词项检索实测；不代表真实模型召回、写入判断或 API 延迟。\n\n',
            '| 记忆条数 | 原版中位数 / P95 | 当前中位数 / P95 | 冷启动 | 基础预载 / 候选 | Recall@12 |\n',
            '|---|---|---|---|---|---|\n']
    for b in report['benchmarks']:
        a, z, p = b['baseline'], b['current'], b['personal_context']
        body.append(f"| {b['size']} | {a['median_ms']} / {a['p95_ms']} ms | {z['median_ms']} / {z['p95_ms']} ms | {b['cold_search_ms']} ms | {p['loaded']} / {p['total']} | {b['quality']['recall_at_12']:.1%} |\n")
    body.append('\nRecall@12 是人工标注问题的相关记忆命中比例，不能当作模型回答准确率。中位数/P95 来自同机交替重复查询；冷启动只有一个样本，含读取与建索引。\n')
    if report.get('capacity_before'):
        old, new = report['capacity_before'], report['capacity_current']
        body.append('\n## 连续浏览的上下文容量\n\n同一个 500 条库，模拟子 Agent 每阶段连续请求 20 页后提交结果。该模拟故意忽略容量提醒，验证容量限制独立于模型配合。字符统计包含模型输入消息 JSON，不包含工具定义；不是 Token 数。\n\n| 阶段 | 原版最大单次输入 | 当前最大单次输入 |\n|---|---|---|\n')
        for phase, label in [('recall', '上下文整理'), ('review', '记忆维护'), ('answer', '主 Agent')]:
            body.append(f"| {label} | {old['phases'][phase]['max_input_json_chars']:,} 字符 | {new['phases'][phase]['max_input_json_chars']:,} 字符 |\n")
        body.append(f"\n当前交接 {new['handoff_memories']} 条原始记忆，共 {new['handoff_memory_json_chars']:,} JSON 字符；整理阶段读入 {new['context_read_chars']:,} 字符，维护阶段读入 {new['review_read_chars']:,} 字符（各上限 24,000）。场景与事实更长时会容纳更少条目，省略信息写入流程记录。\n")
    for b in report['benchmarks']:
        body.append(f"\n## {b['size']} 条记忆的检索质量\n\n")
        for kind, q in b['quality']['by_kind'].items():
            body.append(f"- {kind}: {q['cases']} 题，Recall@5 {q['recall_at_5']:.1%}，Recall@12 {q['recall_at_12']:.1%}\n")
        body.append(f"- 无相关记忆的 {b['quality']['negative_cases']} 题中，{b['quality']['negative_queries_with_hits']} 题返回了词项候选；最终是否相关仍须模型筛选。\n")
        body.append('\n| 问题 | 相关记忆 | 前三条返回 |\n|---|---|---|\n')
        for c in b['cases']:
            body.append(f"| {c['query']} | {', '.join(c['expected_keys']) or '无'} | {', '.join(c['returned_keys'][:3]) or '无'} |\n")
    (output / 'report.md').write_text(''.join(body), encoding='utf-8')
    write_html(report, output)


def write_html(report, output):
    """An inspectable offline report with expandable labelled cases."""
    timing_rows, sections = [], []
    for b in report['benchmarks']:
        a, z, p = b['baseline'], b['current'], b['personal_context']
        timing_rows.append(f"<tr><td>{b['size']:,}</td><td>{a['median_ms']} / {a['p95_ms']}</td><td>{z['median_ms']} / {z['p95_ms']}</td><td>{b['cold_search_ms']}</td><td>{p['loaded']} / {p['total']}</td><td>{b['quality']['recall_at_12']:.1%}</td></tr>")
        cards = []
        for c in b['cases']:
            expected = ', '.join(c['expected_keys']) or '没有相关记忆'
            found = ', '.join(c['returned_keys']) or '没有候选'
            label = '无相关事实' if not c['expected_keys'] else f"Recall@12 {c['recall_at_12']:.0%}"
            cards.append(f"<details class='case'><summary>{html.escape(c['name'])} · {label}</summary><p>查询：{html.escape(c['query'])}</p><p>人工标注：{html.escape(expected)}</p><p>候选排序：{html.escape(found)}</p><small>{html.escape(c['kind'])} · 原版与当前排序{'一致' if c['ranking_matches_baseline'] else '不同'}</small></details>")
        sections.append(f"<details class='cohort'><summary>{b['size']:,} 条记忆 · {len(b['cases'])} 个标注问题 · 展开检索结果</summary>{''.join(cards)}</details>")
    capacity = ''
    if report.get('capacity_before'):
        old, new = report['capacity_before'], report['capacity_current']
        capacity = f"<section><h2>连续浏览的容量测试</h2><p>500 条库，脚本化模型每阶段连续读取 20 页。维护阶段最大单次输入从 <strong>{old['phases']['review']['max_input_json_chars']:,}</strong> 降到 <strong>{new['phases']['review']['max_input_json_chars']:,}</strong> 字符。</p><p>当前主 Agent 交接 {new['handoff_memories']} 条原始记忆，共 {new['handoff_memory_json_chars']:,} JSON 字符。模型质量未在此测试中评估。</p></section>"
    domain_count = len(manifest(dataset(500))['domains'])
    content = """<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Mori · 跨领域记忆测试</title>
<style>body{margin:0;background:#f5f5ef;color:#27362e;font:16px/1.65 system-ui,sans-serif}main{max-width:1080px;margin:auto;padding:36px 20px}h1{font-size:32px}h2{font-size:21px}small,.muted{color:#64736a}.badge{display:inline-block;background:#fff0c7;padding:5px 12px;border-radius:8px;font-size:13px}section,.cohort{background:#fff;border:1px solid #dce3da;border-radius:14px;padding:20px;margin:20px 0}table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;border-bottom:1px solid #e4e9e1;padding:12px 8px;white-space:nowrap}.scroll{overflow:auto}summary{cursor:pointer;font-weight:600}.case{padding:12px;margin:10px 0;border:1px solid #e4e9e1;border-radius:9px}.case p{overflow-wrap:anywhere;margin:8px 0}.links a{color:#39704a;margin-right:16px}</style>
<main><span class='badge'>虚构数据 · 正式记忆库未用于测试</span><h1>跨领域记忆测试</h1><p>DOMAIN_COUNT 个主题领域，保留原有 4 种分类。比较同一台电脑上的原版与当前词项检索。</p><p class='muted'>这份报告没有调用真实模型。耗时是本地检索，不包含 API 网络、模型推理或完整聊天延迟。Recall 是人工标注问题的记忆 ID 命中比例。</p>
<section><h2>规模与查询耗时</h2><p>中位数 / P95，单位毫秒；冷启动包含第一次读取和构建索引。</p><div class='scroll'><table><thead><tr><th>记忆数</th><th>原版中位 / P95</th><th>当前中位 / P95</th><th>当前冷启动</th><th>基础预载 / 候选</th><th>Recall@12</th></tr></thead><tbody>""" + ''.join(timing_rows) + "</tbody></table></div></section>" + capacity + "<h2>逐题检查</h2><p>候选并非已经认定的事实，需由记忆子 Agent 筛选。共享中文单字会产生无关候选，缺少词项重合的同义表达可能漏召回。</p>" + ''.join(sections) + "<p class='links'><a href='report.md'>详细报告</a><a href='report.json'>原始测量数据</a><a href='dataset-manifest.json'>数据清单</a></p></main></html>"
    (output / 'report.html').write_text(content.replace('DOMAIN_COUNT', str(domain_count)), encoding='utf-8')


def serve(port, open_browser):
    import hashlib
    import urllib.request
    import webbrowser
    from server import load_env, make_server
    load_env()
    url = f'http://127.0.0.1:{port}'
    workspace = hashlib.sha256(str((LAB_ROOT / 'sandbox').resolve()).encode()).hexdigest()[:16]
    try:
        with urllib.request.urlopen(url + '/api/health', timeout=2) as response:
            health = json.load(response)
        if health.get('app') == 'mori' and health.get('workspace_id') == workspace:
            if open_browser:
                webbrowser.open(url)
            print('Existing fictional memory lab: ' + url)
            return
    except (OSError, ValueError):
        pass
    # A clearly labelled scripted trace lets the user inspect retrieval and
    # capacity before connecting their actual model. Never set a fake API key
    # or replace their model settings in the web app.
    store = Store(LAB_ROOT / 'sandbox' / 'mori.sqlite3')
    if not store.runs():
        from agent.core import Agent
        from agent.providers import Provider
        from tools.memory_lab_probe import ProbeModel
        session = store.create_session('模拟容量流程 · 未调用真实模型')
        client = Provider('deepseek', 'http://127.0.0.1:9876', 'scripted-capacity-probe', 'not-a-real-key', ProbeModel('browse'))
        Agent(store, client).run(session['id'], '【模拟容量测试】连续浏览虚构记忆，查看查询与交接的容量限制。', lambda *_: None)
    try:
        server = make_server(port, LAB_ROOT / 'sandbox')
    except OSError:
        raise SystemExit(f'Port {port} is unavailable; choose another --port. No other service was stopped.') from None
    url = f'http://127.0.0.1:{server.server_address[1]}'
    print(f'FICTIONAL memory lab: {url}\nSeparate database; Ctrl+C or stop-memory-lab.cmd to stop.', flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def real_evaluation(args):
    """Opt-in evaluation using the existing Provider and real LangGraph nodes."""
    import os
    from server import load_env
    from agent.core import Agent
    from agent.providers import DEFAULTS, Provider
    load_env()
    name = args.real_provider
    key = os.getenv(name.upper() + '_API_KEY', '')
    if not key:
        raise SystemExit(f'Missing {name.upper()}_API_KEY; no real API was called. Configure the environment or use the test web app.')
    base = args.base_url or os.getenv(name.upper() + '_BASE_URL', DEFAULTS[name]['base_url'])
    model = args.model or os.getenv(name.upper() + '_MODEL', DEFAULTS[name]['model'])
    provider = Provider(name, base, model, key)
    cases = [c for c in applicable_cases(dataset(500)) if c['name'] in
             ('异步爬虫', '奶制品', '京都行程', '隐含咖啡问法', '中文无关问题')]
    writes = [
        dict(name='修改咖啡习惯', query='我现在下午两点以后不喝咖啡了，之前的三点需要更新。', expected_keys=['food.coffee'], expected_operation='update', target='food.coffee'),
        dict(name='更新居住地', query='我最近从悉尼搬到了阿德莱德，今后以新住址为准。', expected_keys=['basic.city'], expected_operation='update', target='basic.city'),
        dict(name='提取新项目', query='我开始学习西班牙语了，每周三晚上练习半小时，计划坚持三个月。', expected_keys=[], expected_operation='create'),
        dict(name='忘记自行车', query='把我通勤自行车的品牌型号从长期记忆中删掉。', expected_keys=['fitness.bike'], expected_operation='delete', target='fitness.bike'),
        dict(name='虚构引用不提取', query='这是小说人物的台词：“我最喜欢吃重辣。”分析一下人物性格，这句话不是说我。', expected_keys=[], expected_operation='skip'),
        dict(name='普通问题不提取', query='Python 的列表和元组有什么区别？', expected_keys=[], expected_operation='skip'),
    ]
    results = []
    for c in (cases + writes)[:args.case_limit]:
        with tempfile.TemporaryDirectory(prefix='mori-real-eval-') as directory:
            store = Store(Path(directory) / 'memory.sqlite3')
            seed(store, dataset(500))
            before = {r['key']: r for r in store.memories()}
            session = store.create_session('虚构真实模型评测')
            result, elapsed = ms(lambda: Agent(store, provider).run(session['id'], c['query'], lambda *_: None))
            trace = store.run(result['answer']['meta']['run_id'])
            selected = [result['context']['seen'][mid]['key'] for mid in result['context']['memory_ids']]
            changes = result['memory_result']['changes']
            expected_op = c.get('expected_operation')
            target_id = before[c['target']]['id'] if c.get('target') else None
            match = None if not expected_op else (result['memory_result']['status'] == 'skipped' if expected_op == 'skip' else
                    any(op['operation'] == expected_op and (not target_id or op['id'] == target_id) for op in changes))
            r = dict(c, selected_keys=selected, summary=result['context']['summary'], answer=result['answer']['content'],
                     memory_result=result['memory_result'], elapsed_ms=round(elapsed, 3),
                     id_recall_at_handoff=len(set(c['expected_keys']).intersection(selected)) / len(c['expected_keys']) if c['expected_keys'] else None,
                     operation_match=match, phase_ms={k: n['duration_ms'] for k, n in trace['nodes'].items()},
                     usage=result['answer']['meta']['usage'], trace=trace,
                     facts_after=[dict(id=m['id'], key=m['key'], content=m['content']) for m in store.memories()
                                  if any(op['id'] == m['id'] for op in changes)])
            results.append(r)
            print(json.dumps(dict(case=c['name'], elapsed_ms=r['elapsed_ms'], operation_match=match, context_ok=result['context']['ok'])), flush=True)
            if result['answer']['meta'].get('error') and result['memory_result']['status'] == 'error':
                break
    report = dict(real_model=True, model=model, provider=name, synthetic=True, cases=results,
        attempted_model_calls=sum(n['model_calls'] for r in results for n in r['trace']['nodes'].values()),
        note='ID recall and operation matching are mechanical checks. Review summaries, extracted content and answers manually for semantic correctness.')
    (LAB_ROOT / 'real-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--sizes', type=int, nargs='+', default=[10, 100, 500, 2000])
    parser.add_argument('--repeats', type=int, default=8)
    parser.add_argument('--seed-only', action='store_true')
    parser.add_argument('--serve', action='store_true')
    parser.add_argument('--open', action='store_true')
    parser.add_argument('--port', type=int, default=8767)
    parser.add_argument('--real', action='store_true', help='Opt in to billable real-model tests; requires an environment API key')
    parser.add_argument('--real-provider', choices=['deepseek', 'openai'], default='deepseek')
    parser.add_argument('--model')
    parser.add_argument('--base-url')
    parser.add_argument('--case-limit', type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 50 or any(not 10 <= n <= 10000 for n in args.sizes):
        parser.error('sizes: 10..10000; repeats: 1..50')
    LAB_ROOT.mkdir(parents=True, exist_ok=True)
    sandbox = Store(LAB_ROOT / 'sandbox' / 'mori.sqlite3')
    rows = dataset(500)
    created = seed(sandbox, rows)
    sandbox.set_settings({'lab_dataset': {'synthetic': True, 'name': '跨领域记忆测试', 'initial_count': 500}})
    (LAB_ROOT / 'dataset-manifest.json').write_text(json.dumps(manifest(rows), ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'sandbox': str(sandbox.file), 'created': created, 'records': len(sandbox.memories()), 'synthetic': True}))
    if args.seed_only:
        return
    if args.serve:
        serve(args.port, args.open)
        return
    if args.real:
        if not 1 <= args.case_limit <= 11:
            parser.error('case-limit: 1..11')
        real_evaluation(args)
        return
    report = dict(version=2, measured_at=datetime.now(timezone.utc).isoformat(),
                  environment=dict(python=platform.python_version(), os=platform.platform()),
                  methodology='isolated synthetic corpus; alternating baseline/current; no network or real model', benchmarks=[])
    for size in args.sizes:
        b = benchmark(size, args.repeats)
        report['benchmarks'].append(b)
        print(json.dumps({k: b[k] for k in ('size', 'baseline', 'current', 'cold_search_ms', 'quality', 'personal_context')}), flush=True)
    from tools.memory_lab_probe import probe
    capacity = probe(500, 'browse')
    (LAB_ROOT / 'current-probe.json').write_text(json.dumps(capacity, ensure_ascii=False, indent=2), encoding='utf-8')
    report['capacity_current'] = {k: v for k, v in capacity.items() if k != 'trace'}
    old = LAB_ROOT / 'baseline-probe.json'
    if old.exists():
        report['capacity_before'] = {k: v for k, v in json.loads(old.read_text(encoding='utf-8')).items() if k != 'trace'}
    write_report(report, LAB_ROOT)
    print(json.dumps({'report': str(LAB_ROOT / 'report.md')}))


if __name__ == '__main__':
    main()
