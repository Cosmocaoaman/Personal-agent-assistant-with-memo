"""Opt-in short real-API transport check; source DB is opened read-only."""
import argparse
import json
import sqlite3
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.providers import Provider
from agent.transport import PooledHTTP
from agent.core import Agent
from agent.store import Store


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--real-api', action='store_true')
    parser.add_argument('--key-data-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.real_api: parser.error('Real API access requires --real-api.')
    database = (args.key_data_dir / 'mori.sqlite3').resolve()
    connection = sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)
    # Reuse validated credential access without exporting keys or modifying source.
    settings = {r[0]: json.loads(r[1]) for r in connection.execute('SELECT key,value FROM settings')}
    credential = connection.execute("SELECT base_url,api_key FROM credentials WHERE provider='openai'").fetchone()
    connection.close()
    config = settings['openai']
    if not credential or credential[0] != config['base_url'] or not credential[1]:
        raise SystemExit('No matching saved OpenAI credential.')
    key = credential[1]
    options = dict(config.get('options', {}), max_output_tokens=512, web_search=False)
    pool = PooledHTTP()
    rows = []
    prompt = [{'role': 'user', 'content': '用中文列出五条整理书桌的小建议，每条一句，不超过20字。'}]
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args): return None
    def legacy(url, token, payload):
        request = urllib.request.Request(url, data=json.dumps(payload).encode(),
            headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + token})
        with urllib.request.build_opener(NoRedirect).open(request, timeout=90) as response:
            return json.load(response)
    try:
        for label, transport in [('legacy_json', legacy), ('pooled_stream_cold', pool), ('pooled_stream_warm', pool)]:
            client = Provider('openai', config['base_url'], config['model'], key, transport, options=options)
            first, pieces, began = [], [], time.perf_counter()
            def delta(text):
                if not pieces: first.append(round((time.perf_counter() - began) * 1000, 3))
                pieces.append(text)
            text, calls, usage = client.complete(client.start('请给出简短实用的回答。', prompt), [], final=True, on_delta=delta)
            if not text or calls: raise RuntimeError('Probe did not produce a complete answer.')
            rows.append(dict(mode=label, total_ms=round((time.perf_counter()-began)*1000, 3),
                first_visible_ms=first[0] if first else None, chunks=len(pieces), output_chars=len(text),
                input_tokens=usage.get('input_tokens'), output_tokens=usage.get('output_tokens'), request=usage.get('request_timing')))
        with tempfile.TemporaryDirectory() as root:
            store = Store(Path(root) / 'probe.sqlite3'); session = store.create_session()
            store.save_memory('喜欢简洁中文回答', 'preference', 'style')
            client = Provider('openai', config['base_url'], config['model'], key, pool, options=options)
            events = []
            result = Agent(store, client).run(session['id'], prompt[0]['content'], lambda name, value: events.append((name, value['elapsed_ms'])))
            if result['answer']['meta'].get('error'): raise RuntimeError('Real graph answer failed.')
            trace = store.run(result['answer']['meta']['run_id'])
            graph = {'status': trace['status'], 'events': events, 'nodes': {name: {
                k: node.get(k) for k in ('duration_ms','model_calls','first_output_ms','requests')} for name, node in trace['nodes'].items()}}
        report = dict(model=config['model'], measurements=rows, graph=graph,
            limitation='One short synthetic question, one sample per mode. Not a statistical performance comparison or a 500-memory benchmark.')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(report, ensure_ascii=True))
    finally: pool.close()


if __name__ == '__main__': main()
