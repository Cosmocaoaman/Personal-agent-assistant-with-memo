"""Run with the project virtual environment, or use start.cmd."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import threading
import urllib.parse
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from agent.core import Agent
from agent.providers import DEFAULTS, Provider, ProviderError, validate_base
from agent.transport import PooledHTTP
from agent.store import Store
from agent.config import model_options, runtime_options, EFFORTS
from agent.semantic import EMBEDDING_DEFAULTS, EmbeddingProvider, HybridRetriever, embedding_options
from agent.routing import router_context

ROOT = Path(__file__).resolve().parent


def load_env():
    file = ROOT / '.env'
    if file.exists():
        for line in file.read_text(encoding='utf-8-sig').splitlines():
            if line.strip() and not line.lstrip().startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ.setdefault(key.strip(), value.strip().strip('\"').strip("'"))


class App:
    def __init__(self, data_dir):
        self.store = Store(Path(data_dir) / 'mori.sqlite3')
        self.workspace_id = hashlib.sha256(str(Path(data_dir).resolve()).encode()).hexdigest()[:16]
        self.csrf = secrets.token_urlsafe(32)
        self.lock = threading.Lock()
        self.transport = PooledHTTP()
        saved = self.store.settings()
        self.dataset = saved.get('lab_dataset')
        self.settings = dict(provider=saved.get('provider', os.getenv('AGENT_PROVIDER', 'deepseek')),
                             auto_memory=saved.get('auto_memory', True), persona=saved.get('persona', ''),
                             runtime=runtime_options(saved.get('runtime')))
        if self.settings['provider'] not in DEFAULTS:
            self.settings['provider'] = 'deepseek'
        self.keys = {}
        for name, default in DEFAULTS.items():
            cfg = saved.get(name, {})
            self.settings[name] = {k: cfg.get(k, os.getenv(f'{name.upper()}_{k.upper()}', v)) for k, v in default.items()}
            self.settings[name]['options'] = model_options(name, cfg.get('options'))
            self.settings[name]['base_url'] = self.settings[name]['base_url'].rstrip('/')
            credential = self.store.credential(name)
            if credential is not None:
                # An empty record means explicitly cleared: do not reload from .env.
                key = credential['api_key'] if credential['base_url'] == self.settings[name]['base_url'] else ''
            else:
                key = os.getenv(f'{name.upper()}_API_KEY', '').strip()
                env_base = os.getenv(f'{name.upper()}_BASE_URL', default['base_url']).rstrip('/')
                if self.settings[name]['base_url'] != env_base or len(key) > 1000 or '\n' in key or '\r' in key:
                    key = ''
                if key:
                    # Import legacy environment keys once and preserve their endpoint.
                    self.store.set_settings({name: self.settings[name], 'provider': self.settings['provider']},
                                            dict(provider=name, base_url=env_base, api_key=key))
            if key:
                self.keys[name] = key
        self.settings['embedding'] = embedding_options(dict(dict(EMBEDDING_DEFAULTS,
            base_url=self.settings['openai']['base_url']), **saved.get('embedding', {})))
        credential = self.store.credential('embedding')
        if credential and credential['base_url'] == self.settings['embedding']['base_url'] and credential['api_key']:
            self.keys['embedding'] = credential['api_key']
        self.retriever = HybridRetriever(self.store, self.embedding_provider, lambda: self.settings['embedding'])
        if self.settings['embedding']['enabled']:
            self.retriever.schedule()

    def public_settings(self):
        return dict(self.settings, keys={p: bool(self.keys.get(p)) for p in DEFAULTS},
                    embedding_key_available=bool(self.embedding_key()), separate_embedding_key=bool(self.keys.get('embedding')),
                    key_storage=dict(method='local_database', encrypted=False),
                    efforts=EFFORTS, dataset=self.dataset)

    def embedding_key(self):
        config = self.settings['embedding']
        if config['key_source'] == 'openai':
            return self.keys.get('openai', '') if config['base_url'] == self.settings['openai']['base_url'] else ''
        return self.keys.get('embedding', '')

    def embedding_provider(self):
        key = self.embedding_key()
        if not key:
            raise ValueError('Embedding 没有可用 Key；复用 OpenAI Key 时 API 地址必须一致')
        return EmbeddingProvider(self.settings['embedding'], key, self.transport)

    def provider(self):
        name = self.settings['provider']
        key = self.keys.get(name)
        if not key:
            raise ValueError('请先在「连接设置」中填写 API Key')
        return Provider(name, **self.settings[name], key=key, transport=self.transport)

    def configure(self, data):
        name = data.get('provider')
        if name not in DEFAULTS:
            raise ValueError('请选择 DeepSeek 或 OpenAI')
        base = validate_base(data.get('base_url', ''))
        model = data.get('model', '')
        if not isinstance(model, str) or not 1 <= len(model.strip()) <= 100:
            raise ValueError('模型名称需要 1–100 个字符')
        persona = data.get('persona', '')
        if not isinstance(persona, str) or len(persona) > 2000:
            raise ValueError('回答风格不能超过 2000 个字符')
        auto = data.get('auto_memory', True)
        if type(auto) is not bool:
            raise ValueError('自动记忆开关格式错误')
        key = data.get('api_key', '')
        if not isinstance(key, str) or len(key) > 1000 or '\n' in key or '\r' in key:
            raise ValueError('API Key 格式错误')
        clear = data.get('clear_key', False)
        if type(clear) is not bool:
            raise ValueError('clear_key 格式错误')
        options = model_options(name, data.get('options', self.settings[name]['options']))
        runtime = runtime_options(data.get('runtime', self.settings['runtime']))
        embedding_data = data.get('embedding')
        embedding = dict(self.settings['embedding'])
        embedding_credential = None
        embedding_key, embedding_clear = '', False
        if embedding_data is not None:
            if not isinstance(embedding_data, dict):
                raise ValueError('Embedding 配置格式错误')
            fields = dict(embedding_data)
            embedding_key, embedding_clear = fields.pop('api_key', ''), fields.pop('clear_key', False)
            if not isinstance(embedding_key, str) or len(embedding_key) > 1000 or '\n' in embedding_key or '\r' in embedding_key or type(embedding_clear) is not bool:
                raise ValueError('Embedding Key 格式错误')
            embedding = embedding_options(dict(embedding, **fields))
            openai_base = base if name == 'openai' else self.settings['openai']['base_url']
            if embedding['key_source'] == 'openai' and embedding['base_url'] != openai_base:
                raise ValueError('复用 OpenAI Key 时，Embedding 地址需与 OpenAI 聊天地址一致；其他地址请使用独立 Key')
            if embedding_key.strip() and embedding['key_source'] != 'separate':
                raise ValueError('填写 Embedding Key 前请选择独立 Key')
        credential = None
        updated_keys = dict(self.keys)
        if clear or base != self.settings[name]['base_url']:
            updated_keys.pop(name, None)
            credential = dict(provider=name, base_url=base, api_key='')
        if key.strip() and not clear:
            updated_keys[name] = key.strip()
            credential = dict(provider=name, base_url=base, api_key=key.strip())
        if embedding_clear or embedding['base_url'] != self.settings['embedding']['base_url']:
            updated_keys.pop('embedding', None)
            embedding_credential = dict(provider='embedding', base_url=embedding['base_url'], api_key='')
        if embedding_key.strip() and not embedding_clear:
            updated_keys['embedding'] = embedding_key.strip()
            embedding_credential = dict(provider='embedding', base_url=embedding['base_url'], api_key=embedding_key.strip())
        updated = dict(self.settings, provider=name, auto_memory=auto, persona=persona, runtime=runtime, embedding=embedding)
        updated[name] = dict(base_url=base, model=model.strip(), options=options)
        # Configuration and Key commit together; failures preserve both old values.
        self.store.set_settings(updated, [c for c in (credential, embedding_credential) if c is not None])
        self.settings = updated
        self.keys = updated_keys
        if embedding['enabled']:
            self.retriever.schedule()
        return self.public_settings()


class HTTPError(Exception):
    def __init__(self, status, message):
        self.status, self.message = status, message


class Handler(BaseHTTPRequestHandler):
    server_version = 'Mori/1.0'

    @property
    def app(self):
        return self.server.app

    def log_message(self, *args):
        pass  # No request bodies, keys, messages, or query strings in console logs.

    def send_headers(self, status, content_type, length=None):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('X-Frame-Options', 'DENY')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'")
        if length is not None:
            self.send_header('Content-Length', str(length))
        self.end_headers()

    def json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_headers(status, 'application/json; charset=utf-8', len(body))
        self.wfile.write(body)

    def guard(self, mutation=False):
        port = self.server.server_address[1]
        allowed_hosts = {f'127.0.0.1:{port}', f'localhost:{port}'}
        host = self.headers.get('Host', '')
        if host not in allowed_hosts:
            raise HTTPError(403, '只允许本机访问')
        origin = self.headers.get('Origin')
        if origin and origin not in {f'http://{h}' for h in allowed_hosts}:
            raise HTTPError(403, '不允许跨来源请求')
        if self.headers.get('Sec-Fetch-Site') == 'cross-site':
            raise HTTPError(403, '不允许跨站访问')
        if mutation:
            if not secrets.compare_digest(self.headers.get('X-Mori-Token', ''), self.app.csrf):
                raise HTTPError(403, '页面已过期，请刷新后再试')
            if self.headers.get_content_type() != 'application/json':
                raise HTTPError(415, '只接受 JSON')

    def read_json(self):
        try:
            size = int(self.headers.get('Content-Length', '0'))
        except ValueError:
            raise HTTPError(400, '请求长度无效')
        if not 0 < size <= 128000:
            raise HTTPError(413, '请求内容过大或为空')
        try:
            data = json.loads(self.rfile.read(size))
        except (ValueError, UnicodeDecodeError):
            raise HTTPError(400, 'JSON 格式错误')
        if not isinstance(data, dict):
            raise HTTPError(400, '请求必须是 JSON 对象')
        return data

    def dispatch(self, mutation=False):
        streaming = False
        acquired = False
        try:
            self.guard(mutation)
            parsed = urllib.parse.urlsplit(self.path)
            route, query = parsed.path, urllib.parse.parse_qs(parsed.query)
            store = self.app.store
            if not mutation:
                if route == '/api/health':
                    return self.json({'app': 'mori', 'version': '2.8', 'orchestration': 'langgraph', 'workspace_id': self.app.workspace_id})
                if route == '/api/runs':
                    return self.json(store.runs(query.get('session', [None])[0]))
                if route == '/api/runs/detail':
                    return self.json(store.run(query.get('id', [''])[0]))
                if route == '/api/state':
                    return self.json(dict(settings=self.app.public_settings(), csrf=self.app.csrf,
                                          sessions=store.sessions(), memories=store.memories(), tasks=store.tasks(),
                                          personal_context=store.personal_context(self.app.settings['runtime']['personal_items'], self.app.settings['runtime']['personal_chars'])))
                if route == '/api/messages':
                    sid = query.get('session', [''])[0]
                    store.session(sid)
                    return self.json(store.messages(sid))
                if route == '/api/memories':
                    search = query.get('q', [''])[0]
                    return self.json(store.search(search, 100) if search else store.memories())
                if route == '/api/retrieval/status':
                    return self.json(self.app.retriever.status())
                if route == '/api/export':
                    return self.json(store.export())
                assets = {'/': ('index.html', 'text/html'), '/app.js': ('app.js', 'text/javascript'),
                          '/flow.js': ('flow.js', 'text/javascript'), '/flow.css': ('flow.css', 'text/css'),
                          '/style.css': ('style.css', 'text/css'), '/favicon.svg': ('favicon.svg', 'image/svg+xml')}
                if route not in assets:
                    raise HTTPError(404, '页面不存在')
                name, kind = assets[route]
                body = (ROOT / 'web' / name).read_bytes()
                self.send_headers(200, kind + '; charset=utf-8', len(body))
                return self.wfile.write(body)
            data = self.read_json()
            acquired = self.app.lock.acquire(blocking=False)
            if not acquired:
                raise HTTPError(409, 'Agent 正在处理请求，请等待本轮结束后再操作')
            if route == '/api/settings':
                return self.json(self.app.configure(data))
            if route == '/api/shutdown':
                self.json({'ok': True})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return
            if route == '/api/probe':
                return self.json(self.app.provider().probe())
            if route == '/api/embedding/probe':
                return self.json(self.app.embedding_provider().probe())
            if route == '/api/embedding/index':
                return self.json(self.app.retriever.schedule(force=True))
            if route == '/api/retrieval/test':
                query_text = data.get('query', '')
                if not isinstance(query_text, str) or not 1 <= len(query_text.strip()) <= 2000:
                    raise ValueError('召回测试查询需为 1–2000 个字符')
                rows, detail = self.app.retriever.search(query_text, 12)
                return self.json(dict(memories=rows, retrieval=detail))
            if route == '/api/sessions':
                return self.json(store.create_session())
            if route == '/api/sessions/delete':
                store.delete_session(self.identifier(data))
                return self.json({'ok': True})
            if route == '/api/memories/save':
                if type(data.get('pinned', False)) is not bool:
                    raise ValueError('置顶开关格式错误')
                memory = store.save_memory(data.get('content'), data.get('category', 'note'),
                                 data.get('key', ''), data.get('pinned', False), memory_id=data.get('id'))
                self.app.retriever.schedule()
                return self.json(memory)
            if route == '/api/memories/delete':
                store.delete_memory(self.identifier(data))
                self.app.retriever.schedule()
                return self.json({'ok': True})
            if route == '/api/tasks/add':
                return self.json(store.add_task(data.get('title')))
            if route == '/api/tasks/set':
                store.set_task(self.identifier(data), data.get('done'))
                return self.json({'ok': True})
            if route == '/api/tasks/delete':
                store.delete_task(self.identifier(data))
                return self.json({'ok': True})
            if route == '/api/chat':
                text = data.get('message', '')
                if not isinstance(text, str) or not 1 <= len(text.strip()) <= 12000:
                    raise ValueError('消息需要 1–12000 个字符')
                sid = data.get('session_id')
                if not isinstance(sid, str):
                    raise ValueError('会话格式错误')
                store.session(sid)
                routed = router_context(data.get('router_context'))
                provider = self.app.provider()
                self.send_headers(200, 'application/x-ndjson; charset=utf-8')
                streaming = True

                def emit(event, payload):
                    self.wfile.write((json.dumps(dict(event=event, **payload), ensure_ascii=False) + '\n').encode())
                    self.wfile.flush()

                agent = Agent(store, provider, self.app.settings['auto_memory'], self.app.settings['persona'], self.app.settings['runtime'], self.app.retriever)
                return agent.run(sid, text.strip(), emit, routed)
            raise HTTPError(404, '接口不存在')
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except Exception as exc:
            status, message = 500, '服务出现内部错误，请刷新后重试。'
            if isinstance(exc, HTTPError):
                status, message = exc.status, exc.message
            elif isinstance(exc, ValueError):
                status, message = 400, str(exc)
            elif isinstance(exc, ProviderError):
                status, message = 502, str(exc)
            try:
                if streaming:
                    self.wfile.write((json.dumps({'event': 'error', 'message': message}, ensure_ascii=False) + '\n').encode())
                else:
                    self.json({'error': message}, status)
            except OSError:
                pass
        finally:
            if acquired:
                self.app.lock.release()

    @staticmethod
    def identifier(data):
        value = data.get('id')
        if not isinstance(value, str) or not 1 <= len(value) <= 100:
            raise ValueError('ID 格式错误')
        return value

    def do_GET(self):
        self.dispatch()

    def do_POST(self):
        self.dispatch(True)


class MoriServer(ThreadingHTTPServer):
    def server_close(self):
        super().server_close()
        if hasattr(self, 'app'):
            self.app.transport.close()


def make_server(port=8765, data_dir=None):
    server = MoriServer(('127.0.0.1', port), Handler)
    server.app = App(data_dir or ROOT / 'data')
    server.app.store.recover_runs()
    return server


def main():
    load_env()
    parser = argparse.ArgumentParser(description='Mori personal agent')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--open', action='store_true', help='Open the local app in your browser')
    parser.add_argument('--stop', action='store_true', help='Stop this workspace\'s existing local server')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    args = parser.parse_args()
    url = f'http://127.0.0.1:{args.port}'
    workspace_id = hashlib.sha256(str(args.data_dir.resolve()).encode()).hexdigest()[:16]

    def existing():
        try:
            with urllib.request.urlopen(url + '/api/health', timeout=2) as response:
                data = json.load(response)
                return data.get('app') == 'mori' and data.get('workspace_id') == workspace_id
        except (OSError, ValueError):
            return False

    if args.stop:
        if not existing():
            print('No running Mori server found for this workspace and port.')
            return 1
        try:
            with urllib.request.urlopen(url + '/api/state', timeout=3) as response:
                token = json.load(response)['csrf']
            req = urllib.request.Request(url + '/api/shutdown', data=b'{}',
                                         headers={'Content-Type': 'application/json', 'X-Mori-Token': token})
            with urllib.request.urlopen(req, timeout=3):
                print('Mori stopped. Your local data is preserved.')
            return 0
        except OSError:
            print('Could not stop Mori. Wait for the active request to finish, then try again.')
            return 1
    if args.open and existing():
        webbrowser.open(url)
        print('Opened the existing Mori server: ' + url)
        return 0
    try:
        server = make_server(args.port, args.data_dir)
    except OSError as exc:
        print(f'Cannot start local server: {exc}. Try --port 8766.')
        return 1
    url = f'http://127.0.0.1:{server.server_address[1]}'
    print(f'Mori is ready: {url}\nPress Ctrl+C to stop.', flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
