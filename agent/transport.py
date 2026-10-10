"""Pooled HTTPS and bounded SSE reading. Request metrics never contain payloads."""
from __future__ import annotations

import atexit
import codecs
import json
import threading
import time

import httpx


class ProviderError(Exception):
    def __init__(self, message, request_timing=None):
        super().__init__(message)
        self.request_timing = request_timing


def output_event(event):
    if event.get('type') in ('response.output_text.delta', 'response.refusal.delta',
                              'response.function_call_arguments.delta'):
        return bool(event.get('delta'))
    for choice in event.get('choices') or []:
        delta = choice.get('delta') or {}
        if delta.get('content') or any((t.get('function') or {}).get('arguments') for t in delta.get('tool_calls') or []):
            return True
    return False


class PooledHTTP:
    def __init__(self, client=None):
        self._client = client
        self._lock = threading.Lock()

    def client(self):
        with self._lock:
            if self._client is None:
                self._client = httpx.Client(follow_redirects=False,
                    limits=httpx.Limits(max_connections=8, max_keepalive_connections=8, keepalive_expiry=120))
            return self._client

    def close(self):
        with self._lock:
            if self._client is not None:
                self._client.close()
                self._client = None

    def __call__(self, url, key, payload, timeout=90, on_event=None):
        started = time.perf_counter()
        timing = dict(transport='HTTP', streaming=False, connection_ms=0, connection_reused=None,
                      headers_ms=None, first_event_ms=None, first_output_ms=None, output_ms=None, total_ms=None)
        connections, network_seen = {}, False

        def elapsed():
            return round((time.perf_counter() - started) * 1000, 3)

        def trace(name, info):
            nonlocal network_seen
            if 'send_request_headers' in name:
                network_seen = True
            if name.endswith(('connect_tcp.started', 'start_tls.started')):
                connections[name.rsplit('.', 1)[0]] = time.perf_counter()
                timing['connection_reused'] = False
            elif name.endswith(('connect_tcp.complete', 'start_tls.complete')):
                begin = connections.pop(name.rsplit('.', 1)[0], None)
                if begin is not None:
                    timing['connection_ms'] += (time.perf_counter() - begin) * 1000

        def finish():
            timing['total_ms'] = elapsed()
            timing['connection_ms'] = None if timing['connection_reused'] is None and not network_seen else round(timing['connection_ms'], 3)
            if network_seen and timing['connection_reused'] is None:
                timing['connection_reused'] = True
            if timing['first_output_ms'] is not None:
                timing['output_ms'] = round(timing['total_ms'] - timing['first_output_ms'], 3)
            return dict(timing)

        try:
            with self.client().stream('POST', url, json=payload, timeout=timeout,
                    headers={'Authorization': f'Bearer {key}', 'User-Agent': 'MoriLocalAgent/2.7',
                             'Accept': 'text/event-stream, application/json'}, follow_redirects=False,
                    extensions={'trace': trace}) as response:
                timing['headers_ms'] = elapsed()
                timing['transport'] = response.http_version
                if not 200 <= response.status_code < 300:
                    hints = {400: '请求格式或模型参数不兼容，请检查模型名称和 API 类型。',
                        401: 'API Key 无效，请重新填写。', 402: 'API 账户余额不足。',
                        403: '账户或区域没有访问权限。', 404: '模型或接口地址不存在。',
                        429: '请求过多或额度不足，请稍后重试。'}
                    raise ProviderError(f"API 返回 HTTP {response.status_code}。{hints.get(response.status_code, '服务暂不可用，请稍后重试。')}")
                stream = 'text/event-stream' in response.headers.get('content-type', '').lower()
                timing['streaming'] = stream
                if stream:
                    if on_event is None:
                        raise ProviderError('此请求不支持流式响应。')
                    done = self._sse(response.iter_bytes(), on_event, timing, elapsed)
                    return {'_streamed': True, '_done': done, '_request_timing': finish()}
                body = bytearray()
                for chunk in response.iter_bytes():
                    body.extend(chunk)
                    if len(body) > 8 * 1024 * 1024:
                        raise ProviderError('API 响应过大')
                data = json.loads(body)
                if not isinstance(data, dict):
                    raise ProviderError('API 返回了不支持的响应格式')
                data['_request_timing'] = finish()
                return data
        except ProviderError as exc:
            exc.request_timing = finish()
            raise
        except httpx.TimeoutException:
            raise ProviderError('API 请求超时，请检查网络后重试。', finish()) from None
        except httpx.HTTPError:
            raise ProviderError('无法连接 API 或响应流已中断，请检查地址、网络或系统代理。', finish()) from None
        except (ValueError, UnicodeError, TypeError):
            raise ProviderError('API 没有返回有效 JSON 或流式事件，请检查接口地址。', finish()) from None

    @staticmethod
    def _sse(chunks, on_event, timing, elapsed):
        decoder, buffer, lines, size, done = codecs.getincrementaldecoder('utf-8')(), '', [], 0, False

        def dispatch():
            nonlocal done
            raw = '\n'.join(lines)
            lines.clear()
            if not raw:
                return
            if raw == '[DONE]':
                done = True
                return
            if done:
                raise ProviderError('API 流式结束标记之后仍返回数据。')
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ProviderError('API 返回了无效流式事件。')
            if timing['first_event_ms'] is None:
                timing['first_event_ms'] = elapsed()
            if timing['first_output_ms'] is None and output_event(data):
                timing['first_output_ms'] = elapsed()
            on_event(data)

        for chunk in chunks:
            size += len(chunk)
            if size > 8 * 1024 * 1024:
                raise ProviderError('API 响应过大')
            buffer += decoder.decode(chunk)
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                line = line.rstrip('\r').lstrip('\ufeff')
                if not line:
                    dispatch()
                elif line.startswith('data:'):
                    lines.append(line[5:].lstrip(' '))
        buffer += decoder.decode(b'', final=True)
        if buffer.startswith('data:'):
            lines.append(buffer[5:].lstrip(' ').rstrip('\r'))
        dispatch()
        return done


_default_http = PooledHTTP()
atexit.register(_default_http.close)


def post_json(url, key, payload, timeout=90, on_event=None):
    return _default_http(url, key, payload, timeout=timeout, on_event=on_event)
