"""Stateless HTTPS adapters shared by main and memory agents; histories stay separate."""
from __future__ import annotations

import json
import urllib.parse
from .config import model_options
from .transport import PooledHTTP, ProviderError, post_json
from .streaming import StreamResult


DEFAULTS = {
    'deepseek': {'base_url': 'https://api.deepseek.com', 'model': 'deepseek-flash'},
    'openai': {'base_url': 'https://api.openai.com/v1', 'model': 'gpt-5.6-terra'},
}


def validate_base(value):
    if not isinstance(value, str) or len(value) > 500:
        raise ValueError('API 地址格式错误')
    parsed = urllib.parse.urlsplit(value.strip())
    local = parsed.hostname in ('127.0.0.1', 'localhost', '::1')
    if (parsed.scheme != 'https' and not (parsed.scheme == 'http' and local)) or not parsed.hostname:
        raise ValueError('API 地址必须使用 HTTPS；本机地址允许 HTTP')
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('API 地址不能包含账户、查询参数或片段')
    return value.strip().rstrip('/')


class Provider:
    def __init__(self, provider, base_url, model, key, transport=post_json, options=None):
        self.provider, self.base_url = provider, validate_base(base_url)
        self.model, self.key, self.transport = model, key, transport
        self.options = model_options(provider, options)

    @property
    def web_enabled(self):
        return self.provider == 'openai' and self.options['web_search']

    def _send(self, path, payload, on_delta=None, on_first_output=None):
        if self.transport is post_json or isinstance(self.transport, PooledHTTP):
            payload['stream'] = True
            if self.provider == 'deepseek':
                payload['stream_options'] = {'include_usage': True}
            result = StreamResult(self.provider, on_delta, on_first_output)
            data = self.transport(self.base_url + path, self.key, payload,
                timeout=self.options['timeout_seconds'], on_event=result)
            timing = data.pop('_request_timing', None)
            if data.get('_streamed'):
                try:
                    data = result.finish(data.get('_done', False))
                except ProviderError as exc:
                    exc.request_timing = timing
                    raise
            elif data.get('error'):
                raise ProviderError('模型处理失败，请重试。', timing)
            data['_request_timing'] = timing
            return data
        return self.transport(self.base_url + path, self.key, payload)

    def start(self, system, messages):
        return [{'role': 'system', 'content': system}] + messages

    def complete(self, history, tools, final=False, allow_web=True, on_delta=None, on_first_output=None):
        opts = self.options
        if self.provider == 'openai':
            definitions = [dict(type='function', name=t['name'], description=t['description'],
                                parameters=t['parameters'], strict=True) for t in tools]
            web = final and allow_web and self.web_enabled
            if web:
                definitions.append(dict(type='web_search', search_context_size=opts['search_context_size'], external_web_access=True))
            payload = dict(model=self.model, input=history, tools=definitions, store=False,
                           include=['reasoning.encrypted_content'], max_output_tokens=opts['max_output_tokens'],
                           tool_choice='auto' if tools or web or not final else 'none', parallel_tool_calls=False)
            if opts['reasoning_effort'] != 'auto': payload['reasoning'] = {'effort': opts['reasoning_effort']}
            if opts['verbosity'] != 'auto' and final: payload['text'] = {'verbosity': opts['verbosity']}
            for key in ('temperature', 'top_p'):
                if opts[key] is not None: payload[key] = opts[key]
            data = self._send('/responses', payload, on_delta, on_first_output)
            if data.get('error') or data.get('status') in ('failed', 'cancelled'):
                raise ProviderError('模型处理失败，请重试或切换模型。', data.get('_request_timing'))
            if data.get('status') == 'incomplete':
                raise ProviderError('模型输出未完成（可能达到生成 Token 上限）。可在设置中提高上限或降低思考强度。', data.get('_request_timing'))
            output = data.get('output', [])
            history.extend(output)
            calls = [dict(id=x['call_id'], name=x['name'], arguments=x['arguments'])
                     for x in output if x.get('type') == 'function_call']
            text = '\n'.join(c.get('text', c.get('refusal', '')) for x in output
                             if x.get('type') == 'message' for c in x.get('content', [])
                             if c.get('type') in ('output_text', 'refusal'))
            usage = dict(data.get('usage') or {})
            usage['request_timing'] = data.get('_request_timing')
            usage['reasoning_tokens'] = (usage.get('output_tokens_details') or {}).get('reasoning_tokens')
            usage['web_searches'] = [{'id': x.get('id', ''), 'ok': x.get('status') == 'completed'}
                                    for x in output if x.get('type') == 'web_search_call']
            citations = [a for x in output if x.get('type') == 'message' for c in x.get('content', [])
                         for a in c.get('annotations', []) if a.get('type') == 'url_citation']
            usage['sources'] = list({a['url']: {'url': a['url'], 'title': a.get('title') or a['url']}
                for a in citations if isinstance(a.get('url'), str) and urllib.parse.urlsplit(a['url']).scheme in ('https', 'http')}.values())
            return text, calls, usage
        payload = dict(model=self.model, messages=history, max_tokens=opts['max_output_tokens'],
                       tools=[{'type': 'function', 'function': t} for t in tools],
                       tool_choice='auto' if tools or not final else 'none')
        effort = opts['reasoning_effort']
        if effort != 'auto':
            payload['thinking'] = {'type': 'disabled' if effort == 'none' else 'enabled'}
            if effort != 'none': payload['reasoning_effort'] = effort
        for key in ('temperature', 'top_p'):
            if opts[key] is not None: payload[key] = opts[key]
        data = self._send('/chat/completions', payload, on_delta, on_first_output)
        choices = data.get('choices') or []
        if not choices:
            raise ProviderError('API 没有返回候选回答', data.get('_request_timing'))
        choice = choices[0]
        if choice.get('finish_reason') in ('length', 'insufficient_system_resource', 'aborted'):
            raise ProviderError('模型输出未完成，可提高生成 Token 上限、降低思考强度或缩短问题后重试。', data.get('_request_timing'))
        message = choice.get('message', {})
        calls = [dict(id=x['id'], name=x['function']['name'], arguments=x['function']['arguments'])
                 for x in message.get('tool_calls') or []]
        # Preserve reasoning content if a compatible gateway returns it.
        history.append({k: message[k] for k in ('role', 'content', 'tool_calls', 'reasoning_content') if k in message})
        usage = dict(data.get('usage') or {})
        usage['request_timing'] = data.get('_request_timing')
        usage['reasoning_tokens'] = (usage.get('completion_tokens_details') or {}).get('reasoning_tokens')
        return message.get('content') or '', calls, usage

    def tool_result(self, history, call_id, result):
        content = json.dumps(result, ensure_ascii=False)
        if self.provider == 'openai':
            history.append(dict(type='function_call_output', call_id=call_id, output=content))
        else:
            history.append(dict(role='tool', tool_call_id=call_id, content=content))

    def probe(self):
        history = self.start('Reply with OK only.', [{'role': 'user', 'content': 'Hi'}])
        text, calls, _ = self.complete(history, [], final=True, allow_web=False)
        if not text or calls:
            raise ProviderError('连接成功，但模型没有返回文本。请检查模型配置。')
        return {'ok': True, 'message': 'API 连接成功，接口已返回模型文本。'}
