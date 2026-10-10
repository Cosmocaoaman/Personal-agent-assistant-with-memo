"""Assemble complete API results; only public text deltas leave this module."""
from .transport import ProviderError, output_event


class StreamResult:
    def __init__(self, provider, on_delta=None, on_first_output=None):
        self.provider, self.on_delta, self.on_first_output = provider, on_delta, on_first_output
        self.first = False
        self.response = None
        self.message = {'role': 'assistant', 'content': ''}
        self.calls, self.usage, self.finish_reason = {}, {}, None

    def __call__(self, event):
        if output_event(event) and not self.first:
            self.first = True
            if self.on_first_output:
                self.on_first_output()
        if self.provider == 'openai':
            kind = event.get('type')
            if kind in ('response.output_text.delta', 'response.refusal.delta') and self.on_delta:
                self.on_delta(event.get('delta') or '')
            elif kind == 'response.completed':
                self.response = event.get('response')
            elif kind == 'response.incomplete':
                raise ProviderError('模型输出未完成，可提高生成 Token 上限或降低思考强度后重试。')
            elif kind in ('error', 'response.failed', 'response.cancelled'):
                raise ProviderError('模型处理失败或流式响应已中断，请重试。')
            return
        if event.get('error'):
            raise ProviderError('模型处理失败，请重试。')
        if event.get('usage'):
            self.usage = event['usage']
        for choice in event.get('choices') or []:
            if choice.get('index', 0) != 0:
                continue
            delta = choice.get('delta') or {}
            text = delta.get('content') or ''
            self.message['content'] += text
            if text and self.on_delta:
                self.on_delta(text)
            # Hidden reasoning is replayed to the provider only, never emitted.
            if delta.get('reasoning_content'):
                self.message['reasoning_content'] = self.message.get('reasoning_content', '') + delta['reasoning_content']
            for part in delta.get('tool_calls') or []:
                index = part['index']
                call = self.calls.setdefault(index, {'id': '', 'type': 'function', 'function': {'name': '', 'arguments': ''}})
                if part.get('id'):
                    call['id'] = part['id']
                function = part.get('function') or {}
                call['function']['name'] += function.get('name') or ''
                call['function']['arguments'] += function.get('arguments') or ''
            if choice.get('finish_reason') is not None:
                self.finish_reason = choice['finish_reason']

    def finish(self, done):
        if self.provider == 'openai':
            if not isinstance(self.response, dict) or self.response.get('status') != 'completed':
                raise ProviderError('API 流式响应未完整结束，请重试。')
            return self.response
        if not done or self.finish_reason is None:
            raise ProviderError('API 流式响应未完整结束，请重试。')
        if self.calls:
            calls = [self.calls[k] for k in sorted(self.calls)]
            if any(not c['id'] or not c['function']['name'] for c in calls):
                raise ProviderError('API 返回了不完整的工具调用。')
            self.message['tool_calls'] = calls
        return {'choices': [{'message': self.message, 'finish_reason': self.finish_reason}], 'usage': self.usage}
