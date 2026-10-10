"""Validated, non-secret settings shared by the UI and actual execution."""
from __future__ import annotations

import math

MODEL_OPTIONS = dict(reasoning_effort='auto', max_output_tokens=4096, timeout_seconds=90,
                     temperature=None, top_p=None, verbosity='auto', web_search=False,
                     search_context_size='medium')
RUNTIME_DEFAULTS = dict(history_messages=30, history_chars=24000, personal_items=24,
                        personal_chars=12000, memory_steps=6, tool_calls_per_step=4,
                        memory_read_chars=24000, handoff_chars=16000,
                        recall_mode='specialist', prefetch_candidates=True, handoff_summary_chars=240)
EFFORTS = {'deepseek': ['auto', 'none', 'low', 'high', 'max'],
           'openai': ['auto', 'none', 'minimal', 'low', 'medium', 'high', 'xhigh']}


def model_defaults(provider):
    return dict(MODEL_OPTIONS, reasoning_effort='none' if provider == 'deepseek' else 'auto')


def integer(value, minimum, maximum, label):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f'{label}需要 {minimum}–{maximum} 的整数')


def model_options(provider, supplied=None):
    if supplied is not None and (not isinstance(supplied, dict) or set(supplied) - set(MODEL_OPTIONS)):
        raise ValueError('模型参数包含未知字段或格式错误')
    values = dict(model_defaults(provider), **(supplied or {}))
    if values['reasoning_effort'] not in EFFORTS[provider]:
        raise ValueError('该服务商不支持所选思考强度')
    integer(values['max_output_tokens'], 256, 131072, '生成 Token 上限')
    integer(values['timeout_seconds'], 10, 300, '单次请求超时秒数')
    for name, lo, hi in [('temperature', 0, 2), ('top_p', 0.01, 1)]:
        value = values[name]
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or not lo <= value <= hi):
            raise ValueError(f'{name} 需要 {lo}–{hi} 的数值，或留空')
    if values['verbosity'] not in ('auto', 'low', 'medium', 'high'):
        raise ValueError('回答详略格式错误')
    if type(values['web_search']) is not bool:
        raise ValueError('联网搜索开关格式错误')
    if values['search_context_size'] not in ('low', 'medium', 'high'):
        raise ValueError('搜索上下文大小格式错误')
    if provider == 'deepseek':
        if values['web_search']:
            raise ValueError('当前 DeepSeek 适配器尚未接入搜索；可使用 OpenAI 内置联网搜索')
        if values['verbosity'] != 'auto':
            raise ValueError('DeepSeek 不支持 API 回答详略参数，请使用回答风格')
        if values['reasoning_effort'] != 'none' and values['temperature'] is not None:
            raise ValueError('DeepSeek 思考模式下 temperature 无效，请留空')
        if values['top_p'] is not None and (values['reasoning_effort'] == 'none' or values['top_p'] < 0.95):
            raise ValueError('DeepSeek Top-P 仅在思考模式生效，范围为 0.95–1')
    return values


def runtime_options(supplied=None):
    if supplied is not None and (not isinstance(supplied, dict) or set(supplied) - set(RUNTIME_DEFAULTS)):
        raise ValueError('Agent 参数包含未知字段或格式错误')
    values = dict(RUNTIME_DEFAULTS, **(supplied or {}))
    if values['recall_mode'] not in ('specialist', 'main'):
        raise ValueError('上下文模式需要 specialist 或 main')
    if type(values['prefetch_candidates']) is not bool:
        raise ValueError('候选预取开关格式错误')
    for key, lo, hi, label in [('history_messages', 0, 200, '近期消息条数'),
        ('history_chars', 1000, 200000, '近期上下文字符上限'), ('personal_items', 0, 100, '基础记忆条数'),
        ('personal_chars', 1000, 50000, '基础记忆字符上限'), ('memory_steps', 1, 12, '每阶段模型调用上限'),
        ('tool_calls_per_step', 1, 4, '单次工具请求上限'),
        ('memory_read_chars', 2000, 200000, '每阶段读入记忆字符上限'),
        ('handoff_chars', 1000, 50000, '交接原始记忆字符上限'),
        ('handoff_summary_chars', 64, 1000, '交接备注字符上限')]:
        integer(values[key], lo, hi, label)
    return values
