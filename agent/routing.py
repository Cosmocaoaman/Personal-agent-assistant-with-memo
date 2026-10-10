"""Accept bounded context from a future external router, without domain gates."""
import json


def router_context(supplied=None):
    if supplied is None:
        return {'route': 'daily', 'context': {}}
    if not isinstance(supplied, dict) or set(supplied) - {'route', 'context'}:
        raise ValueError('路由上下文格式错误')
    value = dict(route=supplied.get('route', 'daily'), context=supplied.get('context', {}))
    if value['route'] not in ('daily', 'home') or not isinstance(value['context'], dict):
        raise ValueError('路由需为 daily 或 home，context 需为 JSON 对象')
    try:
        text = json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (ValueError, TypeError, RecursionError):
        raise ValueError('路由上下文需为有效 JSON 数据') from None
    if len(text) > 8000:
        raise ValueError('路由上下文最多 8,000 JSON 字符')
    return json.loads(text)
