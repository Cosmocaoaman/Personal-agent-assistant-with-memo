"""Incremental Embedding API indexing and bounded hybrid retrieval.

Ranking proposes candidates only. The memory model still decides relevance and
every create/update/delete/skip; no content classifier is introduced here.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict

import numpy as np

from .config import integer
from .providers import ProviderError, post_json, validate_base
from .transport import PooledHTTP
from .retrieval import normalized


EMBEDDING_DEFAULTS = dict(enabled=False, key_source='openai', base_url='https://api.openai.com/v1',
    model='text-embedding-3-small', dimensions=1536, timeout_seconds=20,
    batch_size=16, candidate_limit=24, min_similarity=0.25)


def embedding_options(supplied=None):
    if supplied is not None and (not isinstance(supplied, dict) or set(supplied) - set(EMBEDDING_DEFAULTS)):
        raise ValueError('Embedding 配置包含未知字段或格式错误')
    values = dict(EMBEDDING_DEFAULTS, **(supplied or {}))
    if type(values['enabled']) is not bool or values['key_source'] not in ('openai', 'separate'):
        raise ValueError('Embedding 开关或 Key 来源格式错误')
    values['base_url'] = validate_base(values['base_url'])
    model = values['model']
    if not isinstance(model, str) or not 1 <= len(model.strip()) <= 100:
        raise ValueError('Embedding 模型名称需要 1–100 个字符')
    values['model'] = model.strip()
    for field, lo, hi in [('dimensions', 0, 4096), ('timeout_seconds', 5, 120),
                          ('batch_size', 1, 32), ('candidate_limit', 12, 100)]:
        integer(values[field], lo, hi, 'Embedding ' + field)
    minimum = values['min_similarity']
    if type(minimum) not in (int, float) or not np.isfinite(minimum) or not 0 <= minimum <= 1:
        raise ValueError('语义相似度下限需要 0–1 的数值')
    return values


def fingerprint(config):
    identity = [config['base_url'], config['model'], config['dimensions'], 'key-newline-content-v1']
    return hashlib.sha256(json.dumps(identity).encode()).hexdigest()


class EmbeddingProvider:
    def __init__(self, config, key, transport=post_json):
        self.config = embedding_options(config)
        self.key, self.transport = key, transport
        if not key:
            raise ValueError('Embedding 尚未配置可用 Key')

    def embed(self, inputs):
        # UTF-8 bytes bound token count conservatively; no fact is truncated.
        if not inputs or any(not isinstance(s, str) or not s.strip() or len(s.encode()) > 6000 for s in inputs):
            raise ValueError('Embedding 输入需非空且每条不超过 6,000 UTF-8 字节')
        cfg = self.config
        payload = dict(model=cfg['model'], input=inputs, encoding_format='float')
        if cfg['dimensions']:
            payload['dimensions'] = cfg['dimensions']
        if self.transport is post_json or isinstance(self.transport, PooledHTTP):
            data = self.transport(cfg['base_url'] + '/embeddings', self.key, payload, timeout=cfg['timeout_seconds'])
        else:
            data = self.transport(cfg['base_url'] + '/embeddings', self.key, payload)
        try:
            records = data['data']
            if len(records) != len(inputs) or {r['index'] for r in records} != set(range(len(inputs))):
                raise ValueError()
            records = sorted(records, key=lambda r: r['index'])
            vectors = []
            for record in records:
                if type(record['index']) is not int or not isinstance(record['embedding'], list):
                    raise ValueError()
                if any(type(x) not in (int, float) for x in record['embedding']):
                    raise ValueError()
                vector = np.asarray(record['embedding'], dtype=np.float32)
                if vector.ndim != 1 or not 1 <= len(vector) <= 4096 or not np.isfinite(vector).all():
                    raise ValueError()
                if cfg['dimensions'] and len(vector) != cfg['dimensions']:
                    raise ValueError()
                norm = np.linalg.norm(vector)
                if not np.isfinite(norm) or norm <= 0 or (vectors and len(vector) != len(vectors[0])):
                    raise ValueError()
                vectors.append(vector / norm)
            usage = data.get('usage') or {}
            tokens = usage.get('total_tokens', 0)
            return vectors, tokens if type(tokens) is int and tokens >= 0 else 0
        except (KeyError, TypeError, ValueError, OverflowError):
            raise ProviderError('Embedding 响应的向量数量、维度或数值不正确') from None

    def probe(self):
        start = time.perf_counter()
        vectors, tokens = self.embed(['Mori embedding connection check'])
        return dict(ok=True, model=self.config['model'], dimensions=len(vectors[0]), tokens=tokens,
                    elapsed_ms=round((time.perf_counter()-start)*1000, 1), message='Embedding 连接成功，模型已返回有效向量。')


class HybridRetriever:
    def __init__(self, store, provider_factory, config_factory):
        self.store, self.provider_factory, self.config_factory = store, provider_factory, config_factory
        self._lock = threading.RLock()
        self._worker = None
        self._failed_at = 0
        self._job = dict(state='idle', processed=0, tokens=0, error='')
        self._cache_identity = None
        self._documents, self._matrix = [], np.empty((0, 0), dtype=np.float32)
        self._queries = OrderedDict()

    def status(self):
        cfg = self.config_factory()
        with self._lock:
            job = dict(self._job)
        return dict(self.store.vector_status(fingerprint(cfg)), **job,
                    enabled=cfg['enabled'], model=cfg['model'], dimensions=cfg['dimensions'], keyword_engine='SQLite FTS5 / BM25')

    def schedule(self, force=False):
        if not self.config_factory()['enabled']:
            return self.status()
        with self._lock:
            if self._worker and self._worker.is_alive():
                return self.status()
            if not force and time.monotonic()-self._failed_at < 30:
                return self.status()
            self._job = dict(state='running', processed=0, tokens=0, error='')
            self._worker = threading.Thread(target=self._index, name='mori-embedding-index', daemon=True)
            self._worker.start()
        return self.status()

    def _index(self):
        try:
            while True:
                cfg = self.config_factory()
                if not cfg['enabled']:
                    break
                identity = fingerprint(cfg)
                docs = self.store.pending_vectors(identity, cfg['batch_size'])
                if not docs:
                    break
                provider = self.provider_factory()
                if fingerprint(provider.config) != identity:
                    continue  # Settings changed between selecting docs and obtaining the adapter.
                vectors, tokens = provider.embed([d['document'] for d in docs])
                self.store.save_vectors(identity, docs, vectors)
                with self._lock:
                    self._job['processed'] += len(docs)
                    self._job['tokens'] += tokens
            with self._lock:
                self._job['state'] = 'complete'
        except Exception as exc:
            with self._lock:
                self._failed_at = time.monotonic()
                self._job.update(state='error', error=str(exc) if isinstance(exc, (ProviderError, ValueError)) else '语义索引构建失败')

    def _snapshot(self, identity):
        revisions = self.store.index_revisions()
        with self._lock:
            if self._cache_identity != (identity, revisions):
                revisions, docs = self.store.vector_snapshot(identity)
                vectors = [np.frombuffer(d['vector'], dtype='<f4') for d in docs]
                if vectors and any(len(v) != len(vectors[0]) for v in vectors):
                    raise ProviderError('本地向量维度不一致，请更新语义索引')
                self._documents = docs
                self._matrix = np.stack(vectors) if vectors else np.empty((0, 0), dtype=np.float32)
                self._cache_identity = (identity, revisions)
            return self._documents, self._matrix

    def search(self, query, limit=12):
        start = time.perf_counter()
        cfg = self.config_factory()
        lexical = self.store.search(query, cfg['candidate_limit'])
        detail = dict(mode='keyword', keyword_engine='FTS5 / BM25', lexical_count=len(lexical),
                      semantic_count=0, embedding_ms=0, embedding_tokens=0, query_cache_hit=False,
                      keyword_ms=round((time.perf_counter()-start)*1000, 3), fallback='')
        if not cfg['enabled'] or not query.strip():
            detail['fallback'] = 'Embedding 未启用' if not cfg['enabled'] else '空查询只返回最近记忆'
            return lexical[:limit], detail
        identity = fingerprint(cfg)
        detail.update(self.store.vector_status(identity))
        if detail['pending']:
            self.schedule()
        try:
            provider = self.provider_factory()
            if fingerprint(provider.config) != identity:
                raise ProviderError('Embedding 配置在查询期间变化，请重新查询')
            docs, matrix = self._snapshot(identity)
            if not docs:
                detail['fallback'] = '语义索引尚未就绪，当前使用关键词检索'
                return lexical[:limit], detail
            cache_key = (identity, normalized(query))
            with self._lock:
                cached = self._queries.get(cache_key)
                if cached is not None:
                    self._queries.move_to_end(cache_key)
            if cached is None:
                api_start = time.perf_counter()
                try:
                    vectors, tokens = provider.embed([query])
                finally:
                    detail['embedding_ms'] = round((time.perf_counter()-api_start)*1000, 3)
                detail['embedding_tokens'] = tokens
                cached = vectors[0]
                with self._lock:
                    self._queries[cache_key] = cached
                    if len(self._queries) > 256:
                        self._queries.popitem(last=False)
            else:
                detail['query_cache_hit'] = True
            if matrix.shape[1] != len(cached):
                raise ProviderError('查询向量与本地索引维度不同，请检查模型配置并更新索引')
            local_start = time.perf_counter()
            similarities = np.einsum('ij,j->i', matrix, cached)
            ranks = np.argsort(-similarities, kind='stable')[:cfg['candidate_limit']]
            candidates = [docs[i] for i in ranks if similarities[i] >= cfg['min_similarity']]
            # Hydrate only bounded IDs; reject vectors whose document changed
            # during the network call, including edits by another DB client.
            fresh = self.store.memories_by_ids([d['memory_id'] for d in candidates])
            semantic = [fresh[d['memory_id']] for d in candidates if d['memory_id'] in fresh
                        and fresh[d['memory_id']]['key']+'\n'+fresh[d['memory_id']]['content'] == d['document']]
            scores, rows = {}, {}
            for ranking in (lexical, semantic):
                for rank, row in enumerate(ranking, 1):
                    rows[row['id']] = row
                    scores[row['id']] = scores.get(row['id'], 0) + 1/(60+rank)
            ordered = sorted(scores, key=lambda mid: -scores[mid])
            detail.update(mode='hybrid', semantic_count=len(semantic),
                          vector_ms=round((time.perf_counter()-local_start)*1000, 3),
                          top_similarity=round(float(similarities[ranks[0]]), 4) if len(ranks) else None,
                          fusion='RRF', fallback='语义索引尚有未处理条目，关键词覆盖全库' if detail['pending'] else '')
            return [rows[mid] for mid in ordered[:limit]], detail
        except (ProviderError, ValueError) as exc:
            detail['fallback'] = str(exc)
            return lexical[:limit], detail
