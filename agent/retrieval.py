"""Reusable lexical index. Ranking is identical to the original BM25 scan.

This is a search mechanism, never a classifier deciding what to remember.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict


def normalized(value):
    return re.sub(r'\s+', ' ', value.strip()).casefold()


def tokens(text):
    result = re.findall(r'[a-z0-9_]+', text.casefold())
    for run in re.findall(r'[\u3400-\u9fff]+', text):
        result.extend(run)
        result.extend(run[i:i + 2] for i in range(len(run) - 1))
    return result


def fts_terms(text):
    # Hex tokens keep CJK bigrams and underscores intact in unicode61, and
    # ensure that user text can never become an FTS query operator.
    return ['t' + term.encode('utf-8').hex() for term in tokens(text)]


class LexicalIndex:
    def __init__(self, items):
        self.items = [dict(item) for item in items]
        self.contents = [normalized(item['content']) for item in items]
        docs = [Counter(tokens(r['content'] + ' ' + r['key'])) for r in items]
        average = sum(sum(d.values()) for d in docs) / len(docs) if docs else 1
        average = average or 1
        df = Counter(t for d in docs for t in d)
        self.postings = defaultdict(list)
        self.context_boosts = {}
        for i, (item, doc) in enumerate(zip(items, docs)):
            length = sum(doc.values())
            norm = 1.2 * (.25 + .75 * length / average)
            for term, count in doc.items():
                weight = math.log(1 + (len(docs) - df[term] + .5) / (df[term] + .5)) * (count * 2.2) / (count + norm)
                self.postings[term].append((i, weight))
            if item['pinned']:
                self.context_boosts[i] = 8
            elif item['category'] in ('profile', 'preference'):
                self.context_boosts[i] = 2

    def search(self, query, limit=8, context=False):
        if not query.strip():
            return [dict(item) for item in self.items[:limit]]
        q = set(tokens(query))
        scores = {}
        for term in q:
            for i, weight in self.postings.get(term, ()):
                scores[i] = scores.get(i, 0) + weight
        # A phrase containing only punctuation/emoji still matched in the old
        # implementation. Preserve that edge case, as well as stable ties.
        phrase = normalized(query)
        for i, content in enumerate(self.contents):
            if phrase in content:
                scores[i] = scores.get(i, 0) + 8
        if context:
            for i, boost in self.context_boosts.items():
                scores[i] = scores.get(i, 0) + boost
        ranked = sorted((i for i, score in scores.items() if score > 0), key=lambda i: (-scores[i], i))
        return [dict(self.items[i]) for i in ranked[:limit]]
