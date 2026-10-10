"""SQLite persistence and multilingual lexical retrieval. No external services."""
from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .retrieval import LexicalIndex, normalized, fts_terms, tokens


def now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def uid():
    return uuid.uuid4().hex


def context_memory(item):
    """Minimal model-facing memory, without source messages or evidence copies."""
    return {key: item[key] for key in ('id', 'key', 'content', 'category', 'pinned', 'updated_at')}


def pack_memories(items, char_budget, limit=None, prefix=False):
    """Bound complete model-facing records, never slice a fact or classify it."""
    selected, used = [], 2
    for item in items:
        if limit is not None and len(selected) >= limit:
            break
        size = len(json.dumps(context_memory(item), ensure_ascii=False)) + (2 if selected else 0)
        if used + size > char_budget:
            if prefix:
                break
            continue
        selected.append(item)
        used += size
    return selected, used if selected else 0


class Store:
    def __init__(self, file):
        self.file = Path(file)
        self._search_lock = threading.Lock()
        self.file.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY, title TEXT NOT NULL,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    role TEXT NOT NULL, content TEXT NOT NULL, meta TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS messages_session ON messages(session_id, created_at);
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY, key TEXT UNIQUE NOT NULL, content TEXT NOT NULL,
                    category TEXT NOT NULL, pinned INTEGER NOT NULL DEFAULT 0,
                    source TEXT NOT NULL, evidence TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY, title TEXT NOT NULL, done INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS credentials (
                    provider TEXT PRIMARY KEY, base_url TEXT NOT NULL, api_key TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    status TEXT NOT NULL, preview TEXT NOT NULL, started_at TEXT NOT NULL,
                    elapsed_ms INTEGER NOT NULL, trace TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS runs_started ON runs(started_at);
                CREATE TABLE IF NOT EXISTS index_revision (
                    name TEXT PRIMARY KEY, value INTEGER NOT NULL DEFAULT 0);
                INSERT OR IGNORE INTO index_revision VALUES ('memories',0),('vectors',0);
                CREATE TABLE IF NOT EXISTS search_changes (id TEXT PRIMARY KEY);
                CREATE TABLE IF NOT EXISTS memory_fts_ids (id TEXT PRIMARY KEY, fts_rowid INTEGER UNIQUE NOT NULL);
                CREATE TRIGGER IF NOT EXISTS search_insert AFTER INSERT ON memories BEGIN
                    INSERT INTO search_changes SELECT new.id WHERE NOT EXISTS (SELECT 1 FROM search_changes WHERE id=new.id);
                    UPDATE index_revision SET value=value+1 WHERE name='memories';
                END;
                CREATE TRIGGER IF NOT EXISTS search_update AFTER UPDATE ON memories BEGIN
                    INSERT INTO search_changes SELECT old.id WHERE NOT EXISTS (SELECT 1 FROM search_changes WHERE id=old.id);
                    INSERT INTO search_changes SELECT new.id WHERE NOT EXISTS (SELECT 1 FROM search_changes WHERE id=new.id);
                    UPDATE index_revision SET value=value+1 WHERE name='memories';
                END;
                CREATE TRIGGER IF NOT EXISTS search_delete AFTER DELETE ON memories BEGIN
                    INSERT INTO search_changes SELECT old.id WHERE NOT EXISTS (SELECT 1 FROM search_changes WHERE id=old.id);
                    UPDATE index_revision SET value=value+1 WHERE name='memories';
                END;
                CREATE TABLE IF NOT EXISTS memory_vectors (
                    memory_id TEXT PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
                    fingerprint TEXT NOT NULL, document TEXT NOT NULL,
                    dimensions INTEGER NOT NULL, vector BLOB NOT NULL);
                CREATE TRIGGER IF NOT EXISTS vector_insert AFTER INSERT ON memory_vectors BEGIN
                    UPDATE index_revision SET value=value+1 WHERE name='vectors';
                END;
                CREATE TRIGGER IF NOT EXISTS vector_update AFTER UPDATE ON memory_vectors BEGIN
                    UPDATE index_revision SET value=value+1 WHERE name='vectors';
                END;
                CREATE TRIGGER IF NOT EXISTS vector_delete AFTER DELETE ON memory_vectors BEGIN
                    UPDATE index_revision SET value=value+1 WHERE name='vectors';
                END;
                CREATE INDEX IF NOT EXISTS memories_personal ON memories(pinned DESC,updated_at DESC);
            """)
            exists = db.execute("SELECT 1 FROM sqlite_master WHERE name='memory_fts'").fetchone()
            db.execute('CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(memory_id UNINDEXED, terms)')
            if not exists:
                db.execute('INSERT OR IGNORE INTO search_changes SELECT id FROM memories')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.file, timeout=15)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def settings(self):
        with self.connect() as db:
            return {r['key']: json.loads(r['value']) for r in db.execute("SELECT * FROM settings")}

    def credential(self, provider):
        with self.connect() as db:
            row = db.execute("SELECT * FROM credentials WHERE provider=?", (provider,)).fetchone()
            return dict(row) if row else None

    def set_settings(self, settings, credential=None):
        with self.connect() as db:
            db.executemany("INSERT INTO settings VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                           [(k, json.dumps(v, ensure_ascii=False)) for k, v in settings.items()])
            for item in ([] if credential is None else [credential] if isinstance(credential, dict) else credential):
                db.execute("""INSERT INTO credentials VALUES (:provider,:base_url,:api_key)
                    ON CONFLICT(provider) DO UPDATE SET
                    base_url=excluded.base_url, api_key=excluded.api_key""", item)

    def sessions(self):
        with self.connect() as db:
            return [dict(r) for r in db.execute("SELECT * FROM sessions ORDER BY updated_at DESC")]

    def session(self, session_id):
        with self.connect() as db:
            r = db.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
            if not r:
                raise ValueError("会话不存在")
            return dict(r)

    def create_session(self, title="新的对话"):
        item = dict(id=uid(), title=title, created_at=now(), updated_at=now())
        with self.connect() as db:
            db.execute("INSERT INTO sessions VALUES (:id,:title,:created_at,:updated_at)", item)
        return item

    def delete_session(self, session_id):
        with self.connect() as db:
            db.execute("DELETE FROM sessions WHERE id=?", (session_id,))

    def messages(self, session_id):
        with self.connect() as db:
            result = [dict(r) for r in db.execute("SELECT * FROM messages WHERE session_id=? ORDER BY rowid", (session_id,))]
        for r in result:
            r['meta'] = json.loads(r['meta'])
        return result

    def add_message(self, session_id, role, content, meta=None):
        self.session(session_id)
        item = dict(id=uid(), session_id=session_id, role=role, content=content,
                    meta=json.dumps(meta or {}, ensure_ascii=False), created_at=now())
        with self.connect() as db:
            db.execute("INSERT INTO messages VALUES (:id,:session_id,:role,:content,:meta,:created_at)", item)
            count = db.execute("SELECT COUNT(*) FROM messages WHERE session_id=? AND role='user'", (session_id,)).fetchone()[0]
            if count == 1 and role == 'user':
                db.execute("UPDATE sessions SET title=? WHERE id=?", (content[:28], session_id))
            db.execute("UPDATE sessions SET updated_at=? WHERE id=?", (now(), session_id))
        item['meta'] = meta or {}
        return item

    def memories(self):
        with self.connect() as db:
            return [dict(r) for r in db.execute("SELECT * FROM memories ORDER BY pinned DESC, updated_at DESC")]

    def personal_context(self, limit=24, char_budget=12000):
        """Always-on personal context, bounded without cutting facts mid-sentence.

        This is a read policy requested by the user, never a gate on memory writes.
        Refresh from the current table each turn so edits and deletions take effect.
        """
        selected, used = [], 2
        with self.connect() as db:
            db.execute('BEGIN')
            total = db.execute("SELECT COUNT(*) FROM memories WHERE category IN ('profile','preference') OR pinned=1").fetchone()[0]
            rows = db.execute("""SELECT * FROM memories WHERE category IN ('profile','preference') OR pinned=1
                ORDER BY pinned DESC, updated_at DESC, id""")
            for row in rows:
                if len(selected) >= limit:
                    break
                item = dict(row)
                size = len(json.dumps(context_memory(item), ensure_ascii=False)) + (2 if selected else 0)
                if len(selected) < limit and used + size <= char_budget:
                    selected.append(item)
                    used += size
        return {'memories': selected, 'total': total, 'omitted': total - len(selected)}

    def memory_page(self, offset=0, limit=20):
        with self.connect() as db:
            rows = [dict(r) for r in db.execute(
                'SELECT * FROM memories ORDER BY pinned DESC, updated_at DESC, id LIMIT ? OFFSET ?', (limit + 1, offset))]
        return rows[:limit], len(rows) > limit

    def update_message_meta(self, message_id, meta):
        with self.connect() as db:
            db.execute('UPDATE messages SET meta=? WHERE id=?', (json.dumps(meta, ensure_ascii=False), message_id))
            row = db.execute('SELECT * FROM messages WHERE id=?', (message_id,)).fetchone()
            if not row:
                raise ValueError('消息不存在')
        item = dict(row)
        item['meta'] = meta
        return item

    def save_run(self, trace):
        with self.connect() as db:
            db.execute('''INSERT INTO runs VALUES (?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET
                status=excluded.status, elapsed_ms=excluded.elapsed_ms, trace=excluded.trace''',
                (trace['id'], trace['session_id'], trace['status'], trace['preview'], trace['started_at'],
                 trace['elapsed_ms'], json.dumps(trace, ensure_ascii=False)))

    def runs(self, session_id=None, limit=50):
        query = 'SELECT id,session_id,status,preview,started_at,elapsed_ms FROM runs'
        args = []
        if session_id:
            query += ' WHERE session_id=?'
            args.append(session_id)
        with self.connect() as db:
            return [dict(r) for r in db.execute(query + ' ORDER BY started_at DESC, rowid DESC LIMIT ?', args + [limit])]

    def run(self, run_id):
        with self.connect() as db:
            row = db.execute('SELECT trace FROM runs WHERE id=?', (run_id,)).fetchone()
        if not row:
            raise ValueError('流程记录不存在，可能已随会话删除')
        return json.loads(row['trace'])

    def recover_runs(self):
        """Only at server startup: never pretend an interrupted node completed."""
        with self.connect() as db:
            rows = db.execute("SELECT id,trace FROM runs WHERE status='running'").fetchall()
            for row in rows:
                trace = json.loads(row['trace'])
                trace['status'] = 'interrupted'
                for node, item in trace['nodes'].items():
                    if item['status'] not in ('running', 'waiting'):
                        continue
                    patch = {'status': 'interrupted' if item['status'] == 'running' else 'cancelled'}
                    item.update(patch)
                    trace['events'].append(dict(seq=len(trace['events']) + 1, kind='node_interrupted', node=node,
                        elapsed_ms=trace['elapsed_ms'], patch=patch, message='服务重启，未记录到节点完成'))
                trace['events'].append(dict(seq=len(trace['events']) + 1, kind='run_end', node=None,
                    elapsed_ms=trace['elapsed_ms'], status='interrupted', message='上次运行已中断，耗时为末次记录值'))
                db.execute("UPDATE runs SET status='interrupted',trace=? WHERE id=?", (json.dumps(trace, ensure_ascii=False), row['id']))

    def apply_memory_plan(self, operations, source, seen):
        """Atomic model-selected changes. No content/importance classifier here.

        Updates/deletes must reference a row the child read, with its original
        version. Retrying a stale plan cannot overwrite a manual edit.
        """
        changes = []
        touched = set()
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            for op in operations:
                action = op['operation']
                mid = op['id']
                old = None
                if action not in ('create', 'update', 'delete'):
                    raise ValueError('未知记忆操作')
                if action != 'create':
                    old = db.execute('SELECT * FROM memories WHERE id=?', (mid,)).fetchone()
                    if mid not in seen or not old:
                        raise ValueError('修改或删除前需要读取现有记忆 ID')
                    if dict(old) != seen[mid]:
                        raise ValueError('记忆已变化，请重新查询再提交')
                    if mid in touched:
                        raise ValueError('同一个 ID 在一个事务中只能修改一次')
                    touched.add(mid)
                elif mid:
                    raise ValueError('新建记忆的 ID 必须为空')
                if action == 'delete':
                    db.execute('DELETE FROM memories WHERE id=?', (mid,))
                    changes.append({'operation': action, 'id': mid, 'key': old['key']})
                    continue
                content, key = op['content'].strip(), op['key'].strip()
                if not 1 <= len(content) <= 1500 or not 1 <= len(key) <= 100 or len(op['evidence']) > 2000:
                    raise ValueError('内容需 1–1500 字符，主题标识需 1–100 字符，依据最多 2000 字符')
                if op['category'] not in ('profile', 'preference', 'project', 'note'):
                    raise ValueError('记忆分类不支持')
                conflict = db.execute('SELECT * FROM memories WHERE key=?', (key,)).fetchone()
                if action == 'create' and conflict and conflict['content'] == content:
                    continue  # Exact idempotence, not a judgment of memory value.
                if conflict and (not old or conflict['id'] != old['id']):
                    raise ValueError('该主题已存在，请查询真实 ID 后改为更新')
                stamp = now()
                if action == 'create':
                    mid = uid()
                    db.execute('INSERT INTO memories VALUES (?,?,?,?,?,?,?,?,?)',
                               (mid, key, content, op['category'], 0, source, op['evidence'], stamp, stamp))
                else:
                    db.execute('UPDATE memories SET key=?,content=?,category=?,source=?,evidence=?,updated_at=? WHERE id=?',
                               (key, content, op['category'], source, op['evidence'], stamp, mid))
                changes.append({'operation': action, 'id': mid, 'key': key})
        return changes

    def save_memory(self, content, category="note", key="", pinned=False, source="manual", evidence="", memory_id=None):
        if not isinstance(content, str) or not 1 <= len(content.strip()) <= 1500:
            raise ValueError("记忆需要 1–1500 个字符")
        if category not in ('profile', 'preference', 'project', 'note'):
            raise ValueError("未知记忆分类")
        if not isinstance(key, str) or len(key) > 100:
            raise ValueError("记忆标识不能超过 100 个字符")
        content = content.strip()
        stamp = now()
        with self.connect() as db:
            old = None
            if memory_id:
                old = db.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
                if not old:
                    raise ValueError("记忆不存在")
            elif key.strip():
                old = db.execute("SELECT * FROM memories WHERE key=?", (key.strip(),)).fetchone()
            if not old and not memory_id:
                old = next((r for r in db.execute("SELECT * FROM memories") if normalized(r['content']) == normalized(content)), None)
            item = dict(id=old['id'] if old else uid(), key=key.strip() or (old['key'] if old else uid()),
                        content=content, category=category, pinned=int(pinned), source=source, evidence=evidence,
                        created_at=old['created_at'] if old else stamp, updated_at=stamp)
            # An automatic refresh should preserve a user pin.
            if old and source != 'manual':
                item['pinned'] = old['pinned']
            try:
                db.execute("""INSERT INTO memories VALUES (:id,:key,:content,:category,:pinned,:source,:evidence,:created_at,:updated_at)
                    ON CONFLICT(id) DO UPDATE SET key=excluded.key, content=excluded.content, category=excluded.category,
                    pinned=excluded.pinned, source=excluded.source, evidence=excluded.evidence, updated_at=excluded.updated_at""", item)
            except sqlite3.IntegrityError as exc:
                raise ValueError("这个记忆标识已经存在，请编辑原记忆或使用其他标识") from exc
        return item

    def delete_memory(self, memory_id):
        with self.connect() as db:
            db.execute("DELETE FROM memories WHERE id=?", (memory_id,))

    def search(self, query, limit=8, context=False):
        """Persistent FTS5 BM25. Triggers also capture external SQL changes.

        Only dirty documents are tokenized. Ordinary queries read the index and
        a bounded set of rows, rather than taking a full database snapshot.
        The legacy context-boost mode is retained for compatibility, unused by
        the agent (personal context has its own bounded SQL read).
        """
        if context:
            return LexicalIndex(self.memories()).search(query, limit, True)
        with self._search_lock:
            with self.connect() as db:
                db.execute('BEGIN')
                if db.execute('SELECT 1 FROM search_changes LIMIT 1').fetchone():
                    db.rollback()
                    db.execute('BEGIN IMMEDIATE')
                    changes = db.execute('''SELECT c.id,m.key,m.content FROM search_changes c
                        LEFT JOIN memories m ON m.id=c.id''').fetchall()
                    for row in changes:
                        old = db.execute('SELECT fts_rowid FROM memory_fts_ids WHERE id=?', (row['id'],)).fetchone()
                        if old:
                            db.execute('DELETE FROM memory_fts WHERE rowid=?', (old[0],))
                            db.execute('DELETE FROM memory_fts_ids WHERE id=?', (row['id'],))
                        if row['content'] is not None:
                            cursor = db.execute('INSERT INTO memory_fts(memory_id,terms) VALUES (?,?)',
                                                (row['id'], ' '.join(fts_terms(row['content'] + ' ' + row['key']))))
                            db.execute('INSERT INTO memory_fts_ids VALUES (?,?)', (row['id'], cursor.lastrowid))
                    db.execute('DELETE FROM search_changes')
                if not query.strip():
                    rows = db.execute('SELECT * FROM memories ORDER BY pinned DESC,updated_at DESC,id LIMIT ?', (limit,))
                else:
                    terms = list(dict.fromkeys(fts_terms(query)))[:128]
                    if terms:
                        rows = db.execute('''SELECT m.* FROM memory_fts JOIN memories m ON m.id=memory_fts.memory_id
                            WHERE memory_fts MATCH ? ORDER BY
                            CASE WHEN instr(lower(m.content),?)>0 THEN 0 ELSE 1 END,
                            bm25(memory_fts),m.pinned DESC,m.updated_at DESC,m.id LIMIT ?''',
                            (' OR '.join(terms), normalized(query), limit))
                    else:
                        rows = db.execute('''SELECT * FROM memories WHERE instr(lower(content),?)>0
                            ORDER BY pinned DESC,updated_at DESC,id LIMIT ?''', (normalized(query), limit))
                return [dict(row) for row in rows]

    def index_revisions(self):
        with self.connect() as db:
            return tuple(row[0] for row in db.execute('SELECT value FROM index_revision ORDER BY name'))

    def vector_status(self, fingerprint):
        with self.connect() as db:
            total = db.execute('SELECT COUNT(*) FROM memories').fetchone()[0]
            indexed = db.execute('''SELECT COUNT(*) FROM memories m JOIN memory_vectors v ON v.memory_id=m.id
                WHERE v.fingerprint=? AND v.document=m.key||char(10)||m.content''', (fingerprint,)).fetchone()[0]
        return dict(total=total, indexed=indexed, pending=total-indexed)

    def pending_vectors(self, fingerprint, limit=16):
        with self.connect() as db:
            return [dict(r) for r in db.execute('''SELECT m.id,m.key||char(10)||m.content AS document
                FROM memories m LEFT JOIN memory_vectors v ON v.memory_id=m.id
                WHERE v.memory_id IS NULL OR v.fingerprint<>? OR v.document<>m.key||char(10)||m.content
                ORDER BY m.pinned DESC,m.updated_at DESC,m.id LIMIT ?''', (fingerprint, limit))]

    def save_vectors(self, fingerprint, documents, vectors):
        with self.connect() as db:
            for doc, vector in zip(documents, vectors):
                # Never attach an embedding to a document edited during the API call.
                db.execute('''INSERT INTO memory_vectors(memory_id,fingerprint,document,dimensions,vector)
                    SELECT id,?,?,?,? FROM memories WHERE id=? AND key||char(10)||content=?
                    ON CONFLICT(memory_id) DO UPDATE SET fingerprint=excluded.fingerprint,
                    document=excluded.document,dimensions=excluded.dimensions,vector=excluded.vector''',
                    (fingerprint, doc['document'], len(vector), vector.astype('<f4').tobytes(), doc['id'], doc['document']))

    def vector_snapshot(self, fingerprint):
        with self.connect() as db:
            db.execute('BEGIN')
            revisions = tuple(r[0] for r in db.execute('SELECT value FROM index_revision ORDER BY name'))
            rows = [dict(r) for r in db.execute('''SELECT v.* FROM memory_vectors v JOIN memories m ON m.id=v.memory_id
                WHERE v.fingerprint=? AND v.document=m.key||char(10)||m.content ORDER BY v.memory_id''', (fingerprint,))]
        return revisions, rows

    def memories_by_ids(self, ids):
        if not ids:
            return {}
        with self.connect() as db:
            rows = db.execute('SELECT * FROM memories WHERE id IN (' + ','.join('?' for _ in ids) + ')', ids)
            return {r['id']: dict(r) for r in rows}

    def tasks(self):
        with self.connect() as db:
            return [dict(r) for r in db.execute("SELECT * FROM tasks ORDER BY done, created_at DESC")]

    def add_task(self, title):
        if not isinstance(title, str) or not 1 <= len(title.strip()) <= 300:
            raise ValueError("待办需要 1–300 个字符")
        item = dict(id=uid(), title=title.strip(), done=0, created_at=now(), updated_at=now())
        with self.connect() as db:
            duplicate = db.execute("SELECT * FROM tasks WHERE title=? AND done=0", (item['title'],)).fetchone()
            if duplicate:
                return dict(duplicate)
            db.execute("INSERT INTO tasks VALUES (:id,:title,:done,:created_at,:updated_at)", item)
        return item

    def set_task(self, task_id, done):
        if type(done) is not bool:
            raise ValueError("done 必须是布尔值")
        with self.connect() as db:
            if not db.execute("UPDATE tasks SET done=?,updated_at=? WHERE id=?", (int(done), now(), task_id)).rowcount:
                raise ValueError("待办不存在")

    def delete_task(self, task_id):
        with self.connect() as db:
            db.execute("DELETE FROM tasks WHERE id=?", (task_id,))

    def export(self):
        return dict(version=2, exported_at=now(), memories=self.memories(), tasks=self.tasks(),
                    sessions=[dict(s, messages=self.messages(s['id']), runs=[self.run(r['id']) for r in self.runs(s['id'], 1000000)]) for s in self.sessions()])
