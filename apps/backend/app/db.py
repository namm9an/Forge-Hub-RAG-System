import asyncio
from contextlib import asynccontextmanager
from contextvars import ContextVar
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
import re
import uuid

import asyncpg
from loguru import logger
import numpy as np

from .config import get_settings

_pool = None
_use_memory_fallback = False

SCHEMA_SQL = """
-- Forge Hub fresh schema; legacy upgrades use versioned migrations.
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS sessions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    session_id TEXT UNIQUE NOT NULL,
    user_id TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_activity TIMESTAMPTZ DEFAULT NOW(),
    total_messages INTEGER DEFAULT 0,
    total_documents INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_sessions_last_activity ON sessions(last_activity);
CREATE INDEX IF NOT EXISTS idx_sessions_owner_activity ON sessions(user_id, last_activity);
CREATE TABLE IF NOT EXISTS documents (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    file_size BIGINT NOT NULL,
    total_pages INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_documents_session_id ON documents(session_id);
CREATE TABLE IF NOT EXISTS chunks (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    heading TEXT NOT NULL DEFAULT '',
    content_hash TEXT NOT NULL,
    page_number INTEGER NOT NULL DEFAULT 1,
    page_numbers INTEGER[] NOT NULL DEFAULT '{}',
    token_count INTEGER NOT NULL DEFAULT 0,
    content_tsv tsvector GENERATED ALWAYS AS (
        to_tsvector('english', CASE WHEN heading = '' THEN content ELSE heading || chr(10) || content END)
    ) STORED,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(document_id, chunk_index),
    CONSTRAINT chunks_document_hash_unique UNIQUE(document_id, content_hash)
);
CREATE INDEX IF NOT EXISTS idx_chunks_content_tsv ON chunks USING GIN(content_tsv);
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_content_hash ON chunks(content_hash);
CREATE TABLE IF NOT EXISTS embeddings (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    chunk_id UUID NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    embedding vector(768) NOT NULL,
    model_version TEXT DEFAULT 'all-mpnet-base-v2',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(chunk_id)
);
CREATE INDEX IF NOT EXISTS idx_embeddings_vector_cosine
ON embeddings USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE TABLE IF NOT EXISTS chat_history (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    user_message TEXT NOT NULL,
    assistant_message TEXT NOT NULL,
    source_documents JSONB DEFAULT '[]',
    processing_time_ms INTEGER NOT NULL,
    retrieval_ms INTEGER,
    generation_ms INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_chat_history_session ON chat_history(session_id);
CREATE INDEX IF NOT EXISTS idx_chat_history_created_at ON chat_history(created_at DESC);
CREATE OR REPLACE FUNCTION cleanup_expired_sessions(p_user_id TEXT, max_age interval)
RETURNS INTEGER AS $$
DECLARE deleted_count INTEGER;
BEGIN
    DELETE FROM sessions WHERE user_id = p_user_id AND last_activity <= NOW() - max_age;
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;
CREATE OR REPLACE FUNCTION log_usage_event(p_session_id TEXT, p_event_type TEXT, p_event_data JSONB DEFAULT '{}'::jsonb)
RETURNS VOID AS $$
BEGIN
    UPDATE sessions SET last_activity=NOW(),
        total_messages=CASE WHEN p_event_type='chat_message' THEN total_messages+1 ELSE total_messages END,
        total_documents=CASE WHEN p_event_type='document_upload' THEN total_documents+1 ELSE total_documents END
    WHERE session_id=p_session_id;
END;
$$ LANGUAGE plpgsql;
CREATE TABLE IF NOT EXISTS auth_rate_limits (
    user_id TEXT PRIMARY KEY,
    window_start TIMESTAMPTZ NOT NULL,
    used INTEGER NOT NULL CHECK (used > 0)
);
CREATE INDEX IF NOT EXISTS idx_auth_rate_limits_window ON auth_rate_limits(window_start);
"""

class InMemoryStore:
    """Deterministic development/test adapter; never a durable database."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.sessions, self.documents, self.chunks, self.embeddings = {}, {}, {}, {}
        self.chat_history = []
        self.rate_limits = {}

    def clear(self):
        self.reset()


_memory_store = InMemoryStore()
_transaction_connection = ContextVar("forge_hub_connection", default=None)
_memory_transaction = ContextVar("forge_hub_memory_transaction", default=False)
_memory_lock = asyncio.Lock()
_memory_lock_loop = None
_pool_lock = asyncio.Lock()


async def get_pool():
    global _pool, _use_memory_fallback
    if _pool is not None:
        return _pool
    # Tests explicitly select this isolated adapter before application calls.
    if _use_memory_fallback:
        return None
    settings = get_settings()
    mode = getattr(settings, "storage_mode", "postgres")
    if mode == "memory":
        if getattr(settings, "environment", "development").lower() == "production":
            raise RuntimeError("Production cannot use memory storage")
        _use_memory_fallback = True
        return None
    if mode != "postgres":
        raise RuntimeError("Unsupported storage mode")
    dsn = settings.vector_db_url or settings.database_url
    if not dsn:
        raise RuntimeError("PostgreSQL storage requires DATABASE_URL or VECTOR_DB_URL")
    async with _pool_lock:
        if _pool is None:
            try:
                _pool = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=10, timeout=10.0)
            except Exception as error:
                logger.error("PostgreSQL connection unavailable ({})", type(error).__name__)
                raise RuntimeError("PostgreSQL storage is unavailable") from None
    return _pool


async def close_pool():
    global _pool, _use_memory_fallback
    if _pool is not None:
        await _pool.close()
    _pool = None
    _use_memory_fallback = False


class _MemoryConnection:
    async def execute(self, query, *args):
        return await execute(query, *args)

    async def fetch(self, query, *args):
        return await fetch(query, *args)

    async def fetchval(self, query, *args):
        return await fetchval(query, *args)

    async def fetchrow(self, query, *args):
        return await fetchrow(query, *args)


@asynccontextmanager
async def _memory_scope():
    snapshot = deepcopy(_memory_store.__dict__)
    token = _memory_transaction.set(True)
    try:
        yield _MemoryConnection()
    except BaseException:
        _memory_store.__dict__.clear()
        _memory_store.__dict__.update(snapshot)
        raise
    finally:
        _memory_transaction.reset(token)


@asynccontextmanager
async def transaction():
    """Bind wrappers to one real connection, or serialize/rollback memory writes."""
    current = _transaction_connection.get()
    if current is not None:
        async with current.transaction():
            yield current
        return
    pool = await get_pool()
    if pool is not None:
        async with pool.acquire() as connection:
            async with connection.transaction():
                token = _transaction_connection.set(connection)
                try:
                    yield connection
                finally:
                    _transaction_connection.reset(token)
        return
    if _memory_transaction.get():
        async with _memory_scope() as connection:
            yield connection
    else:
        # Test runners create a new event loop per case; a contended asyncio lock
        # remains bound to its original loop. Runtime development uses one loop.
        global _memory_lock, _memory_lock_loop
        loop = asyncio.get_running_loop()
        if _memory_lock_loop is not loop:
            if _memory_lock.locked():
                raise RuntimeError("Memory storage is already in use by another event loop")
            _memory_lock = asyncio.Lock()
            _memory_lock_loop = loop
        async with _memory_lock:
            async with _memory_scope() as connection:
                yield connection


async def ensure_schema():
    """Run at startup, never from request handlers."""
    if await get_pool() is None:
        return
    migration_dir = Path(__file__).resolve().parents[3] / "scripts" / "migrations"
    steps = [
        ("001_legacy_columns", (migration_dir / "001_legacy_columns.sql").read_text(encoding="utf-8")),
        ("002_private_storage", SCHEMA_SQL),
        ("003_hash_attribution", (migration_dir / "003_hash_attribution.sql").read_text(encoding="utf-8")),
        ("004_auth_rate_limits", (migration_dir / "004_auth_rate_limits.sql").read_text(encoding="utf-8")),
        ("005_chunk_headings", (migration_dir / "005_chunk_headings.sql").read_text(encoding="utf-8")),
    ]
    async with transaction() as connection:
        await connection.execute("SELECT pg_advisory_xact_lock(hashtext(current_schema()), 20261116)")
        await connection.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations "
            "(version TEXT PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW())"
        )
        for version, ddl in steps:
            if not await connection.fetchval("SELECT 1 FROM schema_migrations WHERE version=$1", version):
                await connection.execute(ddl)
                await connection.execute("INSERT INTO schema_migrations(version) VALUES($1)", version)


def _parse_vector(value):
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (list, tuple)):
        return np.array(value, dtype=np.float32)
    if isinstance(value, str):
        clean = value.strip("[]() ")
        if clean:
            return np.array([float(x) for x in clean.split(",")], dtype=np.float32)
    return np.zeros(768, dtype=np.float32)


def _now():
    return datetime.now(timezone.utc)


def _insert_values(query, args):
    match = re.search(r"INSERT\s+INTO\s+(\w+)\s*\(([^)]+)\)\s*VALUES\s*\(", query, re.I)
    if not match:
        raise NotImplementedError("Unsupported memory INSERT")
    columns = [part.strip().lower() for part in match.group(2).split(",")]
    start, level, end = match.end(), 1, match.end()
    while end < len(query) and level:
        level += (query[end] == "(") - (query[end] == ")")
        end += 1
    expressions, current, level = [], [], 0
    for char in query[start:end - 1] + ",":
        if char == "," and level == 0:
            expressions.append("".join(current).strip())
            current = []
        else:
            current.append(char)
            level += (char == "(") - (char == ")")
    values = {}
    for column, expression in zip(columns, expressions):
        parameter = re.match(r"^\$(\d+)", expression)
        if parameter:
            values[column] = args[int(parameter.group(1)) - 1]
        elif "gen_random_uuid" in expression.lower():
            values[column] = str(uuid.uuid4())
        elif expression.upper() == "NOW()":
            values[column] = _now()
        elif expression.upper() == "NULL":
            values[column] = None
        else:
            raise NotImplementedError("Use parameters for memory INSERT values")
    return match.group(1).lower(), values


def _memory_insert(query, args):
    table, values = _insert_values(query, args)
    record = {"id": str(uuid.uuid4()), "created_at": _now(), **values}
    conflict = None
    if table == "sessions":
        record = {"last_activity": _now(), "total_messages": 0, "total_documents": 0, "user_id": None, **record}
        key, target = record["session_id"], _memory_store.sessions
        conflict = target.get(key)
    elif table == "documents":
        if values["session_id"] not in _memory_store.sessions:
            raise asyncpg.ForeignKeyViolationError("Unknown document session")
        record = {"total_pages": 1, **record}
        key, target = record["id"], _memory_store.documents
    elif table == "chunks":
        record["document_id"] = str(record["document_id"])
        if record["document_id"] not in _memory_store.documents:
            raise asyncpg.ForeignKeyViolationError("Unknown chunk document")
        record = {"page_number": 1, "page_numbers": [], "token_count": 0, "heading": "", **record}
        key, target = record["id"], _memory_store.chunks
        conflict = next((c for c in target.values() if c["document_id"] == record["document_id"] and (
            c["chunk_index"] == record["chunk_index"] or c["content_hash"] == record["content_hash"]
        )), None)
    elif table == "embeddings":
        record["chunk_id"] = str(record["chunk_id"])
        if record["chunk_id"] not in _memory_store.chunks:
            raise asyncpg.ForeignKeyViolationError("Unknown embedding chunk")
        record["embedding"] = _parse_vector(record["embedding"])
        if record["embedding"].shape != (768,):
            raise ValueError("Expected a 768-dimensional embedding")
        key, target = record["chunk_id"], _memory_store.embeddings
        conflict = target.get(key)
    elif table == "chat_history":
        if record["session_id"] not in _memory_store.sessions:
            raise asyncpg.ForeignKeyViolationError("Unknown chat session")
        record = {"source_documents": "[]", "retrieval_ms": None, "generation_ms": None, **record}
        _memory_store.chat_history.append(record)
        return record
    else:
        raise NotImplementedError("Unsupported memory table")
    if conflict:
        clause = re.search(r"ON\s+CONFLICT\s*(?:\(([^)]+)\))?\s+DO", query, re.I)
        if clause is None:
            raise asyncpg.UniqueViolationError("Duplicate record")
        if clause.group(1) and table == "chunks":
            columns = {column.strip().lower() for column in clause.group(1).split(",")}
            matches_target = (
                columns == {"document_id", "chunk_index"} and conflict["chunk_index"] == record["chunk_index"]
            ) or (
                columns == {"document_id", "content_hash"} and conflict["content_hash"] == record["content_hash"]
            )
            if not matches_target:
                raise asyncpg.UniqueViolationError("Duplicate chunk outside the conflict target")
        if "DO UPDATE" in query.upper():
            conflict["last_activity"] = _now()
        return None
    target[key] = record
    return record


def _active(row, timeout):
    activity = row.get("last_activity")
    return isinstance(activity, datetime) and activity > _now() - timeout


def _owned_document(document, owner):
    return _memory_store.sessions.get(document["session_id"], {}).get("user_id") == owner


def _delete_document(did):
    _memory_store.documents.pop(did, None)
    for cid in [cid for cid, c in _memory_store.chunks.items() if c["document_id"] == did]:
        _memory_store.chunks.pop(cid, None)
        _memory_store.embeddings.pop(cid, None)


def _delete_session(sid):
    _memory_store.sessions.pop(sid, None)
    for did in [did for did, d in _memory_store.documents.items() if d["session_id"] == sid]:
        _delete_document(did)
    _memory_store.chat_history[:] = [h for h in _memory_store.chat_history if h["session_id"] != sid]


def _memory_delete(query, args):
    upper, removed = " ".join(query.upper().split()), []
    if upper.startswith("DELETE FROM SESSIONS"):
        if "WHERE USER_ID" in upper:
            owner, timeout = args
            ids = [sid for sid, row in _memory_store.sessions.items() if row.get("user_id") == owner and not _active(row, timeout)]
        elif "WHERE LAST_ACTIVITY" in upper:
            ids = [sid for sid, row in _memory_store.sessions.items() if not _active(row, args[0])]
        else:
            ids = [str(args[0])] if args else []
            if "AND USER_ID" in upper:
                ids = [sid for sid in ids if _memory_store.sessions.get(sid, {}).get("user_id") == args[1]]
        for sid in ids:
            if sid in _memory_store.sessions:
                removed.append(dict(_memory_store.sessions[sid]))
                _delete_session(sid)
    elif upper.startswith("DELETE FROM DOCUMENTS"):
        did = str(args[0])
        row = _memory_store.documents.get(did)
        if row and (len(args) == 1 or row["session_id"] == args[1]):
            removed.append(dict(row))
            _delete_document(did)
    elif upper.startswith("DELETE FROM CHUNKS"):
        cid = str(args[0])
        row = _memory_store.chunks.pop(cid, None)
        if row:
            removed.append(row)
            _memory_store.embeddings.pop(cid, None)
    else:
        raise NotImplementedError("Unsupported memory DELETE")
    return removed


async def execute(query, *args):
    connection = _transaction_connection.get()
    if connection is not None:
        return await connection.execute(query, *args)
    pool = await get_pool()
    if pool is not None:
        async with pool.acquire() as connection:
            return await connection.execute(query, *args)
    upper = " ".join(query.upper().split())
    if upper.startswith("INSERT"):
        return "INSERT 0 1" if _memory_insert(query, args) else "INSERT 0 0"
    if upper.startswith("DELETE"):
        return "DELETE " + str(len(_memory_delete(query, args)))
    if upper.startswith("UPDATE SESSIONS"):
        row = _memory_store.sessions.get(args[0])
        if row is None:
            return "UPDATE 0"
        row["last_activity"] = _now()
        for field in ("total_documents", "total_messages"):
            match = re.search(field.upper() + r"\s*=\s*" + field.upper() + r"\s*\+\s*\$(\d+)", upper)
            if match:
                row[field] += args[int(match.group(1)) - 1]
        if "GREATEST(TOTAL_DOCUMENTS - 1, 0)" in upper:
            row["total_documents"] = max(row["total_documents"] - 1, 0)
        return "UPDATE 1"
    raise NotImplementedError("Unsupported memory command")


async def fetchval(query, *args):
    connection = _transaction_connection.get()
    if connection is not None:
        return await connection.fetchval(query, *args)
    pool = await get_pool()
    if pool is not None:
        async with pool.acquire() as connection:
            return await connection.fetchval(query, *args)
    if query.lstrip().upper().startswith("INSERT"):
        record = _memory_insert(query, args)
        if record is None:
            return None
        return record["session_id"] if "RETURNING SESSION_ID" in query.upper() else record["id"]
    rows = await fetch(query, *args)
    return next(iter(rows[0].values())) if rows else None


async def fetchrow(query, *args):
    rows = await fetch(query, *args)
    return rows[0] if rows else None


async def fetch(query, *args):
    connection = _transaction_connection.get()
    if connection is not None:
        return [dict(row) for row in await connection.fetch(query, *args)]
    pool = await get_pool()
    if pool is not None:
        async with pool.acquire() as connection:
            return [dict(row) for row in await connection.fetch(query, *args)]
    upper = " ".join(query.upper().split())
    if upper.startswith("DELETE"):
        return _memory_delete(query, args)
    if "TOTAL_DOCUMENTS" in upper and "TOTAL_EMBEDDINGS" in upper:
        owner, timeout = args if args else (None, None)
        sids = {sid for sid, s in _memory_store.sessions.items() if owner is None or (
            s.get("user_id") == owner and _active(s, timeout)
        )}
        docs = {did for did, d in _memory_store.documents.items() if d["session_id"] in sids}
        chunks = {cid for cid, c in _memory_store.chunks.items() if c["document_id"] in docs}
        return [{"total_documents": len(docs), "total_chunks": len(chunks),
                 "total_embeddings": sum(cid in chunks for cid in _memory_store.embeddings),
                 "total_sessions": len(sids),
                 "total_messages": sum(h["session_id"] in sids for h in _memory_store.chat_history)}]
    if re.search(r"FROM SESSIONS(?:\s+S)? WHERE", upper):
        row = _memory_store.sessions.get(args[0])
        if row is None:
            return []
        if "USER_ID" in upper and row.get("user_id") != args[1]:
            return []
        if len(args) > 2 and not _active(row, args[2]):
            return []
        return [dict(row)]
    if "FROM DOCUMENTS" in upper and "SUM(FILE_SIZE)" in upper:
        docs = [d for d in _memory_store.documents.values() if d["session_id"] == args[0]]
        return [{"count": len(docs), "total_bytes": sum(d["file_size"] for d in docs)}]
    if "FROM DOCUMENTS" in upper and "WHERE" in upper:
        if re.search(r"WHERE (?:D\.)?ID\s*=", upper):
            document = _memory_store.documents.get(str(args[0]))
            if not document or ("JOIN SESSIONS" in upper and not _owned_document(document, args[1])):
                return []
            return [dict(document)]
        sid = args[0] if args else None
        owner_scoped = "JOIN SESSIONS" in upper
        return [{**document, "chunks": sum(c["document_id"] == did for c in _memory_store.chunks.values())}
                for did, document in _memory_store.documents.items()
                if (sid is None or document["session_id"] == sid)
                and (not owner_scoped or (len(args) > 1 and args[1] is not None and _owned_document(document, args[1])))]
    if "FROM CHUNKS C" in upper and "LEFT JOIN EMBEDDINGS E" in upper:
        sid = args[0] if args else None
        return [{"id": cid, "content": chunk["content"], "heading": chunk.get("heading", "")}
                for cid, chunk in _memory_store.chunks.items()
                if cid not in _memory_store.embeddings and (
                    sid is None or _memory_store.documents[chunk["document_id"]]["session_id"] == sid
                )][:2000]
    if "EMBEDDINGS E" in upper and "<=>" in query:
        q_vec = _parse_vector(args[0])
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm
        sid, limit = args[1], args[2]
        owner = args[3] if len(args) > 3 else None
        rows = []
        for cid, emb in _memory_store.embeddings.items():
            chunk = _memory_store.chunks[cid]
            document = _memory_store.documents[chunk["document_id"]]
            if sid is not None and document["session_id"] != sid:
                continue
            if owner is not None and not _owned_document(document, owner):
                continue
            vec = emb["embedding"]
            norm = np.linalg.norm(vec)
            score = float(np.dot(q_vec, vec / norm if norm > 0 else vec))
            rows.append({"chunk_id": cid, "document_id": chunk["document_id"], "content": chunk["content"],
                         "page_number": chunk["page_number"], "filename": document["filename"], "dense_score": score})
        return sorted(rows, key=lambda r: r["dense_score"], reverse=True)[:limit]
    if "CONTENT_TSV" in upper or "TS_RANK_CD" in upper or "WEBSEARCH_TO_TSQUERY" in upper:
        terms = set(re.findall(r"\w+", str(args[0]).lower()))
        sid, limit = args[1], args[2]
        owner = args[3] if len(args) > 3 else None
        rows = []
        for cid, chunk in _memory_store.chunks.items():
            document = _memory_store.documents[chunk["document_id"]]
            if sid is not None and document["session_id"] != sid:
                continue
            if owner is not None and not _owned_document(document, owner):
                continue
            words = re.findall(r"\w+", (chunk.get("heading", "") + "\n" + chunk["content"]).lower())
            overlap = terms.intersection(words)
            if not overlap:
                continue
            score = sum(words.count(t) for t in overlap) / max(1.0, np.log1p(len(words))) * len(overlap) / max(1, len(terms))
            rows.append({"chunk_id": cid, "document_id": chunk["document_id"], "content": chunk["content"],
                         "page_number": chunk["page_number"], "filename": document["filename"], "sparse_score": float(score)})
        return sorted(rows, key=lambda r: r["sparse_score"], reverse=True)[:limit]
    if "FROM CHAT_HISTORY" in upper:
        sid = args[0] if args else None
        limit = args[1] if len(args) > 1 else 50
        owner = args[2] if len(args) > 2 else None
        if owner is not None and _memory_store.sessions.get(sid, {}).get("user_id") != owner:
            return []
        rows = [h for h in _memory_store.chat_history if sid is None or h["session_id"] == sid]
        return list(reversed(rows))[:limit]
    if "COUNT(*)" in upper:
        for table in ("sessions", "documents", "chunks", "embeddings", "chat_history"):
            if "FROM " + table.upper() in upper:
                return [{"count": len(getattr(_memory_store, table))}]
    raise NotImplementedError("Unsupported memory query")


async def owner_stats(user_id):
    timeout = timedelta(seconds=get_settings().session_timeout_seconds)
    rows = await fetch(
        """SELECT
        (SELECT COUNT(*) FROM documents d JOIN sessions s ON s.session_id=d.session_id WHERE s.user_id=$1 AND s.last_activity>NOW()-$2::interval) AS total_documents,
        (SELECT COUNT(*) FROM chunks c JOIN documents d ON d.id=c.document_id JOIN sessions s ON s.session_id=d.session_id WHERE s.user_id=$1 AND s.last_activity>NOW()-$2::interval) AS total_chunks,
        (SELECT COUNT(*) FROM embeddings e JOIN chunks c ON c.id=e.chunk_id JOIN documents d ON d.id=c.document_id JOIN sessions s ON s.session_id=d.session_id WHERE s.user_id=$1 AND s.last_activity>NOW()-$2::interval) AS total_embeddings,
        (SELECT COUNT(*) FROM sessions s WHERE s.user_id=$1 AND s.last_activity>NOW()-$2::interval) AS total_sessions,
        (SELECT COUNT(*) FROM chat_history h JOIN sessions s ON s.session_id=h.session_id WHERE s.user_id=$1 AND s.last_activity>NOW()-$2::interval) AS total_messages
        """, user_id, timeout,
    )
    return rows[0]


async def cleanup_expired_sessions():
    """Internal maintenance only; request handlers expose owner-scoped cleanup."""
    timeout = timedelta(seconds=get_settings().session_timeout_seconds)
    async with transaction():
        removed = await fetch(
            "DELETE FROM sessions WHERE last_activity<=NOW()-$1::interval RETURNING session_id", timeout,
        )
        if await get_pool() is not None:
            await execute("DELETE FROM auth_rate_limits WHERE window_start < NOW()-interval '2 minutes'")
        else:
            cutoff = _now() - timedelta(minutes=2)
            for owner in list(_memory_store.rate_limits):
                if _memory_store.rate_limits[owner]["window_start"] < cutoff:
                    del _memory_store.rate_limits[owner]
    return len(removed)


async def check_schema():
    """Read-only production readiness when startup migrations are disabled."""
    if await get_pool() is None:
        return
    try:
        rows = await fetch("SELECT version FROM schema_migrations")
    except asyncpg.UndefinedTableError as error:
        raise RuntimeError("Database migrations must be applied before serving requests") from error
    expected = {"001_legacy_columns", "002_private_storage", "003_hash_attribution", "004_auth_rate_limits",
                "005_chunk_headings"}
    if not expected.issubset({row["version"] for row in rows}):
        raise RuntimeError("Database migrations must be applied before serving requests")


async def create_session_record(session_id, user_id):
    """Caller transaction serializes a user's session budget across processes."""
    settings = get_settings()
    if await get_pool() is None:
        if session_id in _memory_store.sessions:
            return
        timeout = timedelta(seconds=settings.session_timeout_seconds)
        count = sum(row.get("user_id") == user_id and _active(row, timeout) for row in _memory_store.sessions.values())
    else:
        await execute("SELECT pg_advisory_xact_lock(20261117, hashtext($1))", user_id)
        if await fetchval("SELECT EXISTS(SELECT 1 FROM sessions WHERE session_id=$1)", session_id):
            return
        count = await fetchval(
            "SELECT COUNT(*) FROM sessions WHERE user_id=$1 AND last_activity>NOW()-$2::interval",
            user_id, timedelta(seconds=settings.session_timeout_seconds),
        )
    if count >= settings.max_sessions_per_user:
        from fastapi import HTTPException
        raise HTTPException(429, "Active session quota reached")
    await execute("INSERT INTO sessions(session_id,user_id) VALUES($1,$2) ON CONFLICT(session_id) DO NOTHING", session_id, user_id)


async def consume_rate_limit(user_id):
    """Atomic fixed-minute authenticated request budget shared by every API worker."""
    limit = get_settings().requests_per_minute
    async with transaction():
        if await get_pool() is None:
            window = _now().replace(second=0, microsecond=0)
            bucket = _memory_store.rate_limits.get(user_id)
            if bucket is None or bucket["window_start"] != window:
                _memory_store.rate_limits[user_id] = {"window_start": window, "used": 1}
                return True
            if bucket["used"] >= limit:
                return False
            bucket["used"] += 1
            return True
        admitted = await fetchval(
            """INSERT INTO auth_rate_limits(user_id,window_start,used)
            VALUES($1,date_trunc('minute',NOW()),1)
            ON CONFLICT(user_id) DO UPDATE SET
                window_start=date_trunc('minute',NOW()),
                used=CASE WHEN auth_rate_limits.window_start=date_trunc('minute',NOW())
                          THEN auth_rate_limits.used+1 ELSE 1 END
            WHERE auth_rate_limits.window_start<>date_trunc('minute',NOW()) OR auth_rate_limits.used<$2
            RETURNING used""", user_id, limit,
        )
        return admitted is not None


async def check_runtime_privileges():
    """SELECT-only guard for the effective production runtime role, not an RLS claim.

    Migrations/provisioning use a separate identity. Runtime credentials must
    belong to a dedicated non-owner login, never an administrator using SET ROLE.
    """
    message = "Runtime PostgreSQL role must be least-privilege with only required application permissions"
    if await get_pool() is None:
        raise RuntimeError(message)
    mutable = {"sessions", "documents", "chunks", "embeddings", "chat_history", "auth_rate_limits"}
    required = mutable | {"schema_migrations"}
    async with transaction() as connection:
        schema = await connection.fetchval("SELECT current_schema()")
        if schema is None:
            raise RuntimeError(message)
        state = await connection.fetchrow(
            """SELECT
            EXISTS(SELECT 1 FROM pg_roles r WHERE pg_has_role(r.oid, 'MEMBER') AND
                (r.rolsuper OR r.rolbypassrls OR r.rolcreatedb OR r.rolcreaterole OR r.rolreplication
                 OR left(r.rolname, 3)='pg_')) AS elevated,
            EXISTS(SELECT 1 FROM pg_database d WHERE d.datname=current_database()
                AND pg_has_role(d.datdba, 'MEMBER')) AS database_owner,
            has_database_privilege(current_database(), 'CREATE') AS database_create,
            has_schema_privilege($1, 'USAGE') AS schema_usage,
            has_schema_privilege($1, 'CREATE') AS schema_create,
            EXISTS(SELECT 1 FROM pg_namespace n WHERE n.nspname=$1
                AND pg_has_role(n.nspowner, 'MEMBER')) AS schema_owner,
            EXISTS(SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
                WHERE n.nspname=$1 AND c.relkind IN ('r','p','v','m','f','S')
                AND pg_has_role(c.relowner, 'MEMBER')) AS object_owner
            """, schema,
        )
        if not state["schema_usage"] or any(state[key] for key in (
            "elevated", "database_owner", "database_create", "schema_create", "schema_owner", "object_owner",
        )):
            raise RuntimeError(message)
        rows = await connection.fetch(
            """SELECT c.relname, c.relkind::text AS relkind,
            has_table_privilege(c.oid, 'SELECT') AS can_read,
            has_table_privilege(c.oid, 'INSERT') AS can_insert,
            has_table_privilege(c.oid, 'UPDATE') AS can_update,
            has_table_privilege(c.oid, 'DELETE') AS can_delete,
            has_any_column_privilege(c.oid, 'INSERT') AS can_insert_column,
            has_any_column_privilege(c.oid, 'UPDATE') AS can_update_column,
            (has_any_column_privilege(c.oid, 'REFERENCES') OR
             EXISTS(SELECT 1 FROM unnest(ARRAY['SELECT WITH GRANT OPTION','INSERT WITH GRANT OPTION',
                 'UPDATE WITH GRANT OPTION','REFERENCES WITH GRANT OPTION']) AS privilege
                 WHERE has_any_column_privilege(c.oid, privilege))) AS excess_columns,
            EXISTS(SELECT 1 FROM unnest($3::text[]) AS privilege
                WHERE has_table_privilege(c.oid, privilege)) AS excess
            FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
            WHERE n.nspname=$1 AND c.relname=ANY($2::text[])
            """, schema, sorted(required), [
                "TRUNCATE", "REFERENCES", "TRIGGER", "SELECT WITH GRANT OPTION",
                "INSERT WITH GRANT OPTION", "UPDATE WITH GRANT OPTION", "DELETE WITH GRANT OPTION",
            ],
        )
        if {row["relname"] for row in rows} != required:
            raise RuntimeError(message)
        for row in rows:
            writes = [row[key] for key in ("can_insert", "can_update", "can_delete")]
            if row["relkind"] not in {"r", "p"} or not row["can_read"] or row["excess"] or row["excess_columns"]:
                raise RuntimeError(message)
            if (row["relname"] in mutable and not all(writes)) or (row["relname"] == "schema_migrations" and any(
                writes + [row["can_insert_column"], row["can_update_column"]]
            )):
                raise RuntimeError(message)
