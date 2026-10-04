import asyncio
import re
import uuid
import json
from typing import Optional, Any, List, Dict
import asyncpg
from loguru import logger
import numpy as np
from .config import get_settings

_pool: Optional[asyncpg.pool.Pool] = None
_use_memory_fallback: bool = False

SCHEMA_SQL = """
-- Enable required PostgreSQL extensions
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS vector;

-- 1. Sessions Table
CREATE TABLE IF NOT EXISTS sessions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    session_id TEXT UNIQUE NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_activity TIMESTAMPTZ DEFAULT NOW(),
    total_messages INTEGER DEFAULT 0,
    total_documents INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_sessions_last_activity ON sessions(last_activity);

-- 2. Documents Table (Cascading foreign key)
CREATE TABLE IF NOT EXISTS documents (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    file_size BIGINT NOT NULL,
    total_pages INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_documents_session_id ON documents(session_id);

-- 3. Chunks Table (With Full-Text Search tsvector & Metadata)
-- Rule A-1: content_tsv GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
-- Rule A-5: page_number INT and token_count INT columns
CREATE TABLE IF NOT EXISTS chunks (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    page_number INTEGER NOT NULL DEFAULT 1,
    token_count INTEGER NOT NULL DEFAULT 0,
    content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(document_id, chunk_index)
);

-- Rule A-2: GIN index for lexical search
CREATE INDEX IF NOT EXISTS idx_chunks_content_tsv ON chunks USING GIN(content_tsv);
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_content_hash ON chunks(content_hash);

-- 4. Embeddings Table
-- Rule A-3: Vector index tuned for cosine similarity using HNSW
-- Rule A-4: Cascading foreign keys on chunk_id
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

-- 5. Chat History Table
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

-- 6. Functions
CREATE OR REPLACE FUNCTION cleanup_expired_sessions(max_age interval DEFAULT interval '1 hour')
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM sessions WHERE last_activity < NOW() - max_age;
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION log_usage_event(p_session_id TEXT, p_event_type TEXT, p_event_data JSONB DEFAULT '{}'::jsonb)
RETURNS VOID AS $$
BEGIN
    UPDATE sessions
    SET last_activity = NOW(),
        total_messages = CASE WHEN p_event_type = 'chat_message' THEN total_messages + 1 ELSE total_messages END,
        total_documents = CASE WHEN p_event_type = 'document_upload' THEN total_documents + 1 ELSE total_documents END
    WHERE session_id = p_session_id;
END;
$$ LANGUAGE plpgsql;
"""


class InMemoryStore:
    """
    High-Fidelity In-Memory Store for unit testing and offline development.
    Emulates PostgreSQL 16 + pgvector relational behaviors, cascading constraints,
    HNSW cosine similarity, and full-text search matching without requiring an active PostgreSQL daemon.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.chunks: Dict[str, Dict[str, Any]] = {}
        self.embeddings: Dict[str, Dict[str, Any]] = {}
        self.chat_history: List[Dict[str, Any]] = []

    def clear(self):
        self.sessions.clear()
        self.documents.clear()
        self.chunks.clear()
        self.embeddings.clear()
        self.chat_history.clear()


_memory_store = InMemoryStore()


async def get_pool() -> Optional[asyncpg.pool.Pool]:
    global _pool, _use_memory_fallback
    if _pool is not None:
        return _pool

    if _use_memory_fallback:
        return None

    settings = get_settings()
    db_url = settings.vector_db_url or settings.database_url
    if not db_url:
        logger.warning("No DATABASE_URL or VECTOR_DB_URL configured. Activating high-fidelity in-memory engine.")
        _use_memory_fallback = True
        return None

    try:
        logger.info(f"Connecting to Postgres connection pool: {db_url.split('@')[-1] if '@' in db_url else 'configured DSN'}")
        _pool = await asyncpg.create_pool(dsn=db_url, min_size=1, max_size=10, timeout=10.0)
        return _pool
    except Exception as e:
        logger.warning(f"Could not connect to PostgreSQL ({e}). Falling back to in-memory engine.")
        _use_memory_fallback = True
        return None


async def close_pool() -> None:
    global _pool, _use_memory_fallback
    if _pool:
        await _pool.close()
        _pool = None
    _use_memory_fallback = False


async def ensure_schema() -> None:
    pool = await get_pool()
    if pool:
        async with pool.acquire() as conn:
            try:
                await conn.execute(SCHEMA_SQL)
                logger.info("Database schema successfully ensured on PostgreSQL")
            except Exception as exc:
                logger.error(f"Failed ensuring PostgreSQL schema: {exc}")
                raise
    else:
        logger.info("In-memory database schema ready.")


def _parse_vector(vec_val: Any) -> np.ndarray:
    if isinstance(vec_val, np.ndarray):
        return vec_val
    if isinstance(vec_val, (list, tuple)):
        return np.array(vec_val, dtype=np.float32)
    if isinstance(vec_val, str):
        # Format: "[0.1, 0.2, ...]"
        clean = vec_val.strip("[]() ")
        if clean:
            return np.array([float(x) for x in clean.split(",")], dtype=np.float32)
    return np.zeros(768, dtype=np.float32)


async def execute(query: str, *args: Any) -> str:
    pool = await get_pool()
    if pool:
        async with pool.acquire() as conn:
            return await conn.execute(query, *args)

    # In-memory execution
    q = query.strip()
    q_upper = q.upper()

    if q_upper.startswith("INSERT INTO SESSIONS"):
        sid = args[0] if args else str(uuid.uuid4())
        _memory_store.sessions[sid] = {
            "id": str(uuid.uuid4()),
            "session_id": sid,
            "total_messages": 0,
            "total_documents": 0,
        }
        return "INSERT 1"

    if q_upper.startswith("INSERT INTO DOCUMENTS"):
        doc_id = str(uuid.uuid4())
        sid, filename, file_size, pages = args[0], args[1], args[2], args[3]
        _memory_store.documents[doc_id] = {
            "id": doc_id,
            "session_id": sid,
            "filename": filename,
            "file_size": file_size,
            "total_pages": pages,
        }
        return "INSERT 1"

    if q_upper.startswith("INSERT INTO CHUNKS"):
        # document_id, chunk_index, content, content_hash, page_number, token_count
        doc_id, idx, content, chash = args[0], args[1], args[2], args[3]
        page_no = args[4] if len(args) > 4 else 1
        tokens = args[5] if len(args) > 5 else len(content.split())
        cid = str(uuid.uuid4())
        _memory_store.chunks[cid] = {
            "id": cid,
            "document_id": str(doc_id),
            "chunk_index": idx,
            "content": content,
            "content_hash": chash,
            "page_number": page_no,
            "token_count": tokens,
        }
        return "INSERT 1"

    if q_upper.startswith("INSERT INTO EMBEDDINGS"):
        chunk_id, emb_val, model_ver = args[0], args[1], args[2]
        emb_arr = _parse_vector(emb_val)
        _memory_store.embeddings[str(chunk_id)] = {
            "id": str(uuid.uuid4()),
            "chunk_id": str(chunk_id),
            "embedding": emb_arr,
            "model_version": model_ver,
        }
        return "INSERT 1"

    if q_upper.startswith("INSERT INTO CHAT_HISTORY"):
        sid, u_msg, a_msg, src_docs, proc_time = args[0], args[1], args[2], args[3], args[4]
        ret_ms = args[5] if len(args) > 5 else None
        gen_ms = args[6] if len(args) > 6 else None
        _memory_store.chat_history.append({
            "id": str(uuid.uuid4()),
            "session_id": sid,
            "user_message": u_msg,
            "assistant_message": a_msg,
            "source_documents": src_docs,
            "processing_time_ms": proc_time,
            "retrieval_ms": ret_ms,
            "generation_ms": gen_ms,
            "created_at": "NOW()",
        })
        return "INSERT 1"

    if q_upper.startswith("DELETE FROM CHUNKS"):
        target_id = str(args[0]) if args else None
        if target_id and target_id in _memory_store.chunks:
            del _memory_store.chunks[target_id]
            if target_id in _memory_store.embeddings:
                del _memory_store.embeddings[target_id]
        return "DELETE 1"

    if q_upper.startswith("DELETE FROM DOCUMENTS"):
        doc_id = str(args[0]) if args else None
        if doc_id and doc_id in _memory_store.documents:
            del _memory_store.documents[doc_id]
        # Cascades to chunks and embeddings (Rule A-4)
        cids_to_del = [cid for cid, c in _memory_store.chunks.items() if c["document_id"] == doc_id]
        for cid in cids_to_del:
            del _memory_store.chunks[cid]
            if cid in _memory_store.embeddings:
                del _memory_store.embeddings[cid]
        return "DELETE 1"

    if q_upper.startswith("DELETE FROM SESSIONS"):
        sid = str(args[0]) if args else None
        if sid and sid in _memory_store.sessions:
            del _memory_store.sessions[sid]
        # Cascades to documents, chunks, embeddings, chat_history (Rule A-4 / Cascading Foreign Key)
        docs_to_del = [did for did, d in _memory_store.documents.items() if d["session_id"] == sid]
        for did in docs_to_del:
            del _memory_store.documents[did]
            cids = [cid for cid, c in _memory_store.chunks.items() if c["document_id"] == did]
            for cid in cids:
                del _memory_store.chunks[cid]
                if cid in _memory_store.embeddings:
                    del _memory_store.embeddings[cid]
        _memory_store.chat_history = [h for h in _memory_store.chat_history if h["session_id"] != sid]
        return "DELETE 1"

    if q_upper.startswith("UPDATE SESSIONS"):
        sid = args[0]
        if sid in _memory_store.sessions:
            _memory_store.sessions[sid]["last_activity"] = "NOW()"
        return "UPDATE 1"

    return "COMMAND OK"


async def fetchval(query: str, *args: Any) -> Any:
    pool = await get_pool()
    if pool:
        async with pool.acquire() as conn:
            return await conn.fetchval(query, *args)

    # In-memory execution
    q_upper = query.strip().upper()

    if "INSERT INTO SESSIONS" in q_upper and "RETURNING SESSION_ID" in q_upper:
        sid = str(uuid.uuid4())
        _memory_store.sessions[sid] = {
            "id": str(uuid.uuid4()),
            "session_id": sid,
            "total_messages": 0,
            "total_documents": 0,
        }
        return sid

    if "INSERT INTO DOCUMENTS" in q_upper and "RETURNING ID" in q_upper:
        doc_id = str(uuid.uuid4())
        sid, filename, file_size, pages = args[0], args[1], args[2], args[3]
        # Ensure session exists (Rule A-4 / B-5)
        if sid not in _memory_store.sessions:
            _memory_store.sessions[sid] = {
                "id": str(uuid.uuid4()),
                "session_id": sid,
                "total_messages": 0,
                "total_documents": 1,
            }
        else:
            _memory_store.sessions[sid]["total_documents"] += 1

        _memory_store.documents[doc_id] = {
            "id": doc_id,
            "session_id": sid,
            "filename": filename,
            "file_size": file_size,
            "total_pages": pages,
        }
        return doc_id

    if "INSERT INTO CHUNKS" in q_upper and "RETURNING ID" in q_upper:
        cid = str(uuid.uuid4())
        doc_id, idx, content, chash = args[0], args[1], args[2], args[3]
        page_no = args[4] if len(args) > 4 else 1
        tokens = args[5] if len(args) > 5 else len(content.split())
        _memory_store.chunks[cid] = {
            "id": cid,
            "document_id": str(doc_id),
            "chunk_index": idx,
            "content": content,
            "content_hash": chash,
            "page_number": page_no,
            "token_count": tokens,
        }
        return cid

    if "COUNT(*)" in q_upper:
        if "FROM SESSIONS" in q_upper:
            return len(_memory_store.sessions)
        if "FROM DOCUMENTS" in q_upper:
            return len(_memory_store.documents)
        if "FROM CHUNKS" in q_upper:
            return len(_memory_store.chunks)
        if "FROM EMBEDDINGS" in q_upper:
            return len(_memory_store.embeddings)

    return None


async def fetch(query: str, *args: Any) -> List[Dict[str, Any]]:
    pool = await get_pool()
    if pool:
        async with pool.acquire() as conn:
            records = await conn.fetch(query, *args)
            return [dict(r) for r in records]

    # In-memory execution
    q = query.strip()
    q_upper = q.upper()

    # Documents query
    if "FROM DOCUMENTS" in q_upper and "WHERE" in q_upper and "SESSION_ID" in q_upper:
        sid = args[0] if args else None
        res = []
        for did, d in _memory_store.documents.items():
            if sid is None or d["session_id"] == sid:
                # Count chunks
                c_count = sum(1 for c in _memory_store.chunks.values() if c["document_id"] == did)
                res.append({
                    "id": did,
                    "filename": d["filename"],
                    "file_size": d["file_size"],
                    "total_pages": d["total_pages"],
                    "session_id": d["session_id"],
                    "chunks": c_count,
                    "created_at": "NOW()",
                })
        return res

    # Single document query
    if "FROM DOCUMENTS" in q_upper and "WHERE ID =" in q_upper:
        did = str(args[0]) if args else None
        if did and did in _memory_store.documents:
            d = _memory_store.documents[did]
            return [{
                "id": did,
                "session_id": d["session_id"],
                "filename": d["filename"],
                "file_size": d["file_size"],
                "total_pages": d["total_pages"],
            }]
        return []

    # System Stats
    if "SELECT COUNT(DISTINCT" in q_upper or ("TOTAL_CHUNKS" in q_upper and "TOTAL_EMBEDDINGS" in q_upper):
        return [{
            "total_documents": len(_memory_store.documents),
            "total_chunks": len(_memory_store.chunks),
            "total_embeddings": len(_memory_store.embeddings),
            "total_sessions": len(_memory_store.sessions),
            "total_messages": len(_memory_store.chat_history),
        }]

    # Chunks missing embeddings
    if "FROM CHUNKS C" in q_upper and "LEFT JOIN EMBEDDINGS E" in q_upper:
        results = []
        for cid, chunk in _memory_store.chunks.items():
            if cid not in _memory_store.embeddings:
                results.append({"id": cid, "content": chunk["content"]})
        return results

    # Dense Vector Search Query
    if "EMBEDDINGS E" in q_upper and "<=>" in query:
        q_vec_str = args[0]
        q_vec = _parse_vector(q_vec_str)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        session_filter = args[1] if len(args) > 1 else None
        top_k = args[2] if len(args) > 2 else 10

        candidates = []
        for cid, emb_data in _memory_store.embeddings.items():
            chunk = _memory_store.chunks.get(cid)
            if not chunk:
                continue
            doc = _memory_store.documents.get(chunk["document_id"])
            if not doc:
                continue
            if session_filter and doc["session_id"] != session_filter:
                continue

            doc_vec = emb_data["embedding"]
            d_norm = np.linalg.norm(doc_vec)
            if d_norm > 0:
                doc_vec = doc_vec / d_norm

            # Real cosine similarity (Rule C-1)
            cosine_sim = float(np.dot(q_vec, doc_vec))
            candidates.append({
                "chunk_id": cid,
                "content": chunk["content"],
                "page_number": chunk["page_number"],
                "document_id": chunk["document_id"],
                "filename": doc["filename"],
                "dense_score": cosine_sim,
            })

        candidates.sort(key=lambda x: x["dense_score"], reverse=True)
        return candidates[:top_k]

    # Sparse Lexical FTS Search Query (supports websearch_to_tsquery semantics)
    if "CONTENT_TSV" in q_upper or "TS_RANK_CD" in q_upper or "WEBSEARCH_TO_TSQUERY" in q_upper:
        query_text = str(args[0]).lower()
        session_filter = args[1] if len(args) > 1 else None
        top_k = args[2] if len(args) > 2 else 10

        q_terms = set(re.findall(r"\w+", query_text))
        candidates = []
        for cid, chunk in _memory_store.chunks.items():
            doc = _memory_store.documents.get(chunk["document_id"])
            if not doc:
                continue
            if session_filter and doc["session_id"] != session_filter:
                continue

            c_text = chunk["content"].lower()
            c_terms = re.findall(r"\w+", c_text)
            c_set = set(c_terms)
            overlap = q_terms.intersection(c_set)
            if not overlap and len(q_terms) > 0:
                continue

            # Real lexical scoring proportional to term frequency / coverage (BM25 surrogate)
            tf = sum(c_terms.count(term) for term in overlap)
            length_norm = np.log1p(len(c_terms))
            sparse_score = float((tf / max(1.0, length_norm)) * (len(overlap) / max(1.0, len(q_terms))))

            candidates.append({
                "chunk_id": cid,
                "content": chunk["content"],
                "page_number": chunk["page_number"],
                "document_id": chunk["document_id"],
                "filename": doc["filename"],
                "sparse_score": sparse_score,
            })

        candidates.sort(key=lambda x: x["sparse_score"], reverse=True)
        return candidates[:top_k]

    # Chat history query
    if "FROM CHAT_HISTORY" in q_upper:
        sid = args[0] if args else None
        limit = args[1] if len(args) > 1 else 50
        filtered = [h for h in _memory_store.chat_history if sid is None or h["session_id"] == sid]
        return list(reversed(filtered))[:limit]

    return []
