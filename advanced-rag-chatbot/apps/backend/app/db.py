import asyncio
from typing import Optional, Any
import asyncpg
from loguru import logger
from .config import get_settings

_pool: Optional[asyncpg.pool.Pool] = None


async def get_pool() -> asyncpg.pool.Pool:
    global _pool
    if _pool is None:
        settings = get_settings()
        db_url = (
            settings.vector_db_url
            or settings.database_url
            or settings.__dict__.get("DATABASE_URL")
            or None
        )
        if not db_url:
            raise RuntimeError("DATABASE_URL / VECTOR_DB_URL not configured")
        logger.info("Initializing Postgres connection pool")
        _pool = await asyncpg.create_pool(dsn=db_url, min_size=1, max_size=10)
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


SCHEMA_SQL = """
-- Enable required PostgreSQL extensions
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS sessions (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  session_id TEXT UNIQUE NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  last_activity TIMESTAMPTZ DEFAULT NOW(),
  total_messages INTEGER DEFAULT 0,
  total_documents INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS documents (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  session_id TEXT NOT NULL,
  filename TEXT NOT NULL,
  file_size BIGINT NOT NULL,
  total_pages INTEGER,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chunks (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
  chunk_index INTEGER NOT NULL,
  content TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  page_number INTEGER,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(document_id, chunk_index)
);

CREATE TABLE IF NOT EXISTS embeddings (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  chunk_id UUID REFERENCES chunks(id) ON DELETE CASCADE,
  embedding vector(768) NOT NULL,
  model_version TEXT DEFAULT 'all-mpnet-base-v2',
  created_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(chunk_id)
);

-- Vector index tuned for cosine similarity
CREATE INDEX IF NOT EXISTS embeddings_vec_cos_idx
  ON embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE TABLE IF NOT EXISTS chat_history (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  session_id TEXT NOT NULL,
  user_message TEXT NOT NULL,
  assistant_message TEXT NOT NULL,
  source_documents JSONB DEFAULT '[]',
  processing_time_ms INTEGER,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Helpful indexes
CREATE INDEX IF NOT EXISTS sessions_last_activity_idx ON sessions(last_activity);
CREATE INDEX IF NOT EXISTS chat_history_session_idx ON chat_history(session_id);
CREATE INDEX IF NOT EXISTS chat_history_created_at_idx ON chat_history(created_at DESC);

-- Optional utility functions
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
      total_messages = CASE WHEN p_event_type = 'chat_message' THEN total_messages + 1 ELSE total_messages END
  WHERE session_id = p_session_id;
END;
$$ LANGUAGE plpgsql;
"""


async def ensure_schema() -> None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        try:
            await conn.execute(SCHEMA_SQL)
            logger.info("Database schema ensured (tables present)")
        except Exception as exc:
            logger.exception("Failed ensuring schema: {}", exc)
            raise


async def fetchval(query: str, *args: Any) -> Any:
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchval(query, *args)


async def execute(query: str, *args: Any) -> str:
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.execute(query, *args)


async def fetch(query: str, *args: Any) -> list[asyncpg.Record]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *args)
        return rows
