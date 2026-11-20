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
    content_hash TEXT NOT NULL,
    page_number INTEGER NOT NULL DEFAULT 1,
    page_numbers INTEGER[] NOT NULL DEFAULT '{}',
    token_count INTEGER NOT NULL DEFAULT 0,
    content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
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
