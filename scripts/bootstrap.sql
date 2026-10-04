-- =============================================================================
-- Forge Hub RAG System — Production Database Bootstrap DDL
-- Target: PostgreSQL 16 + pgvector
-- Specification: docs/FORGE_HUB_UPGRADE_BLUEPRINT.md
-- Compliance: docs/FORGE_HUB_EVALUATION_RULES.md (Rules A-1 through A-5)
-- =============================================================================

-- Enable required PostgreSQL extensions
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS vector;

-- 1. Sessions Table (Dynamic, isolated multi-user sessions)
CREATE TABLE IF NOT EXISTS sessions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    session_id TEXT UNIQUE NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_activity TIMESTAMPTZ DEFAULT NOW(),
    total_messages INTEGER DEFAULT 0,
    total_documents INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_sessions_last_activity ON sessions(last_activity);

-- 2. Documents Table (Cascading foreign key from sessions)
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
-- Rule A-3: Vector index tuned for cosine similarity
-- Rule A-4: Cascades on chunk deletion
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

-- 6. Helper Functions & Procedures
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
