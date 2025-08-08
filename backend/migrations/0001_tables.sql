-- 0001_tables.sql
-- Initializes core tables for sessions, chat_history, embedding_cache, and rate_limit_tracking
-- Safe to run multiple times due to IF NOT EXISTS and conditional constraints.

BEGIN;

-- Ensure required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector; -- pgvector

-- Sessions table: tracks anonymous or identified chat sessions
CREATE TABLE IF NOT EXISTS public.sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT TIMEZONE('utc'::text, NOW()),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT TIMEZONE('utc'::text, NOW()),
    last_active_at TIMESTAMPTZ NOT NULL DEFAULT TIMEZONE('utc'::text, NOW()),
    expires_at TIMESTAMPTZ NULL,
    metadata JSONB NULL DEFAULT '{}'::jsonb
);

-- Trigger to keep updated_at in sync
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger WHERE tgname = 'trg_sessions_updated_at'
    ) THEN
        CREATE OR REPLACE FUNCTION public.set_timestamp_updated_at()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = TIMEZONE('utc'::text, NOW());
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;

        CREATE TRIGGER trg_sessions_updated_at
        BEFORE UPDATE ON public.sessions
        FOR EACH ROW EXECUTE FUNCTION public.set_timestamp_updated_at();
    END IF;
END $$;

-- Chat history linked to sessions
CREATE TABLE IF NOT EXISTS public.chat_history (
    id BIGSERIAL PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES public.sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user','assistant','system')),
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT TIMEZONE('utc'::text, NOW())
);

CREATE INDEX IF NOT EXISTS idx_chat_history_session_created
    ON public.chat_history (session_id, created_at DESC);

-- Embedding cache for deduplicated content vectors
-- Assume default dimension of 384 (MiniLM-L6-v2). Adjust via CHECK if needed later.
CREATE TABLE IF NOT EXISTS public.embedding_cache (
    id BIGSERIAL PRIMARY KEY,
    content_hash TEXT NOT NULL UNIQUE,
    content_preview TEXT NULL,
    embedding VECTOR(384) NOT NULL,
    usage_count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT TIMEZONE('utc'::text, NOW()),
    last_used TIMESTAMPTZ NULL,
    expires_at TIMESTAMPTZ NULL
);

CREATE INDEX IF NOT EXISTS idx_embedding_cache_last_used
    ON public.embedding_cache (last_used DESC NULLS LAST);

-- Rate limit tracking (optional persistent store)
CREATE TABLE IF NOT EXISTS public.rate_limit_tracking (
    key TEXT NOT NULL,
    window_start TIMESTAMPTZ NOT NULL,
    count INTEGER NOT NULL DEFAULT 0,
    metadata JSONB NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (key, window_start)
);

COMMIT;

