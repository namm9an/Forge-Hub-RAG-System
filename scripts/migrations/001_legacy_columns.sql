-- Add columns before the fresh schema's indexes are evaluated.
-- Legacy sessions deliberately retain NULL ownership and cannot be claimed.
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS vector;
ALTER TABLE IF EXISTS sessions ADD COLUMN IF NOT EXISTS user_id TEXT;
ALTER TABLE IF EXISTS sessions ADD COLUMN IF NOT EXISTS last_activity TIMESTAMPTZ DEFAULT NOW();
ALTER TABLE IF EXISTS sessions ADD COLUMN IF NOT EXISTS total_messages INTEGER DEFAULT 0;
ALTER TABLE IF EXISTS sessions ADD COLUMN IF NOT EXISTS total_documents INTEGER DEFAULT 0;
ALTER TABLE IF EXISTS chunks ADD COLUMN IF NOT EXISTS content_hash TEXT;
ALTER TABLE IF EXISTS chunks ADD COLUMN IF NOT EXISTS page_number INTEGER NOT NULL DEFAULT 1;
ALTER TABLE IF EXISTS chunks ADD COLUMN IF NOT EXISTS page_numbers INTEGER[] NOT NULL DEFAULT '{}';
ALTER TABLE IF EXISTS chunks ADD COLUMN IF NOT EXISTS token_count INTEGER NOT NULL DEFAULT 0;
ALTER TABLE IF EXISTS chunks ADD COLUMN IF NOT EXISTS content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;
ALTER TABLE IF EXISTS chat_history ADD COLUMN IF NOT EXISTS retrieval_ms INTEGER;
ALTER TABLE IF EXISTS chat_history ADD COLUMN IF NOT EXISTS generation_ms INTEGER;
DO $$
BEGIN
    IF to_regclass('sessions') IS NOT NULL THEN
        UPDATE sessions SET last_activity=COALESCE(created_at, NOW()) WHERE last_activity IS NULL;
    END IF;
    IF to_regclass('chunks') IS NOT NULL THEN
        UPDATE chunks SET content_hash=encode(digest(content, 'sha256'), 'hex') WHERE content_hash IS NULL;
        ALTER TABLE chunks ALTER COLUMN content_hash SET NOT NULL;
    END IF;
END;
$$;
-- Remove the legacy global cleanup function; the replacement requires an owner.
DROP FUNCTION IF EXISTS cleanup_expired_sessions(interval);
