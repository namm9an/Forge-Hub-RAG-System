-- 0002_functions.sql
-- Defines helper functions: get_vector_stats() and cleanup_expired_cache()

BEGIN;

-- get_vector_stats: summarize vector/embedding usage across tables
CREATE OR REPLACE FUNCTION public.get_vector_stats()
RETURNS JSONB
LANGUAGE plpgsql
AS $$
DECLARE
    result JSONB;
    embedding_rows BIGINT := 0;
    cache_rows BIGINT := 0;
    cache_last_used TIMESTAMPTZ := NULL;
    cache_expired BIGINT := 0;
BEGIN
    -- Optional: embeddings table may or may not exist
    IF to_regclass('public.embeddings') IS NOT NULL THEN
        EXECUTE 'SELECT COUNT(*) FROM public.embeddings' INTO embedding_rows;
    END IF;

    SELECT COUNT(*), MAX(last_used) INTO cache_rows, cache_last_used FROM public.embedding_cache;
    SELECT COUNT(*) INTO cache_expired FROM public.embedding_cache WHERE expires_at IS NOT NULL AND expires_at < TIMEZONE('utc'::text, NOW());

    result := jsonb_build_object(
        'embeddings_count', embedding_rows,
        'embedding_cache_count', cache_rows,
        'embedding_cache_last_used', cache_last_used,
        'embedding_cache_expired', cache_expired
    );

    RETURN result;
END;
$$;

-- cleanup_expired_cache: delete expired cache rows and return how many were removed
CREATE OR REPLACE FUNCTION public.cleanup_expired_cache()
RETURNS INTEGER
LANGUAGE plpgsql
AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM public.embedding_cache
    WHERE expires_at IS NOT NULL AND expires_at < TIMEZONE('utc'::text, NOW());

    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$;

COMMIT;

