-- Phase 3: Vector Embeddings Migration
-- Enable pgvector extension (if not already enabled)
CREATE EXTENSION IF NOT EXISTS "vector";

-- Add new columns to embeddings table
ALTER TABLE public.embeddings 
ADD COLUMN IF NOT EXISTS model_version TEXT DEFAULT 'text-embedding-004',
ADD COLUMN IF NOT EXISTS embedding_metadata JSONB DEFAULT '{}'::jsonb,
ADD COLUMN IF NOT EXISTS processing_time_ms INTEGER,
ADD COLUMN IF NOT EXISTS similarity_cache JSONB DEFAULT '{}'::jsonb,
ADD COLUMN IF NOT EXISTS status TEXT DEFAULT 'completed' CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
ADD COLUMN IF NOT EXISTS error_message TEXT,
ADD COLUMN IF NOT EXISTS retry_count INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc'::text, NOW()) NOT NULL;

-- Create embedding_jobs table for batch processing
CREATE TABLE IF NOT EXISTS public.embedding_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID REFERENCES public.documents(id) ON DELETE CASCADE NOT NULL,
    chunk_ids UUID[] NOT NULL,
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'completed', 'failed', 'cancelled')),
    batch_size INTEGER DEFAULT 10,
    total_chunks INTEGER NOT NULL,
    processed_chunks INTEGER DEFAULT 0,
    failed_chunks INTEGER DEFAULT 0,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    estimated_completion TIMESTAMP WITH TIME ZONE,
    processing_metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc'::text, NOW()) NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc'::text, NOW()) NOT NULL
);

-- Create vector_search_cache table
CREATE TABLE IF NOT EXISTS public.vector_search_cache (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES public.profiles(id) ON DELETE CASCADE NOT NULL,
    query_hash TEXT NOT NULL,
    query_text TEXT NOT NULL,
    query_embedding VECTOR(768),
    search_results JSONB NOT NULL,
    similarity_threshold REAL DEFAULT 0.7,
    result_count INTEGER NOT NULL,
    search_metadata JSONB DEFAULT '{}'::jsonb,
    hit_count INTEGER DEFAULT 1,
    last_accessed TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc'::text, NOW()) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc'::text, NOW()) NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE DEFAULT (TIMEZONE('utc'::text, NOW()) + INTERVAL '1 hour')
);

-- Create embedding_cache table for duplicate detection
CREATE TABLE IF NOT EXISTS public.embedding_cache (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    content_hash TEXT UNIQUE NOT NULL,
    content_preview TEXT NOT NULL,
    embedding VECTOR(768) NOT NULL,
    model_version TEXT DEFAULT 'text-embedding-004',
    usage_count INTEGER DEFAULT 1,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc'::text, NOW()) NOT NULL,
    last_used TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc'::text, NOW()) NOT NULL
);

-- Create rate_limit_tracking table
CREATE TABLE IF NOT EXISTS public.rate_limit_tracking (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    service_name TEXT NOT NULL,
    user_id UUID REFERENCES public.profiles(id) ON DELETE CASCADE,
    requests_count INTEGER DEFAULT 0,
    tokens_used INTEGER DEFAULT 0,
    reset_time TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc'::text, NOW()) NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc'::text, NOW()) NOT NULL,
    UNIQUE(service_name, user_id, reset_time)
);

-- Create optimized indexes
DROP INDEX IF EXISTS embeddings_vector_idx;
CREATE INDEX embeddings_vector_cosine_idx ON public.embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX embeddings_vector_l2_idx ON public.embeddings USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);
CREATE INDEX embeddings_vector_ip_idx ON public.embeddings USING ivfflat (embedding vector_ip_ops) WITH (lists = 100);

-- Additional indexes for performance
CREATE INDEX IF NOT EXISTS embedding_jobs_document_id_idx ON public.embedding_jobs(document_id);
CREATE INDEX IF NOT EXISTS embedding_jobs_status_idx ON public.embedding_jobs(status);
CREATE INDEX IF NOT EXISTS embedding_jobs_created_at_idx ON public.embedding_jobs(created_at DESC);

CREATE INDEX IF NOT EXISTS vector_search_cache_user_id_idx ON public.vector_search_cache(user_id);
CREATE INDEX IF NOT EXISTS vector_search_cache_query_hash_idx ON public.vector_search_cache(query_hash);
CREATE INDEX IF NOT EXISTS vector_search_cache_expires_at_idx ON public.vector_search_cache(expires_at);

CREATE INDEX IF NOT EXISTS embedding_cache_content_hash_idx ON public.embedding_cache(content_hash);
CREATE INDEX IF NOT EXISTS embedding_cache_last_used_idx ON public.embedding_cache(last_used DESC);

CREATE INDEX IF NOT EXISTS rate_limit_tracking_service_user_idx ON public.rate_limit_tracking(service_name, user_id);
CREATE INDEX IF NOT EXISTS rate_limit_tracking_reset_time_idx ON public.rate_limit_tracking(reset_time);

-- Enable RLS for new tables
ALTER TABLE public.embedding_jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.vector_search_cache ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.embedding_cache ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.rate_limit_tracking ENABLE ROW LEVEL SECURITY;

-- RLS policies for embedding_jobs
CREATE POLICY "Users can view own embedding jobs" ON public.embedding_jobs
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM public.documents
            WHERE documents.id = embedding_jobs.document_id
            AND documents.user_id = auth.uid()
        )
    );

CREATE POLICY "Users can insert own embedding jobs" ON public.embedding_jobs
    FOR INSERT WITH CHECK (
        EXISTS (
            SELECT 1 FROM public.documents
            WHERE documents.id = embedding_jobs.document_id
            AND documents.user_id = auth.uid()
        )
    );

CREATE POLICY "Users can update own embedding jobs" ON public.embedding_jobs
    FOR UPDATE USING (
        EXISTS (
            SELECT 1 FROM public.documents
            WHERE documents.id = embedding_jobs.document_id
            AND documents.user_id = auth.uid()
        )
    );

-- RLS policies for vector_search_cache
CREATE POLICY "Users can view own search cache" ON public.vector_search_cache
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own search cache" ON public.vector_search_cache
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own search cache" ON public.vector_search_cache
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete own search cache" ON public.vector_search_cache
    FOR DELETE USING (auth.uid() = user_id);

-- RLS policies for embedding_cache (shared resource)
CREATE POLICY "All users can read embedding cache" ON public.embedding_cache
    FOR SELECT USING (true);

CREATE POLICY "All users can insert embedding cache" ON public.embedding_cache
    FOR INSERT WITH CHECK (true);

CREATE POLICY "All users can update embedding cache" ON public.embedding_cache
    FOR UPDATE USING (true);

-- RLS policies for rate_limit_tracking
CREATE POLICY "Users can view own rate limits" ON public.rate_limit_tracking
    FOR SELECT USING (auth.uid() = user_id OR user_id IS NULL);

CREATE POLICY "Users can insert own rate limits" ON public.rate_limit_tracking
    FOR INSERT WITH CHECK (auth.uid() = user_id OR user_id IS NULL);

CREATE POLICY "Users can update own rate limits" ON public.rate_limit_tracking
    FOR UPDATE USING (auth.uid() = user_id OR user_id IS NULL);

-- Create updated_at triggers for new tables
CREATE TRIGGER handle_updated_at_embedding_jobs
    BEFORE UPDATE ON public.embedding_jobs
    FOR EACH ROW EXECUTE FUNCTION public.handle_updated_at();

CREATE TRIGGER handle_updated_at_embeddings
    BEFORE UPDATE ON public.embeddings
    FOR EACH ROW EXECUTE FUNCTION public.handle_updated_at();

CREATE TRIGGER handle_updated_at_rate_limit_tracking
    BEFORE UPDATE ON public.rate_limit_tracking
    FOR EACH ROW EXECUTE FUNCTION public.handle_updated_at();

-- Function to clean up expired cache entries
CREATE OR REPLACE FUNCTION public.cleanup_expired_cache()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER := 0;
BEGIN
    -- Clean up expired search cache
    DELETE FROM public.vector_search_cache 
    WHERE expires_at < TIMEZONE('utc'::text, NOW());
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    -- Clean up old embedding cache (keep last 30 days)
    DELETE FROM public.embedding_cache 
    WHERE last_used < TIMEZONE('utc'::text, NOW()) - INTERVAL '30 days'
    AND usage_count = 1;
    
    -- Clean up old rate limit tracking (keep last 7 days)
    DELETE FROM public.rate_limit_tracking 
    WHERE reset_time < TIMEZONE('utc'::text, NOW()) - INTERVAL '7 days';
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to get vector similarity statistics
CREATE OR REPLACE FUNCTION public.get_vector_stats()
RETURNS TABLE(
    total_embeddings BIGINT,
    total_documents BIGINT,
    avg_chunks_per_document NUMERIC,
    cache_hit_rate NUMERIC,
    storage_size_mb NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(*)::BIGINT as total_embeddings,
        COUNT(DISTINCT dc.document_id)::BIGINT as total_documents,
        ROUND(COUNT(*)::NUMERIC / NULLIF(COUNT(DISTINCT dc.document_id), 0), 2) as avg_chunks_per_document,
        ROUND(
            COALESCE(
                (SELECT SUM(hit_count - 1)::NUMERIC / NULLIF(SUM(hit_count), 0) * 100
                 FROM public.vector_search_cache), 0
            ), 2
        ) as cache_hit_rate,
        ROUND(
            (pg_total_relation_size('public.embeddings') + 
             pg_total_relation_size('public.embedding_cache') + 
             pg_total_relation_size('public.vector_search_cache'))::NUMERIC / 1024 / 1024, 2
        ) as storage_size_mb
    FROM public.embeddings e
    JOIN public.document_chunks dc ON e.chunk_id = dc.id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to optimize vector indexes
CREATE OR REPLACE FUNCTION public.optimize_vector_indexes()
RETURNS TEXT AS $$
BEGIN
    -- Reindex vector indexes for better performance
    REINDEX INDEX embeddings_vector_cosine_idx;
    REINDEX INDEX embeddings_vector_l2_idx;
    REINDEX INDEX embeddings_vector_ip_idx;
    
    -- Analyze tables for better query planning
    ANALYZE public.embeddings;
    ANALYZE public.document_chunks;
    ANALYZE public.documents;
    
    RETURN 'Vector indexes optimized successfully';
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Create view for embedding job progress
CREATE OR REPLACE VIEW public.embedding_job_progress AS
SELECT 
    ej.id,
    ej.document_id,
    d.title as document_title,
    d.file_name,
    ej.status,
    ej.total_chunks,
    ej.processed_chunks,
    ej.failed_chunks,
    ROUND((ej.processed_chunks::NUMERIC / ej.total_chunks * 100), 2) as progress_percentage,
    ej.started_at,
    ej.completed_at,
    ej.estimated_completion,
    CASE 
        WHEN ej.status = 'completed' THEN ej.completed_at - ej.started_at
        WHEN ej.status = 'processing' THEN TIMEZONE('utc'::text, NOW()) - ej.started_at
        ELSE NULL
    END as processing_time,
    ej.error_message,
    ej.retry_count
FROM public.embedding_jobs ej
JOIN public.documents d ON ej.document_id = d.id;

-- Create similarity search function
CREATE OR REPLACE FUNCTION public.similarity_search(
    query_embedding vector(768),
    similarity_threshold float DEFAULT 0.7,
    match_count int DEFAULT 20
)
RETURNS TABLE(
    chunk_id UUID,
    document_id UUID,
    chunk_index INTEGER,
    content TEXT,
    chunk_metadata JSONB,
    chunk_created_at TIMESTAMP WITH TIME ZONE,
    user_id UUID,
    document_title TEXT,
    file_name TEXT,
    file_type TEXT,
    content_preview TEXT,
    upload_path TEXT,
    document_created_at TIMESTAMP WITH TIME ZONE,
    document_updated_at TIMESTAMP WITH TIME ZONE,
    model_version TEXT,
    processing_time_ms INTEGER,
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        dc.id AS chunk_id,
        dc.document_id,
        dc.chunk_index,
        dc.content,
        dc.metadata AS chunk_metadata,
        dc.created_at AS chunk_created_at,
        d.user_id,
        d.title AS document_title,
        d.file_name,
        d.file_type,
        d.content_preview,
        d.upload_path,
        d.created_at AS document_created_at,
        d.updated_at AS document_updated_at,
        e.model_version,
        e.processing_time_ms,
        1 - (e.embedding <=> query_embedding) AS similarity
    FROM public.embeddings e
    JOIN public.document_chunks dc ON e.chunk_id = dc.id
    JOIN public.documents d ON dc.document_id = d.id
    WHERE 
        e.status = 'completed'
        AND d.user_id = auth.uid()
        AND 1 - (e.embedding <=> query_embedding) >= similarity_threshold
    ORDER BY e.embedding <=> query_embedding ASC
    LIMIT match_count;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Create function to increment hit count
CREATE OR REPLACE FUNCTION public.increment_hit_count()
RETURNS INTEGER AS $$
BEGIN
    -- This will be used in UPDATE queries to increment hit_count
    RETURN 1;
END;
$$ LANGUAGE plpgsql;

-- Grant necessary permissions
GRANT SELECT ON public.embedding_job_progress TO authenticated;
GRANT EXECUTE ON FUNCTION public.cleanup_expired_cache() TO authenticated;
GRANT EXECUTE ON FUNCTION public.get_vector_stats() TO authenticated;
GRANT EXECUTE ON FUNCTION public.optimize_vector_indexes() TO authenticated;
GRANT EXECUTE ON FUNCTION public.similarity_search(vector(768), float, int) TO authenticated;
GRANT EXECUTE ON FUNCTION public.increment_hit_count() TO authenticated;

-- Create indexes on the view
CREATE INDEX IF NOT EXISTS embedding_job_progress_document_id_idx ON public.embedding_jobs(document_id);
CREATE INDEX IF NOT EXISTS embedding_job_progress_status_idx ON public.embedding_jobs(status);
