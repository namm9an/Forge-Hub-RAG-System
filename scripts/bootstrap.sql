-- Bootstrap script for RAG database
-- This script sets up required PostgreSQL extensions and initial schema

-- Enable required PostgreSQL extensions
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS vector;

-- Create sessions table
CREATE TABLE IF NOT EXISTS sessions (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  session_id TEXT UNIQUE NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  last_activity TIMESTAMPTZ DEFAULT NOW(),
  total_messages INTEGER DEFAULT 0,
  total_documents INTEGER DEFAULT 0
);

-- Create documents table
CREATE TABLE IF NOT EXISTS documents (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  session_id TEXT NOT NULL,
  filename TEXT NOT NULL,
  file_size BIGINT NOT NULL,
  total_pages INTEGER,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create chunks table
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

-- Create embeddings table
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

-- Create chat history table
CREATE TABLE IF NOT EXISTS chat_history (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  session_id TEXT NOT NULL,
  user_message TEXT NOT NULL,
  assistant_message TEXT NOT NULL,
  source_documents JSONB DEFAULT '[]',
  processing_time_ms INTEGER,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Validation query to test extensions
DO $$
BEGIN
    -- Test pgcrypto
    PERFORM gen_random_uuid();
    RAISE NOTICE 'pgcrypto extension is working';
    
    -- Test vector
    PERFORM '[(0)]'::vector;
    RAISE NOTICE 'vector extension is working';
END $$;
