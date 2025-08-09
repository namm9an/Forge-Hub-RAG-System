-- Validation script to test PostgreSQL extensions
-- Run this after bootstrap to ensure everything is working

-- Test pgcrypto extension
SELECT gen_random_uuid() AS test_uuid;

-- Test vector extension
SELECT '[(0)]'::vector AS test_vector;

-- Show installed extensions
SELECT extname, extversion 
FROM pg_extension 
WHERE extname IN ('pgcrypto', 'vector')
ORDER BY extname;

-- Show created tables
SELECT tablename 
FROM pg_tables 
WHERE schemaname = 'public' 
AND tablename IN ('sessions', 'documents', 'chunks', 'embeddings', 'chat_history')
ORDER BY tablename;

-- Test creating a sample session
INSERT INTO sessions (session_id) 
VALUES ('test_session_' || gen_random_uuid()::text)
RETURNING id, session_id;

-- Clean up test data
DELETE FROM sessions WHERE session_id LIKE 'test_session_%';

-- Final success message
SELECT 'Database validation successful!' AS status;
