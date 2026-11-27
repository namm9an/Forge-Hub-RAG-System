-- Run with psql after versioned migrations, using the backend runtime connection.
\set ON_ERROR_STOP on
BEGIN READ ONLY;

DO $validate$
DECLARE
    app_schema text := current_schema();
    missing text;
BEGIN
    IF app_schema IS NULL THEN
        RAISE EXCEPTION 'The runtime search_path does not select an application schema';
    END IF;

    IF (SELECT count(*) FROM pg_catalog.pg_extension WHERE extname IN ('pgcrypto', 'vector')) <> 2 THEN
        RAISE EXCEPTION 'Required pgcrypto or vector extension is missing';
    END IF;

    SELECT string_agg(required.name, ', ') INTO missing
    FROM unnest(ARRAY['sessions', 'documents', 'chunks', 'embeddings', 'chat_history',
                      'auth_rate_limits', 'schema_migrations']) AS required(name)
    WHERE NOT EXISTS (
        SELECT 1 FROM pg_catalog.pg_tables
        WHERE schemaname = app_schema AND tablename = required.name
    );
    IF missing IS NOT NULL THEN
        RAISE EXCEPTION 'Missing tables in schema %: %', app_schema, missing;
    END IF;

    -- Keep these minimum versions in sync with app.db.check_schema().
    EXECUTE format(
        'SELECT string_agg(required.version, '', '') FROM unnest($1) AS required(version)
         WHERE NOT EXISTS (SELECT 1 FROM %I.schema_migrations WHERE version = required.version)',
        app_schema
    ) INTO missing USING ARRAY['001_legacy_columns', '002_private_storage',
                              '003_hash_attribution', '004_auth_rate_limits', '005_chunk_headings'];
    IF missing IS NOT NULL THEN
        RAISE EXCEPTION 'Missing migration versions in schema %: %', app_schema, missing;
    END IF;
END;
$validate$;

SELECT current_schema() AS application_schema, gen_random_uuid() AS test_uuid,
       '[0]'::vector AS test_vector;
COMMIT;
