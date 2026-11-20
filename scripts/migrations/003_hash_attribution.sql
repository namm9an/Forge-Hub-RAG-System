-- Consolidate only duplicate derived chunks within one document, retaining
-- the earliest chunk ID, its content, all page attribution and a usable vector.
-- Stored historical citation JSON/text is intentionally not rewritten.
WITH mapping AS (
    SELECT id, first_value(id) OVER (
        PARTITION BY document_id, content_hash ORDER BY chunk_index, created_at, id
    ) AS keep_id FROM chunks
)
INSERT INTO embeddings(chunk_id, embedding, model_version)
SELECT DISTINCT ON (mapping.keep_id) mapping.keep_id, e.embedding, e.model_version
FROM mapping JOIN embeddings e ON e.chunk_id=mapping.id
ORDER BY mapping.keep_id, (mapping.id=mapping.keep_id) DESC, e.created_at
ON CONFLICT (chunk_id) DO NOTHING;

WITH mapping AS (
    SELECT id, first_value(id) OVER (
        PARTITION BY document_id, content_hash ORDER BY chunk_index, created_at, id
    ) AS keep_id FROM chunks
), pages AS (
    SELECT mapping.keep_id, array_agg(DISTINCT page ORDER BY page) AS all_pages
    FROM mapping JOIN chunks c ON c.id=mapping.id
    CROSS JOIN LATERAL unnest(
        CASE WHEN cardinality(c.page_numbers)>0 THEN c.page_numbers ELSE ARRAY[c.page_number] END
    ) AS page
    GROUP BY mapping.keep_id
)
UPDATE chunks SET page_numbers=pages.all_pages FROM pages WHERE chunks.id=pages.keep_id;

WITH ranked AS (
    SELECT id, row_number() OVER (
        PARTITION BY document_id, content_hash ORDER BY chunk_index, created_at, id
    ) AS position FROM chunks
)
DELETE FROM chunks WHERE id IN (SELECT id FROM ranked WHERE position>1);
CREATE UNIQUE INDEX IF NOT EXISTS chunks_document_hash_unique ON chunks(document_id, content_hash);
