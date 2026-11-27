ALTER TABLE chunks ADD COLUMN IF NOT EXISTS heading TEXT NOT NULL DEFAULT '';
ALTER TABLE chunks DROP COLUMN content_tsv;
ALTER TABLE chunks ADD COLUMN content_tsv tsvector GENERATED ALWAYS AS (
    to_tsvector('english', CASE WHEN heading = '' THEN content ELSE heading || chr(10) || content END)
) STORED;
CREATE INDEX idx_chunks_content_tsv ON chunks USING GIN(content_tsv);
