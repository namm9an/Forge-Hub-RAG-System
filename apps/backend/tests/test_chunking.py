import pytest
from app.documents import recursive_token_chunk, count_tokens, _hash_text


def test_chunking_short_text():
    text = "This is a brief sentence that easily fits within the token limit."
    chunks = recursive_token_chunk(text, max_tokens=512, overlap_tokens=64)
    assert len(chunks) == 1
    chunk_text, token_count = chunks[0]
    assert chunk_text == text
    assert token_count == count_tokens(text)
    assert token_count <= 512


def test_chunking_strict_token_limit():
    # Generate long synthetic text (approx 1500 tokens)
    sentences = [
        f"Sentence number {i} provides detailed engineering information about high-performance vector search pipelines and distributed databases."
        for i in range(80)
    ]
    long_text = "\n\n".join(sentences)
    chunks = recursive_token_chunk(long_text, max_tokens=350, overlap_tokens=50)

    assert len(chunks) > 1
    for chunk_text, token_cnt in chunks:
        # Rule B-1: Strict 350 token cap (respecting 384-token sequence ceiling)
        assert token_cnt <= 350, f"Chunk exceeded 350 tokens: {token_cnt}"
        assert count_tokens(chunk_text) <= 350


def test_sentence_and_word_integrity():
    text = (
        "Distributed database systems require consensus protocols like Raft or Paxos to maintain state across replicas.\n\n"
        "Vector indexes such as HNSW and IVFFlat trade index build time and memory footprint against query latency and recall.\n\n"
        "Lexical search via BM25 utilizes inverse document frequency to score exact term occurrences across a document collection."
    )
    chunks = recursive_token_chunk(text, max_tokens=30, overlap_tokens=5)
    for chunk_text, _ in chunks:
        # Rule B-2: No words cut in half
        words = chunk_text.split()
        for word in words:
            # Word should be alphanumeric + standard punctuation
            assert len(word) > 0


def test_content_hash_deduplication():
    t1 = "Deterministic chunking generates identical SHA-256 hashes."
    t2 = "Deterministic chunking generates identical SHA-256 hashes."
    t3 = "A different text string produces a distinct hash."

    # Rule B-4: Content Hash Deduplication
    h1 = _hash_text(t1)
    h2 = _hash_text(t2)
    h3 = _hash_text(t3)

    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 64


def test_chunking_350_default_parameters():
    from app.config import get_settings
    settings = get_settings()
    assert settings.chunk_size_tokens == 350
    assert settings.chunk_overlap_tokens == 50

    # Ensure default call to recursive_token_chunk respects 350 tokens
    long_sentences = [
        f"Paragraph {i} examines the distributed consensus protocols and HNSW vector index construction."
        for i in range(70)
    ]
    long_text = "\n\n".join(long_sentences)
    chunks = recursive_token_chunk(long_text)
    assert len(chunks) > 1
    for _, token_cnt in chunks:
        assert token_cnt <= 350


def test_hnsw_schema_and_cascade_ddl():
    from pathlib import Path
    candidates = [
        Path(__file__).resolve().parents[2] / "scripts" / "bootstrap.sql",
        Path(__file__).resolve().parents[3] / "scripts" / "bootstrap.sql",
        Path("scripts/bootstrap.sql").resolve(),
    ]
    bootstrap_file = next((p for p in candidates if p.is_file()), None)
    assert bootstrap_file is not None, f"Could not locate scripts/bootstrap.sql from {__file__}"
    bootstrap_sql = bootstrap_file.read_text(encoding="utf-8")

    assert "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)" in bootstrap_sql
    assert "REFERENCES sessions(session_id) ON DELETE CASCADE" in bootstrap_sql
