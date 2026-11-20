"""Offline upgrade contracts; reranking cases exercise fallback scoring, not neural quality."""

import pytest
import numpy as np
from pathlib import Path
from httpx import AsyncClient, ASGITransport
from app import app
from app.config import get_settings
from app.documents import recursive_token_chunk, count_tokens
from app.schemas import SearchResultItem, ConfidenceLevel, ChatRequest
from app.db import _memory_store, execute, fetchval, SCHEMA_SQL
from app.gating import (
    CrossEncoderReranker,
    get_reranker,
    EvidenceEvaluator,
    REFUSAL_MESSAGE,
    prune_and_format_context,
)
from app.retrieval import compute_rrf_fusion


def test_chunking_350_token_limit_and_defaults():
    """Verify Upgrade 1: chunk_size_tokens capped at 350, chunk_overlap at 50."""
    settings = get_settings()
    assert settings.chunk_size_tokens == 350
    assert settings.chunk_overlap_tokens == 50

    # Long text (~1200 tokens)
    long_sentences = [
        f"Paragraph {i} examines the distributed consensus protocols and HNSW vector index construction with ef_construction."
        for i in range(70)
    ]
    long_text = "\n\n".join(long_sentences)
    chunks = recursive_token_chunk(long_text)

    assert len(chunks) > 1
    for chunk_text, token_cnt in chunks:
        assert token_cnt <= 350, f"Chunk exceeded 350 tokens: {token_cnt}"
        assert count_tokens(chunk_text) <= 350


def test_hnsw_schema_and_cascade_ddl():
    """Verify Upgrade 2: HNSW vector index and ON DELETE CASCADE in both bootstrap.sql and db.py SCHEMA_SQL."""
    # Find bootstrap.sql from relative paths
    bootstrap_candidates = [
        Path("../../../scripts/bootstrap.sql"),
        Path(__file__).resolve().parents[3] / "scripts" / "bootstrap.sql",
        Path("scripts/bootstrap.sql"),
    ]
    bootstrap_sql = None
    for p in bootstrap_candidates:
        if p.exists():
            bootstrap_sql = p.read_text(encoding="utf-8")
            break

    if bootstrap_sql:
        assert "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)" in bootstrap_sql
        assert "REFERENCES sessions(session_id) ON DELETE CASCADE" in bootstrap_sql

    # Always verify in app/db.py SCHEMA_SQL
    assert "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)" in SCHEMA_SQL
    assert "session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE" in SCHEMA_SQL


def test_websearch_to_tsquery_in_retrieval():
    """Verify Upgrade 3: websearch_to_tsquery in retrieval.py instead of plainto_tsquery."""
    from app.retrieval import _sparse_search
    import inspect
    source = inspect.getsource(_sparse_search)
    assert "websearch_to_tsquery" in source
    assert "plainto_tsquery" not in source


def test_stage2_cross_encoder_reranker():
    """Verify Upgrade 4: Stage 2 Neural Cross-Encoder Reranker & fallback semantic scoring."""
    reranker = CrossEncoderReranker(model_name="ms-marco-TinyBERT-L-2-v2")

    c1 = SearchResultItem(
        chunk_id="chunk_1",
        document_id="doc_1",
        filename="db.pdf",
        page_number=1,
        score=0.03,
        dense_score=0.45,
        sparse_score=0.20,
        content="General database indexing techniques include B-trees and hash indexes.",
    )
    c2 = SearchResultItem(
        chunk_id="chunk_2",
        document_id="doc_1",
        filename="db.pdf",
        page_number=2,
        score=0.02,
        dense_score=0.85,
        sparse_score=0.70,
        content="Write-ahead logging (WAL) guarantees transaction durability by recording modifications before disk writes.",
    )

    query = "How does write-ahead logging provide durability?"
    reranked = reranker.rerank(query=query, candidates=[c1, c2])

    assert len(reranked) == 2
    # c2 is far more relevant to WAL durability than c1, should be ranked #1
    assert reranked[0].chunk_id == "chunk_2"
    assert reranked[0].rerank_score is not None
    assert reranked[1].rerank_score is not None
    assert reranked[0].rerank_score > reranked[1].rerank_score


def test_stage2_cross_encoder_dict_candidates():
    """Verify Upgrade 4: CrossEncoderReranker safely handles dictionary candidates without crashing."""
    reranker = CrossEncoderReranker(model_name="ms-marco-TinyBERT-L-2-v2")

    dict_candidates = [
        {
            "chunk_id": "dict_1",
            "content": "Supervised fine-tuning of neural language models.",
            "dense_score": 0.40,
            "sparse_score": 0.10,
            "score": 0.015,
        },
        {
            "chunk_id": "dict_2",
            "content": "PostgreSQL Write-Ahead Logging WAL enforces durability in transaction rollback and recovery.",
            "dense_score": 0.88,
            "sparse_score": 0.65,
            "score": 0.018,
        },
    ]

    res = reranker.rerank(query="WAL durability recovery", candidates=dict_candidates)
    assert len(res) == 2
    assert res[0]["chunk_id"] == "dict_2"
    assert res[0]["rerank_score"] > res[1]["rerank_score"]


def test_evidence_evaluator_with_rerank_score():
    """Verify EvidenceEvaluator accounts for Stage 2 rerank_score."""
    evaluator = EvidenceEvaluator(default_threshold=0.35)

    c = SearchResultItem(
        chunk_id="c_high",
        document_id="d1",
        filename="distributed.pdf",
        page_number=1,
        score=0.04,
        dense_score=0.60,
        sparse_score=0.30,
        rerank_score=0.75,
        content="Raft algorithm guarantees state machine replication across distributed servers.",
    )

    conf, refuse, _ = evaluator.evaluate(query="Explain Raft state machine replication", candidates=[c])
    assert conf == ConfidenceLevel.HIGH
    assert refuse is False


def test_evidence_evaluator_prevents_false_refusal_on_high_rerank():
    """Verify high neural rerank score prevents false refusal even when lexical term match is zero."""
    evaluator = EvidenceEvaluator(default_threshold=0.35)

    # Paraphrased candidate with zero exact lexical token overlap with query
    c = SearchResultItem(
        chunk_id="c_para",
        document_id="d1",
        filename="distributed.pdf",
        page_number=1,
        score=0.02,
        dense_score=0.48,   # below 0.50
        sparse_score=0.0,   # zero sparse overlap
        rerank_score=0.72,  # high neural cross-encoder confidence
        content="Primary node coordination ensures replica convergence across cluster machines.",
    )

    # Query uses different words but semantic reranker identified it
    conf, refuse, _ = evaluator.evaluate(query="How does leader election synchronize servers?", candidates=[c])
    assert refuse is False
    assert conf != ConfidenceLevel.LOW


def test_prune_and_format_context_preserves_rerank_score():
    """Verify prune_and_format_context carries rerank_score into Source items."""
    c = SearchResultItem(
        chunk_id="c_test",
        document_id="d1",
        filename="file.pdf",
        page_number=3,
        score=0.025,
        dense_score=0.75,
        sparse_score=0.40,
        rerank_score=0.82,
        content="First sentence is relevant. Second sentence is extraneous noise.",
    )
    context_str, sources = prune_and_format_context(query="First sentence", candidates=[c])
    assert len(sources) == 1
    assert sources[0].rerank_score == 0.82
    assert "[Doc: file.pdf, Page: 3, Chunk: c_test]" in context_str


@pytest.mark.asyncio
async def test_db_memory_store_session_cascading_deletion():
    """Verify session cascading deletion cleans up all related documents, chunks, and chat history."""
    _memory_store.reset()

    sid = "cascade_test_session"
    await execute("INSERT INTO SESSIONS (session_id, user_id) VALUES ($1, $2)", sid, "contract-user")

    doc_id = await fetchval(
        "INSERT INTO DOCUMENTS (session_id, filename, file_size, total_pages) VALUES ($1, $2, $3, $4) RETURNING id",
        sid, "test.pdf", 500, 1
    )

    chunk_id = await fetchval(
        "INSERT INTO CHUNKS (document_id, chunk_index, content, content_hash, page_number, token_count) VALUES ($1, $2, $3, $4, $5, $6) RETURNING id",
        doc_id, 0, "Test chunk content", "hash123", 1, 4
    )

    emb = np.zeros(768, dtype=np.float32)
    await execute("INSERT INTO EMBEDDINGS (chunk_id, embedding, model_version) VALUES ($1, $2, $3)", chunk_id, emb, "test")

    await execute(
        "INSERT INTO CHAT_HISTORY (session_id, user_message, assistant_message, source_documents, processing_time_ms) VALUES ($1, $2, $3, $4, $5)",
        sid, "hello", "hi", "[]", 10
    )

    assert sid in _memory_store.sessions
    assert doc_id in _memory_store.documents
    assert chunk_id in _memory_store.chunks
    assert chunk_id in _memory_store.embeddings
    assert len([h for h in _memory_store.chat_history if h["session_id"] == sid]) == 1

    # Delete session
    await execute("DELETE FROM SESSIONS WHERE session_id = $1", sid)

    # Everything must cascade away
    assert sid not in _memory_store.sessions
    assert doc_id not in _memory_store.documents
    assert chunk_id not in _memory_store.chunks
    assert chunk_id not in _memory_store.embeddings
    assert len([h for h in _memory_store.chat_history if h["session_id"] == sid]) == 0


@pytest.mark.asyncio
async def test_chat_stream_sse_and_chunked_formats():
    """Verify Upgrade 5: streaming supports both SSE (text/event-stream) and chunked transfer encoding."""
    _memory_store.reset()
    sid = "stream_format_session"
    await execute("INSERT INTO SESSIONS (session_id, user_id) VALUES ($1, $2)", sid, "contract-user")

    doc_id = await fetchval(
        "INSERT INTO DOCUMENTS (session_id, filename, file_size, total_pages) VALUES ($1, $2, $3, $4) RETURNING id",
        sid, "stream.pdf", 100, 1
    )
    chunk_id = await fetchval(
        "INSERT INTO CHUNKS (document_id, chunk_index, content, content_hash, page_number, token_count) VALUES ($1, $2, $3, $4, $5, $6) RETURNING id",
        doc_id, 0, "Streaming responses support SSE protocol and raw text chunks.", "hash_stream", 1, 10
    )
    emb = np.ones(768, dtype=np.float32) / np.sqrt(768)
    await execute("INSERT INTO EMBEDDINGS (chunk_id, embedding, model_version) VALUES ($1, $2, $3)", chunk_id, emb, "test")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Case 1: Default / Raw text chunked transfer (Accept: */*)
        resp_chunked = await ac.post(
            "/api/chat/stream",
            json={"query": "Streaming responses", "session_id": sid, "threshold": 0.10},
        )
        assert resp_chunked.status_code == 200
        assert "text/plain" in resp_chunked.headers.get("content-type", "")
        assert resp_chunked.headers.get("x-accel-buffering") == "no"
        assert "Streaming responses" in resp_chunked.text

        # Case 2: Explicit SSE (Accept: text/event-stream)
        resp_sse = await ac.post(
            "/api/chat/stream",
            headers={"Accept": "text/event-stream"},
            json={"query": "Streaming responses", "session_id": sid, "threshold": 0.10},
        )
        assert resp_sse.status_code == 200
        assert "text/event-stream" in resp_sse.headers.get("content-type", "")
        assert resp_sse.headers.get("x-accel-buffering") == "no"
        assert "data: " in resp_sse.text
        assert "data: [DONE]" in resp_sse.text
