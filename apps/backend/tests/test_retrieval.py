"""Offline retrieval contracts, not semantic-model or PostgreSQL index quality."""

import pytest
import numpy as np
from httpx import AsyncClient, ASGITransport
from app import app
from app.db import _memory_store, execute, fetchval
from app.retrieval import compute_rrf_fusion, _dense_search, _sparse_search
from app.schemas import HybridSearchRequest
from app.config import get_settings


def test_rrf_score_calculation_isolated():
    """
    Verify mathematical Reciprocal Rank Fusion (RRF) calculation (Rule C-3).
    Formula: score(d) = (w_dense / (k + rank_dense)) + (w_sparse / (k + rank_sparse))
    """
    k = 60
    w_dense = 0.6
    w_sparse = 0.4

    dense_candidates = [
        {"chunk_id": "chunk_A", "dense_score": 0.95, "content": "Text A", "document_id": "doc1", "filename": "doc1.pdf", "page_number": 1},
        {"chunk_id": "chunk_B", "dense_score": 0.85, "content": "Text B", "document_id": "doc1", "filename": "doc1.pdf", "page_number": 2},
    ]

    sparse_candidates = [
        {"chunk_id": "chunk_B", "sparse_score": 0.75, "content": "Text B", "document_id": "doc1", "filename": "doc1.pdf", "page_number": 2},
        {"chunk_id": "chunk_C", "sparse_score": 0.65, "content": "Text C", "document_id": "doc2", "filename": "doc2.pdf", "page_number": 1},
    ]

    results = compute_rrf_fusion(
        dense_results=dense_candidates,
        sparse_results=sparse_candidates,
        k=k,
        dense_weight=w_dense,
        sparse_weight=w_sparse,
        top_k=5,
    )

    result_map = {r.chunk_id: r for r in results}

    # chunk_A: dense rank 1, sparse rank None
    expected_score_A = round(w_dense / (k + 1), 6)
    assert result_map["chunk_A"].score == expected_score_A
    assert result_map["chunk_A"].dense_rank == 1
    assert result_map["chunk_A"].sparse_rank is None

    # chunk_B: dense rank 2, sparse rank 1
    expected_score_B = round((w_dense / (k + 2)) + (w_sparse / (k + 1)), 6)
    assert result_map["chunk_B"].score == expected_score_B
    assert result_map["chunk_B"].dense_rank == 2
    assert result_map["chunk_B"].sparse_rank == 1

    # chunk_C: dense rank None, sparse rank 2
    expected_score_C = round(w_sparse / (k + 2), 6)
    assert result_map["chunk_C"].score == expected_score_C
    assert result_map["chunk_C"].dense_rank is None
    assert result_map["chunk_C"].sparse_rank == 2

    # Verification: chunk_B has highest combined score, should be ranked #1
    assert results[0].chunk_id == "chunk_B"
    assert results[0].score > results[1].score


def test_dense_and_sparse_scores_diverge_never_mocked():
    """
    Verify Rule C-1: Dense scores and sparse scores are NEVER identical or mocked proxies.
    Dense scores reflect vector similarity; sparse scores reflect lexical frequency.
    """
    dense_candidates = [
        {"chunk_id": "chunk_1", "dense_score": 0.8841, "content": "Machine learning architectures", "document_id": "d1", "filename": "ai.pdf", "page_number": 1},
    ]
    sparse_candidates = [
        {"chunk_id": "chunk_1", "sparse_score": 0.3125, "content": "Machine learning architectures", "document_id": "d1", "filename": "ai.pdf", "page_number": 1},
    ]

    results = compute_rrf_fusion(
        dense_results=dense_candidates,
        sparse_results=sparse_candidates,
        k=60,
        dense_weight=0.6,
        sparse_weight=0.4,
        top_k=5,
    )

    item = results[0]
    assert item.dense_score is not None
    assert item.sparse_score is not None
    # Crucial anti-slop check: dense and sparse scores must NOT be identical
    assert abs(item.dense_score - item.sparse_score) > 0.05
    assert item.dense_score != item.sparse_score


@pytest.mark.asyncio
async def test_hybrid_search_end_to_end_and_attribution():
    """
    Verify Rule C-4 (parallel execution) and Rule C-5 (attribution metadata).
    Tests full pipeline with isolated session, chunk insertion, and embedding.
    """
    _memory_store.reset()

    session_id = "test_hybrid_session"
    await execute("INSERT INTO SESSIONS (session_id, user_id) VALUES ($1, $2)", session_id, "contract-user")

    doc_id = await fetchval(
        "INSERT INTO DOCUMENTS (session_id, filename, file_size, total_pages) VALUES ($1, $2, $3, $4) RETURNING id",
        session_id,
        "database_internals.pdf",
        204800,
        10,
    )

    # Insert two chunks
    chunk1_id = await fetchval(
        "INSERT INTO CHUNKS (document_id, chunk_index, content, content_hash, page_number, token_count) VALUES ($1, $2, $3, $4, $5, $6) RETURNING id",
        doc_id,
        0,
        "PostgreSQL utilizes write-ahead logging (WAL) and MVCC for transactional ACID guarantees.",
        "hash1",
        3,
        15,
    )

    chunk2_id = await fetchval(
        "INSERT INTO CHUNKS (document_id, chunk_index, content, content_hash, page_number, token_count) VALUES ($1, $2, $3, $4, $5, $6) RETURNING id",
        doc_id,
        1,
        "HNSW index creates hierarchical graphs for logarithmic approximate nearest neighbor vector search.",
        "hash2",
        5,
        16,
    )

    # Insert deterministic embeddings (768-dim)
    emb1 = np.zeros(768, dtype=np.float32)
    emb1[0] = 1.0
    await execute("INSERT INTO EMBEDDINGS (chunk_id, embedding, model_version) VALUES ($1, $2, $3)", chunk1_id, emb1, "test-model")

    emb2 = np.zeros(768, dtype=np.float32)
    emb2[1] = 1.0
    await execute("INSERT INTO EMBEDDINGS (chunk_id, embedding, model_version) VALUES ($1, $2, $3)", chunk2_id, emb2, "test-model")

    # Perform hybrid search via API endpoint
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post(
            "/api/search/hybrid",
            json={
                "query": "PostgreSQL write-ahead logging WAL ACID",
                "session_id": session_id,
                "top_k": 5,
                "dense_weight": 0.6,
                "sparse_weight": 0.4,
                "rrf_k": 60,
            },
        )
        assert resp.status_code == 200
        data = resp.json()

        assert "results" in data
        assert "retrieval_ms" in data
        assert data["total_results"] >= 1

        top_hit = data["results"][0]
        # Rule C-5 Attribution Output
        assert "chunk_id" in top_hit
        assert "score" in top_hit
        assert "content" in top_hit
        assert "document_id" in top_hit
        assert "filename" in top_hit
        assert "page_number" in top_hit
        assert top_hit["filename"] == "database_internals.pdf"
        assert top_hit["page_number"] in [3, 5]


@pytest.mark.asyncio
async def test_session_isolation_in_retrieval():
    """
    Verify Rule B-5: Queries for session A never return chunks from session B.
    """
    _memory_store.reset()

    # Session A
    await execute("INSERT INTO SESSIONS (session_id, user_id) VALUES ($1, $2)", "session_A", "contract-user")
    doc_A = await fetchval(
        "INSERT INTO DOCUMENTS (session_id, filename, file_size, total_pages) VALUES ($1, $2, $3, $4) RETURNING id",
        "session_A", "doc_A.pdf", 100, 1
    )
    chunk_A = await fetchval(
        "INSERT INTO CHUNKS (document_id, chunk_index, content, content_hash, page_number, token_count) VALUES ($1, $2, $3, $4, $5, $6) RETURNING id",
        doc_A, 0, "Confidential financial statement for Company A", "hA", 1, 10
    )
    emb_A = np.ones(768, dtype=np.float32) / np.sqrt(768)
    await execute("INSERT INTO EMBEDDINGS (chunk_id, embedding, model_version) VALUES ($1, $2, $3)", chunk_A, emb_A, "test")

    # Session B
    await execute("INSERT INTO SESSIONS (session_id, user_id) VALUES ($1, $2)", "session_B", "contract-user")
    doc_B = await fetchval(
        "INSERT INTO DOCUMENTS (session_id, filename, file_size, total_pages) VALUES ($1, $2, $3, $4) RETURNING id",
        "session_B", "doc_B.pdf", 100, 1
    )
    chunk_B = await fetchval(
        "INSERT INTO CHUNKS (document_id, chunk_index, content, content_hash, page_number, token_count) VALUES ($1, $2, $3, $4, $5, $6) RETURNING id",
        doc_B, 0, "Public documentation for Company B", "hB", 1, 10
    )
    emb_B = np.ones(768, dtype=np.float32) / np.sqrt(768)
    await execute("INSERT INTO EMBEDDINGS (chunk_id, embedding, model_version) VALUES ($1, $2, $3)", chunk_B, emb_B, "test")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Search under session_B for confidential term in session_A
        resp = await ac.post(
            "/api/search/hybrid",
            json={
                "query": "Confidential financial statement",
                "session_id": "session_B",
                "top_k": 5,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        for res in data["results"]:
            assert res["filename"] != "doc_A.pdf"
            assert "Company A" not in res["content"]
