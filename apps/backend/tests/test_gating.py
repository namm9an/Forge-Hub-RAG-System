"""Offline gating/streaming contracts; provider and model quality need separate evaluation."""

import pytest
import numpy as np
from httpx import AsyncClient, ASGITransport
from app import app
from app.db import _memory_store, execute, fetchval
from app.schemas import SearchResultItem, ConfidenceLevel
from app.gating import (
    EvidenceEvaluator,
    prune_chunk_sentences,
    score_sentence_relevance,
    REFUSAL_MESSAGE,
)


def test_sentence_level_noise_pruner():
    """
    Verify Rule D-2: Sentence-Level Noise Pruner.
    Filters out irrelevant noise sentences while preserving high-relevance sentences.
    """
    chunk_text = (
        "PostgreSQL uses write-ahead logging to guarantee transaction durability and crash recovery. "
        "The project was originally called POSTGRES at UC Berkeley starting in 1986. "
        "WAL ensures that no committed transaction is lost during a sudden system shutdown."
    )
    query = "How does write-ahead logging WAL provide transaction durability?"

    pruned_text, orig_cnt, ret_cnt = prune_chunk_sentences(chunk_text, query, min_sentence_relevance=0.10)

    # 3 sentences originally
    assert orig_cnt == 3
    # The historical trivia sentence about Berkeley 1986 should be pruned
    assert "UC Berkeley starting in 1986" not in pruned_text
    # The relevant sentences about WAL and durability must be retained
    assert "write-ahead logging" in pruned_text
    assert "durability" in pruned_text
    assert ret_cnt < orig_cnt


def test_evidence_evaluator_refusal_on_low_confidence():
    """
    Verify Rule D-1: Evidence Confidence Evaluator triggers calibrated refusal
    when candidate chunks have low similarity / no query term coverage.
    """
    evaluator = EvidenceEvaluator(default_threshold=0.35)

    low_conf_candidate = SearchResultItem(
        chunk_id="c_low",
        document_id="d1",
        filename="distributed_systems.pdf",
        page_number=1,
        score=0.015,
        dense_score=0.12,  # Well below 0.35 threshold
        sparse_score=0.0,
        content="Raft protocol relies on randomized election timers to avoid split votes.",
    )

    out_of_domain_query = "What is the traditional Italian recipe for baking sourdough bread?"

    confidence, should_refuse, msg = evaluator.evaluate(
        query=out_of_domain_query,
        candidates=[low_conf_candidate],
        threshold=0.35,
    )

    assert confidence == ConfidenceLevel.LOW
    assert should_refuse is True
    assert msg == REFUSAL_MESSAGE


def test_evidence_evaluator_pass_on_high_confidence():
    """
    Verify Evidence Evaluator allows high confidence candidates through.
    """
    evaluator = EvidenceEvaluator(default_threshold=0.35)

    high_conf_candidate = SearchResultItem(
        chunk_id="c_high",
        document_id="d1",
        filename="database_internals.pdf",
        page_number=3,
        score=0.032,
        dense_score=0.88,
        sparse_score=0.65,
        content="Write-ahead logging (WAL) records every state modification to persistent storage prior to writing pages to disk.",
    )

    query = "Explain how write-ahead logging records modifications to persistent storage"

    confidence, should_refuse, msg = evaluator.evaluate(
        query=query,
        candidates=[high_conf_candidate],
        threshold=0.35,
    )

    assert confidence == ConfidenceLevel.HIGH
    assert should_refuse is False
    assert msg == ""


def test_evidence_evaluator_empty_candidates():
    """
    Verify empty retrieval candidates immediately triggers calibrated refusal.
    """
    evaluator = EvidenceEvaluator()
    confidence, should_refuse, msg = evaluator.evaluate(
        query="Any query with zero candidates",
        candidates=[],
    )
    assert confidence == ConfidenceLevel.LOW
    assert should_refuse is True
    assert msg == REFUSAL_MESSAGE


@pytest.mark.asyncio
async def test_chat_endpoint_refusal_gate_integration():
    """
    Verify Rule D-1 end-to-end: Out-of-domain query against ingested documents returns
    the calibrated refusal message without hallucinating.
    """
    _memory_store.reset()

    session_id = "test_gating_session"
    await execute("INSERT INTO SESSIONS (session_id, user_id) VALUES ($1, $2)", session_id, "contract-user")

    doc_id = await fetchval(
        "INSERT INTO DOCUMENTS (session_id, filename, file_size, total_pages) VALUES ($1, $2, $3, $4) RETURNING id",
        session_id,
        "database_engines.pdf",
        10240,
        2,
    )

    chunk_id = await fetchval(
        "INSERT INTO CHUNKS (document_id, chunk_index, content, content_hash, page_number, token_count) VALUES ($1, $2, $3, $4, $5, $6) RETURNING id",
        doc_id,
        0,
        "LSM trees optimize write performance by sequentially appending updates to an in-memory memtable and SSTables.",
        "hash_lsm",
        1,
        18,
    )

    # Embedding with low alignment to out-of-domain query
    emb = np.zeros(768, dtype=np.float32)
    emb[10] = 1.0
    await execute("INSERT INTO EMBEDDINGS (chunk_id, embedding, model_version) VALUES ($1, $2, $3)", chunk_id, emb, "test")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Out-of-domain query
        resp = await ac.post(
            "/api/chat",
            json={
                "query": "How many moons does Jupiter have in our solar system?",
                "session_id": session_id,
                "threshold": 0.50,
            },
        )
        assert resp.status_code == 200
        data = resp.json()

        # Rule D-1: Calibrated refusal
        assert data["refusal_triggered"] is True
        assert data["confidence"] == "LOW"
        assert REFUSAL_MESSAGE in data["answer"]
        assert data["generation_ms"] == 0
        assert data["retrieval_ms"] is not None


@pytest.mark.asyncio
async def test_chat_stream_delivers_tokens():
    """
    Verify Rule D-4: Streaming delivery returns tokens via SSE or chunked transfer.
    """
    _memory_store.reset()

    session_id = "test_stream_session"
    await execute("INSERT INTO SESSIONS (session_id, user_id) VALUES ($1, $2)", session_id, "contract-user")

    doc_id = await fetchval(
        "INSERT INTO DOCUMENTS (session_id, filename, file_size, total_pages) VALUES ($1, $2, $3, $4) RETURNING id",
        session_id,
        "cloud_architecture.pdf",
        5000,
        1,
    )

    chunk_id = await fetchval(
        "INSERT INTO CHUNKS (document_id, chunk_index, content, content_hash, page_number, token_count) VALUES ($1, $2, $3, $4, $5, $6) RETURNING id",
        doc_id,
        0,
        "Kubernetes orchestration schedules containers across node pools with declared replica states.",
        "hash_k8s",
        1,
        14,
    )

    emb = np.ones(768, dtype=np.float32) / np.sqrt(768)
    await execute("INSERT INTO EMBEDDINGS (chunk_id, embedding, model_version) VALUES ($1, $2, $3)", chunk_id, emb, "test")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post(
            "/api/chat/stream",
            json={
                "query": "Kubernetes orchestration replica states",
                "session_id": session_id,
                "threshold": 0.10,
            },
        )
        assert resp.status_code == 200
        content = resp.text
        assert len(content) > 0
        # Verify citation in streamed output (Rule D-3)
        assert "[Doc: cloud_architecture.pdf" in content or "Kubernetes" in content
