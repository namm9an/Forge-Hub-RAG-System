"""Sparse retry control flow; actual query grammar is covered on PostgreSQL."""

import pytest

from app import retrieval


async def test_empty_natural_language_search_retries_with_bound_owner_scope(monkeypatch):
    calls = []
    expected = [{"chunk_id": "owned", "sparse_score": 0.1}]
    text = "How does PostgreSQL's WAL/checkpoint survive an unmatchedword?"

    async def fetch(statement, *args):
        calls.append((statement, args))
        return [] if len(calls) == 1 else expected

    monkeypatch.setattr(retrieval, "fetch", fetch)
    assert await retrieval._sparse_search(text, "owned-session", 7, user_id="owner") == expected
    assert len(calls) == 2
    for statement, args in calls:
        assert args == (text, "owned-session", 7, "owner")
        assert text not in statement
        assert "s.user_id = $4" in statement and "d.session_id = $2" in statement


async def test_native_sparse_results_do_not_trigger_a_second_query(monkeypatch):
    calls = []
    expected = [{"chunk_id": "native", "sparse_score": 0.4}]

    async def fetch(*args):
        calls.append(args)
        return expected

    monkeypatch.setattr(retrieval, "fetch", fetch)
    assert await retrieval._sparse_search("logging", "s", 5, user_id="owner") == expected
    assert len(calls) == 1


@pytest.mark.parametrize("query", [
    '"durable impossible"', '"unterminated phrase', "logging unknown OR missing",
    "logging unknown or missing", "logging -durable", "logging (-durable)",
])
async def test_explicit_websearch_syntax_never_relaxes_empty_results(monkeypatch, query):
    calls = []

    async def fetch(*args):
        calls.append(args)
        return []

    monkeypatch.setattr(retrieval, "fetch", fetch)
    assert await retrieval._sparse_search(query, "s", 5, user_id="owner") == []
    assert len(calls) == 1


async def test_database_errors_propagate_without_a_retry(monkeypatch):
    calls = []

    async def fetch(*args):
        calls.append(args)
        raise RuntimeError("synthetic database failure")

    monkeypatch.setattr(retrieval, "fetch", fetch)
    with pytest.raises(RuntimeError, match="synthetic database failure"):
        await retrieval._sparse_search("durability unmatchedword", "s", 5, user_id="owner")
    assert len(calls) == 1


async def test_sparse_query_limit_is_enforced_before_database_work(monkeypatch):
    async def fetch(*args):
        pytest.fail("oversized queries must not reach PostgreSQL")

    monkeypatch.setattr(retrieval, "fetch", fetch)
    with pytest.raises(ValueError, match="4000"):
        await retrieval._sparse_search("x" * 4001, "s", 5, user_id="owner")
