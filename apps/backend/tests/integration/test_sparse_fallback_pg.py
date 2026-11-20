"""Real English FTS fallback semantics, isolated by rollback for both DDL sources."""

import uuid

import pytest

from app import retrieval

pytestmark = [pytest.mark.postgres, pytest.mark.asyncio]


@pytest.fixture
def sparse_database(monkeypatch, postgres_connection):
    # Execute the application's exact SQL on the fixture's real rollback-owned connection.
    monkeypatch.setattr(retrieval, "fetch", postgres_connection.fetch)
    return postgres_connection


async def seed(connection, sid="owned", owner="alice", content="Durable logging records PostgreSQL WAL checkpoints."):
    await connection.execute("INSERT INTO sessions(session_id,user_id) VALUES($1,$2)", sid, owner)
    document = await connection.fetchval(
        "INSERT INTO documents(session_id,filename,file_size) VALUES($1,'synthetic.pdf',1) RETURNING id", sid,
    )
    return await connection.fetchval(
        "INSERT INTO chunks(document_id,chunk_index,content,content_hash) VALUES($1,0,$2,$3) RETURNING id",
        document, content, uuid.uuid4().hex,
    )


async def test_unmatched_question_word_recovers_owned_lexical_evidence(sparse_database):
    chunk = await seed(sparse_database)
    text = "How does PostgreSQL logging preserve durability against unmatchedword?"
    assert await sparse_database.fetchval(
        "SELECT COUNT(*) FROM chunks WHERE content_tsv @@ websearch_to_tsquery('english',$1)", text,
    ) == 0
    rows = await retrieval._sparse_search(text, "owned", 5, user_id="alice")
    assert [row["chunk_id"] for row in rows] == [chunk]
    assert rows[0]["sparse_score"] > 0


async def test_retry_keeps_both_user_and_session_scope(sparse_database):
    first = await seed(sparse_database)
    second = await seed(sparse_database, "alice-other", "alice")
    foreign = await seed(sparse_database, "bob-session", "bob")
    text = "logging unmatchedword"
    assert [r["chunk_id"] for r in await retrieval._sparse_search(text, "owned", 5, user_id="alice")] == [first]
    assert [r["chunk_id"] for r in await retrieval._sparse_search(text, "alice-other", 5, user_id="alice")] == [second]
    assert [r["chunk_id"] for r in await retrieval._sparse_search(text, "bob-session", 5, user_id="bob")] == [foreign]
    assert await retrieval._sparse_search(text, "owned", 5, user_id="bob") == []
    assert await retrieval._sparse_search(text, "bob-session", 5, user_id="alice") == []


@pytest.mark.parametrize("query", ['"durable impossible"', "logging -durable", "logging unknown OR missing"])
async def test_explicit_phrase_negation_and_or_are_not_relaxed(sparse_database, query):
    await seed(sparse_database)
    assert await retrieval._sparse_search(query, "owned", 5, user_id="alice") == []


@pytest.mark.parametrize("query", ["the and of to", "... !!!", "' "])
async def test_no_meaningful_english_lexemes_returns_empty_without_sql_error(sparse_database, query):
    await seed(sparse_database)
    assert await retrieval._sparse_search(query, "owned", 5, user_id="alice") == []


@pytest.mark.parametrize("query", [
    "How does PostgreSQL's WAL/checkpoint survive unmatchedword?",
    "PostgreSQL's WAL/checkpoint foo'); DROP TABLE chunks; /* unmatchedword",
])
async def test_apostrophes_and_technical_punctuation_remain_bound_data(sparse_database, query):
    chunk = await seed(sparse_database)
    rows = await retrieval._sparse_search(query, "owned", 5, user_id="alice")
    assert [row["chunk_id"] for row in rows] == [chunk]
    assert await sparse_database.fetchval("SELECT COUNT(*) FROM chunks") == 1


async def test_existing_native_phrase_match_still_works(sparse_database):
    chunk = await seed(sparse_database)
    rows = await retrieval._sparse_search('"durable logging"', "owned", 5, user_id="alice")
    assert [row["chunk_id"] for row in rows] == [chunk]
