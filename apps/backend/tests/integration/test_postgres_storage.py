"""Real ownership, migration and rollback regressions on disposable PostgreSQL."""

from contextlib import asynccontextmanager
import json
import uuid

import asyncpg
from fastapi import HTTPException
import pytest

from app import db, sessions
from app.auth import Principal

pytestmark = [pytest.mark.postgres, pytest.mark.asyncio]


class ExistingConnectionPool:
    """The fixture owns the real connection and outer rollback-only transaction."""
    def __init__(self, connection):
        self.connection = connection

    @asynccontextmanager
    async def acquire(self):
        yield self.connection


@pytest.fixture
def application_database(monkeypatch, postgres_connection):
    monkeypatch.setattr(db, "_pool", ExistingConnectionPool(postgres_connection))
    monkeypatch.setattr(db, "_use_memory_fallback", False)
    return postgres_connection


async def test_application_transaction_rolls_back_dependent_writes(application_database):
    sid = "transaction-" + uuid.uuid4().hex
    with pytest.raises(ValueError, match="synthetic interruption"):
        async with db.transaction():
            await db.execute("INSERT INTO sessions(session_id,user_id) VALUES($1,$2)", sid, "owner")
            await db.fetchval(
                "INSERT INTO documents(session_id,filename,file_size,total_pages) VALUES($1,$2,$3,$4) RETURNING id",
                sid, "synthetic.pdf", 100, 1,
            )
            raise ValueError("synthetic interruption")
    assert await application_database.fetchval("SELECT COUNT(*) FROM sessions WHERE session_id=$1", sid) == 0
    assert await application_database.fetchval("SELECT COUNT(*) FROM documents") == 0


async def test_real_session_owner_is_immutable_and_legacy_is_quarantined(application_database):
    sid = "owned-" + uuid.uuid4().hex
    await sessions.authorize_session(sid, "owner-a", create=True)
    with pytest.raises(HTTPException) as error:
        await sessions.authorize_session(sid, "owner-b", create=True)
    assert error.value.status_code == 404
    assert await application_database.fetchval("SELECT user_id FROM sessions WHERE session_id=$1", sid) == "owner-a"
    await application_database.execute("INSERT INTO sessions(session_id) VALUES($1)", "legacy")
    with pytest.raises(HTTPException) as error:
        await sessions.authorize_session("legacy", "owner-a", create=True)
    assert error.value.status_code == 404
    assert await application_database.fetchval("SELECT user_id FROM sessions WHERE session_id='legacy'") is None


async def test_cleanup_deletes_only_callers_expired_sessions(application_database):
    for sid, owner in [("a-expired", "a"), ("b-expired", "b"), ("a-live", "a")]:
        await sessions.authorize_session(sid, owner, create=True)
    await application_database.execute(
        "UPDATE sessions SET last_activity=NOW()-interval '2 hours' WHERE session_id IN ('a-expired','b-expired')"
    )
    assert await sessions.cleanup(user=Principal("a")) == {"deleted": 1}
    rows = await application_database.fetch("SELECT session_id FROM sessions ORDER BY session_id")
    assert [row["session_id"] for row in rows] == ["a-live", "b-expired"]
    with pytest.raises(HTTPException) as error:
        await sessions.authorize_session("b-expired", "b", create=True)
    assert error.value.status_code == 404


async def test_owner_stats_excludes_other_users_and_expired_sessions(application_database):
    for sid, owner in [("a-live", "a"), ("b-live", "b"), ("a-expired", "a")]:
        await sessions.authorize_session(sid, owner, create=True)
        await db.fetchval(
            "INSERT INTO documents(session_id,filename,file_size,total_pages) VALUES($1,$2,$3,$4) RETURNING id",
            sid, "synthetic.pdf", 100, 1,
        )
    await application_database.execute(
        "UPDATE sessions SET last_activity=NOW()-interval '2 hours' WHERE session_id='a-expired'"
    )
    counts = await db.owner_stats("a")
    assert counts == {"total_documents": 1, "total_chunks": 0, "total_embeddings": 0,
                      "total_sessions": 1, "total_messages": 0}


async def test_versioned_upgrade_preserves_legacy_data_and_deduplicates_attribution(application_database):
    connection = application_database
    # Reproduce the prior deployed schema instead of testing only fresh-table DDL.
    await connection.execute("ALTER TABLE sessions DROP COLUMN user_id")
    await connection.execute("ALTER TABLE chunks DROP CONSTRAINT chunks_document_hash_unique")
    await connection.execute("ALTER TABLE chunks DROP COLUMN page_numbers")
    await connection.execute("ALTER TABLE chunks DROP COLUMN content_tsv")
    await connection.execute("ALTER TABLE chat_history DROP COLUMN retrieval_ms, DROP COLUMN generation_ms")
    await connection.execute("INSERT INTO sessions(session_id) VALUES('legacy')")
    doc = await connection.fetchval(
        "INSERT INTO documents(session_id,filename,file_size) VALUES('legacy','legacy.pdf',100) RETURNING id"
    )
    chunks = []
    for index in (0, 1):
        chunks.append(await connection.fetchval(
            "INSERT INTO chunks(document_id,chunk_index,content,content_hash,page_number) "
            "VALUES($1,$2,'same legacy content','same-hash',$3) RETURNING id", doc, index, index + 1,
        ))
    vector = "[1," + ",".join(["0"] * 767) + "]"
    await connection.execute("INSERT INTO embeddings(chunk_id,embedding) VALUES($1,$2::vector)", chunks[1], vector)
    citation = json.dumps([{"id": str(chunks[1]), "content": "same legacy content"}])
    await connection.execute(
        "INSERT INTO chat_history(session_id,user_message,assistant_message,source_documents,processing_time_ms) "
        "VALUES('legacy','q','a',$1::jsonb,1)", citation,
    )
    await db.ensure_schema()
    await db.ensure_schema()  # The same versions are idempotent, not data-rewriting every startup.
    assert await connection.fetchval("SELECT COUNT(*) FROM schema_migrations") == 4
    assert await connection.fetchval("SELECT user_id FROM sessions WHERE session_id='legacy'") is None
    rows = await connection.fetch("SELECT id,page_numbers FROM chunks")
    assert len(rows) == 1
    assert rows[0]["id"] == chunks[0]
    assert rows[0]["page_numbers"] == [1, 2]
    assert await connection.fetchval("SELECT chunk_id FROM embeddings") == chunks[0]
    saved = await connection.fetchval("SELECT source_documents FROM chat_history")
    assert json.loads(saved) == json.loads(citation)
    with pytest.raises(asyncpg.UniqueViolationError):
        async with connection.transaction():
            await connection.execute(
                "INSERT INTO chunks(document_id,chunk_index,content,content_hash) VALUES($1,2,'same legacy content','same-hash')", doc,
            )
    assert await connection.fetchval("SELECT COUNT(*) FROM documents") == 1
