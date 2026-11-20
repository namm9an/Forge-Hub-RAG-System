"""Storage regressions use isolated memory; PostgreSQL semantics have separate tests."""

from datetime import datetime, timedelta, timezone
import asyncio
from pathlib import Path
from types import SimpleNamespace

import asyncpg
import pytest
from fastapi import HTTPException

from app import db, sessions


@pytest.mark.asyncio
async def test_missing_postgres_configuration_never_acknowledges_memory_writes(monkeypatch):
    monkeypatch.setattr(db, "_use_memory_fallback", False)
    monkeypatch.setattr(db, "get_settings", lambda: SimpleNamespace(
        storage_mode="postgres", environment="production", vector_db_url=None, database_url=None,
    ))
    with pytest.raises(RuntimeError, match="PostgreSQL"):
        await db.execute("INSERT INTO sessions(session_id) VALUES($1)", "must-not-persist")
    assert not db._memory_store.sessions
    assert db._use_memory_fallback is False


@pytest.mark.asyncio
async def test_connection_failure_is_not_a_sticky_memory_fallback(monkeypatch):
    monkeypatch.setattr(db, "_use_memory_fallback", False)
    monkeypatch.setattr(db, "get_settings", lambda: SimpleNamespace(
        storage_mode="postgres", environment="production", vector_db_url=None,
        database_url="postgresql://synthetic@127.0.0.1:1/forge_hub_test",
    ))

    async def unavailable(**kwargs):
        raise OSError("synthetic database outage")

    monkeypatch.setattr(db.asyncpg, "create_pool", unavailable)
    with pytest.raises(RuntimeError, match="PostgreSQL"):
        await db.get_pool()
    assert db._use_memory_fallback is False


@pytest.mark.asyncio
async def test_production_explicit_memory_mode_is_rejected(monkeypatch):
    monkeypatch.setattr(db, "_use_memory_fallback", False)
    monkeypatch.setattr(db, "get_settings", lambda: SimpleNamespace(
        storage_mode="memory", environment="production", vector_db_url=None, database_url=None,
    ))
    with pytest.raises(RuntimeError, match="memory"):
        await db.get_pool()


@pytest.mark.asyncio
async def test_memory_transaction_rolls_back_all_dependent_rows():
    with pytest.raises(ValueError, match="synthetic interrupted ingestion"):
        async with db.transaction():
            await db.execute("INSERT INTO sessions(session_id, user_id) VALUES($1, $2)", "rollback-session", "owner")
            doc = await db.fetchval(
                "INSERT INTO documents(session_id, filename, file_size, total_pages) VALUES($1,$2,$3,$4) RETURNING id",
                "rollback-session", "synthetic.pdf", 10, 1,
            )
            await db.fetchval(
                "INSERT INTO chunks(document_id,chunk_index,content,content_hash,page_number,token_count) "
                "VALUES($1,$2,$3,$4,$5,$6) RETURNING id", doc, 0, "synthetic", "synthetic-hash", 1, 1,
            )
            raise ValueError("synthetic interrupted ingestion")
    assert not db._memory_store.sessions
    assert not db._memory_store.documents
    assert not db._memory_store.chunks


@pytest.mark.asyncio
async def test_nested_memory_rollback_preserves_outer_work():
    async with db.transaction():
        await db.execute("INSERT INTO sessions(session_id,user_id) VALUES($1,$2)", "outer-session", "owner")
        with pytest.raises(ValueError):
            async with db.transaction():
                await db.execute("INSERT INTO sessions(session_id,user_id) VALUES($1,$2)", "inner-session", "owner")
                raise ValueError("rollback inner savepoint")
        assert "outer-session" in db._memory_store.sessions
        assert "inner-session" not in db._memory_store.sessions
    assert "outer-session" in db._memory_store.sessions


@pytest.mark.asyncio
async def test_session_claim_is_atomic_and_owner_cannot_change():
    record_id = await sessions.authorize_session("private-session", "owner-a", create=True)
    assert record_id == db._memory_store.sessions["private-session"]["id"]
    with pytest.raises(HTTPException) as error:
        await sessions.authorize_session("private-session", "owner-b", create=True)
    assert error.value.status_code == 404
    assert db._memory_store.sessions["private-session"]["user_id"] == "owner-a"


@pytest.mark.asyncio
async def test_legacy_ownerless_session_cannot_be_claimed():
    await db.execute("INSERT INTO sessions(session_id) VALUES($1)", "legacy-session")
    with pytest.raises(HTTPException) as error:
        await sessions.authorize_session("legacy-session", "new-owner", create=True)
    assert error.value.status_code == 404
    assert db._memory_store.sessions["legacy-session"].get("user_id") is None


@pytest.mark.asyncio
async def test_expired_session_cannot_be_revived():
    await sessions.authorize_session("expired-session", "owner", create=True)
    db._memory_store.sessions["expired-session"]["last_activity"] = datetime.now(timezone.utc) - timedelta(days=1)
    with pytest.raises(HTTPException) as error:
        await sessions.authorize_session("expired-session", "owner", create=True)
    assert error.value.status_code == 404


@pytest.mark.asyncio
@pytest.mark.parametrize("session_id", ["", "default", "anonymous", "x" * 129, "bad/session"])
async def test_session_id_is_explicit_and_bounded(session_id):
    with pytest.raises(HTTPException) as error:
        await sessions.authorize_session(session_id, "owner", create=True)
    assert error.value.status_code == 422


@pytest.mark.asyncio
async def test_memory_rejects_orphan_history_like_postgres():
    with pytest.raises(asyncpg.ForeignKeyViolationError):
        await db.execute(
            "INSERT INTO chat_history(session_id,user_message,assistant_message,source_documents,processing_time_ms) "
            "VALUES($1,$2,$3,$4,$5)", "missing-session", "q", "a", "[]", 1,
        )


@pytest.mark.asyncio
async def test_internal_expiry_maintenance_removes_expired_records_for_all_users():
    for sid, owner in [("expired-a", "a"), ("expired-b", "b"), ("active-a", "a")]:
        await sessions.authorize_session(sid, owner, create=True)
    for sid in ("expired-a", "expired-b"):
        db._memory_store.sessions[sid]["last_activity"] = datetime.now(timezone.utc) - timedelta(days=1)
        await db.fetchval(
            "INSERT INTO documents(session_id,filename,file_size,total_pages) VALUES($1,$2,$3,$4) RETURNING id",
            sid, "expired.pdf", 10, 1,
        )
    assert await db.cleanup_expired_sessions() == 2
    assert set(db._memory_store.sessions) == {"active-a"}
    assert not db._memory_store.documents


@pytest.mark.asyncio
@pytest.mark.parametrize("invocation", [1, 2])
async def test_concurrent_memory_transactions_work_across_independent_test_loops(invocation):
    async def write(sid):
        async with db.transaction():
            await db.execute("INSERT INTO sessions(session_id,user_id) VALUES($1,$2)", sid, "owner")
            await asyncio.sleep(0.001)
    await asyncio.gather(write("first"), write("second"))
    assert set(db._memory_store.sessions) == {"first", "second"}


def test_runtime_and_bootstrap_schema_are_the_same():
    bootstrap = Path(__file__).resolve().parents[3] / "scripts" / "bootstrap.sql"
    assert db.SCHEMA_SQL.strip() == bootstrap.read_text(encoding="utf-8").strip()
