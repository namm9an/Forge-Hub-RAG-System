"""Real RSA verification + HTTP contracts + PostgreSQL, with deterministic model vectors."""
import asyncio
from contextlib import asynccontextmanager
import time

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from app import app, db
from app.config import Settings, get_settings

pytestmark = [pytest.mark.postgres, pytest.mark.asyncio]


@pytest.fixture
def authenticated_database(monkeypatch, postgres_connection):
    class RollbackPool:
        # One rollback-owned connection serializes SQL in this fixture, never a load-test stand-in.
        def __init__(self):
            self.lock = asyncio.Lock()

        @asynccontextmanager
        async def acquire(self):
            async with self.lock:
                yield postgres_connection

    monkeypatch.setattr(db, "_pool", RollbackPool())
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public = key.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo).decode()
    settings = Settings(auth_public_key=public, auth_issuer="forge-test-issuer", auth_audience="forge-test-api")
    app.dependency_overrides[get_settings] = lambda: settings
    def credentials(subject):
        now = int(time.time())
        token = jwt.encode(dict(sub=subject, iss=settings.auth_issuer, aud=settings.auth_audience,
                                iat=now, exp=now+300), key, algorithm="RS256")
        return {"Authorization": "Bearer " + token}
    return credentials, postgres_connection


async def test_two_signed_users_cannot_access_each_others_data(authenticated_database):
    from scripts.benchmark_rag import generate_benchmark_pdf
    credentials, connection = authenticated_database
    a, b = credentials("alice"), credentials("bob")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/documents/upload", headers=a, params={"session_id": "alice-session"},
            files={"files": ("evidence.pdf", generate_benchmark_pdf(["Write-ahead logging records durable changes before data pages are written."]), "application/pdf")})
        assert response.status_code == 200, response.text
        document_id = response.json()["documents"][0]["document_id"]
        assert (await client.get("/api/documents", headers=a, params={"session_id": "alice-session"})).json()["documents"]
        for method, path, options in [
            ("GET", "/api/documents", {"params": {"session_id": "alice-session"}}),
            ("DELETE", "/api/documents/"+document_id, {}),
            ("GET", "/api/sessions/history", {"params": {"session_id": "alice-session"}}),
            ("POST", "/api/search/hybrid", {"json": {"query": "logging", "session_id": "alice-session"}}),
            ("POST", "/api/chat", {"json": {"query": "logging", "session_id": "alice-session"}}),
            ("POST", "/api/embeddings/generate", {"params": {"session_id": "alice-session"}}),
            ("DELETE", "/api/sessions/alice-session", {}),
        ]:
            denied = await client.request(method, path, headers=b, **options)
            assert denied.status_code == 404, (path, denied.text)
            if method == "DELETE" and path.startswith("/api/documents/"):
                missing = await client.delete("/api/documents/00000000-0000-4000-8000-000000000000", headers=b)
                assert denied.json() == missing.json()
        assert (await client.get("/api/stats", headers=b)).json()["total_documents"] == 0
        assert (await client.get("/api/documents", headers=a)).status_code == 422
        assert (await client.get("/api/documents", headers=a, params={"session_id": "default"})).status_code == 422
        own_search = await client.post("/api/search/hybrid", headers=a, json={"query": "logging", "session_id": "alice-session"})
        assert own_search.status_code == 200, own_search.text
        assert own_search.json()["results"][0]["document_id"] == document_id
        await connection.execute("UPDATE sessions SET last_activity=NOW()-interval '2 hours' WHERE session_id='alice-session'")
        expired = await client.get("/api/documents", headers=a, params={"session_id": "alice-session"})
        assert expired.status_code == 404
        assert await db.cleanup_expired_sessions() == 1
        assert await connection.fetchval("SELECT COUNT(*) FROM documents") == 0
        assert await connection.fetchval("SELECT COUNT(*) FROM chunks") == 0
        assert await connection.fetchval("SELECT COUNT(*) FROM embeddings") == 0


async def test_fresh_chat_refusal_has_real_fk_parent_and_private_history(authenticated_database):
    credentials, connection = authenticated_database
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/chat", headers=credentials("a"), json={"query": "No evidence yet", "session_id": "fresh-chat"})
        assert response.status_code == 200, response.text
        assert response.json()["refusal_triggered"] is True
        assert await connection.fetchval("SELECT COUNT(*) FROM chat_history WHERE session_id='fresh-chat'") == 1
        history = await client.get("/api/sessions/history", headers=credentials("a"), params={"session_id": "fresh-chat"})
        assert len(history.json()["messages"]) == 1
        assert (await client.get("/api/sessions/history", headers=credentials("b"), params={"session_id": "fresh-chat"})).status_code == 404


async def test_history_routes_preserve_messages_with_old_source_metadata(authenticated_database):
    credentials, connection = authenticated_database
    owner = credentials("history-owner")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test",
    ) as client:
        response = await client.post("/api/sessions", headers=owner)
        assert response.status_code == 200
        session_id = response.json()["session_id"]
        await connection.execute(
            """
            INSERT INTO chat_history(id, session_id, user_message, assistant_message, source_documents, processing_time_ms)
            VALUES ('00000000-0000-4000-8000-000000000001', $1, 'Older question', 'Older answer',
                    '[{"id":"legacy-valid"}]'::jsonb, 1),
                   ('00000000-0000-4000-8000-000000000002', $1, 'Newer question', 'Newer answer',
                    '[{"chunk_id":"legacy-incomplete"}]'::jsonb, 1)
            """,
            session_id,
        )
        params = {"session_id": session_id}
        responses = [
            await client.get(path, headers=owner, params=params)
            for path in ("/api/chat/history", "/api/sessions/history")
        ]
        assert [response.status_code for response in responses] == [200, 200]
        same_history, session_history = [response.json()["messages"] for response in responses]
        assert same_history == session_history
        assert [turn["assistant_message"] for turn in same_history] == ["Newer answer", "Older answer"]
        assert same_history[0]["source_documents"] == []
        assert same_history[1]["source_documents"][0]["id"] == "legacy-valid"


async def test_real_rate_bucket_and_owner_session_quota(authenticated_database, monkeypatch):
    from app.sessions import authorize_session
    _, connection = authenticated_database
    monkeypatch.setattr(db, "get_settings", lambda: Settings(requests_per_minute=2, max_sessions_per_user=1))
    assert await db.consume_rate_limit("owner")
    assert await db.consume_rate_limit("owner")
    assert not await db.consume_rate_limit("owner")
    assert await db.consume_rate_limit("other")
    await connection.execute("UPDATE auth_rate_limits SET window_start=window_start-interval '1 minute' WHERE user_id='owner'")
    assert await db.consume_rate_limit("owner")
    await authorize_session("one", "owner", create=True)
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as error:
        await authorize_session("two", "owner", create=True)
    assert error.value.status_code == 429
    await authorize_session("three", "other", create=True)


async def test_listing_cannot_read_replacement_owners_documents(authenticated_database, monkeypatch):
    """Force the delete/recreate interleaving between authorization and the document SELECT."""
    from app import documents, sessions
    from app.auth import verify_token
    credentials, connection = authenticated_database
    alice, bob = credentials("alice"), credentials("bob")
    settings = app.dependency_overrides[get_settings]()
    bob_id = verify_token(bob["Authorization"].split(" ", 1)[1], settings).user_id
    authorize = documents.authorize_session
    replaced = []

    async def authorize_then_replace(sid, user_id):
        await authorize(sid, user_id)
        await connection.execute("DELETE FROM sessions WHERE session_id=$1", sid)
        await sessions.authorize_session(sid, bob_id, create=True)
        replaced.append(await connection.fetchval(
            "INSERT INTO documents(session_id,filename,file_size,total_pages) VALUES($1,$2,1,1) RETURNING id",
            sid, "bob-private.pdf",
        ))
        return sid

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        session = await client.post("/api/sessions", headers=alice)
        assert session.status_code == 200
        sid = session.json()["session_id"]
        monkeypatch.setattr(documents, "authorize_session", authorize_then_replace)
        response = await client.get("/api/documents", headers=alice, params={"session_id": sid})
    assert len(replaced) == 1
    assert response.status_code == 200
    assert response.json()["documents"] == []
    assert "bob-private" not in response.text


async def test_stale_document_lookup_cannot_delete_twice(authenticated_database, monkeypatch):
    from app import documents
    credentials, connection = authenticated_database
    headers = credentials("owner")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        session = await client.post("/api/sessions", headers=headers)
        sid = session.json()["session_id"]
        ids = [await connection.fetchval(
            "INSERT INTO documents(session_id,filename,file_size,total_pages) VALUES($1,$2,1,1) RETURNING id",
            sid, name,
        ) for name in ("delete.pdf", "keep.pdf")]
        await connection.execute("UPDATE sessions SET total_documents=2 WHERE session_id=$1", sid)
        assert (await client.delete(f"/api/documents/{ids[0]}", headers=headers)).status_code == 200

        # The duplicate request observed the row before the first deletion committed.
        async def stale_lookup(query, document_id, *args):
            assert document_id == ids[0]
            return [{"id": document_id, "session_id": sid}]

        monkeypatch.setattr(documents, "fetch", stale_lookup)
        assert (await client.delete(f"/api/documents/{ids[0]}", headers=headers)).status_code == 404
        assert await connection.fetchval("SELECT total_documents FROM sessions WHERE session_id=$1", sid) == 1
        assert await connection.fetchval("SELECT id FROM documents WHERE session_id=$1", sid) == ids[1]


@pytest.mark.parametrize("stream", [False, True])
async def test_chat_cannot_write_to_replacement_session(authenticated_database, monkeypatch, stream):
    from app import chat, sessions
    credentials, connection = authenticated_database
    headers = credentials("owner")
    save = chat._save_completed_turn
    replacements = []

    async def replace_before_save(request, principal, *args):
        sid = request.session_id
        old_id = await connection.fetchval("SELECT id FROM sessions WHERE session_id=$1", sid)
        await connection.execute("DELETE FROM sessions WHERE id=$1", old_id)
        await sessions.authorize_session(sid, principal.user_id, create=True)
        new_id = await connection.fetchval("SELECT id FROM sessions WHERE session_id=$1", sid)
        assert new_id != old_id
        replacements.append(new_id)
        return await save(request, principal, *args)

    monkeypatch.setattr(chat, "_save_completed_turn", replace_before_save)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        path = "/api/chat/stream" if stream else "/api/chat"
        response = await client.post(path, headers={**headers, "Accept": "text/event-stream"},
                                     json={"query": "How long are logs retained?", "session_id": "replaced"})
        if stream:
            assert response.status_code == 200
            assert "event: error" in response.text
            assert "event: complete" not in response.text
        else:
            assert response.status_code == 404
        assert len(replacements) == 1
        assert await connection.fetchval("SELECT COUNT(*) FROM chat_history WHERE session_id='replaced'") == 0
