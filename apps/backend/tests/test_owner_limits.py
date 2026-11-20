import asyncio
from datetime import timedelta
import pytest
from fastapi import HTTPException

from app import db, sessions
from app.config import Settings


@pytest.mark.asyncio
async def test_session_cap_is_atomic_and_other_owner_has_own_budget(monkeypatch):
    monkeypatch.setattr(db, "get_settings", lambda: Settings(max_sessions_per_user=2))
    outcomes = await asyncio.gather(*[
        sessions.authorize_session("session-" + str(i), "owner-a", create=True)
        for i in range(3)
    ], return_exceptions=True)
    assert sum(isinstance(outcome, HTTPException) and outcome.status_code == 429 for outcome in outcomes) == 1
    assert len(db._memory_store.sessions) == 2
    await sessions.authorize_session("another-owner", "owner-b", create=True)


@pytest.mark.asyncio
async def test_rate_budget_is_shared_between_requests_and_separate_between_owners(monkeypatch):
    monkeypatch.setattr(db, "get_settings", lambda: Settings(requests_per_minute=2))
    allowed = await asyncio.gather(*[db.consume_rate_limit("a") for _ in range(3)])
    assert allowed.count(True) == 2
    assert allowed.count(False) == 1
    assert await db.consume_rate_limit("b")
    db._memory_store.rate_limits["a"]["window_start"] -= timedelta(minutes=1)
    assert await db.consume_rate_limit("a")
