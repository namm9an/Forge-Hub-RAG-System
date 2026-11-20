from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from app.config import Settings


@pytest.mark.asyncio
async def test_model_warmup_rejects_heuristic_reranking():
    from app.model_readiness import warmup_models
    with pytest.raises(RuntimeError, match="Neural reranker"):
        await warmup_models(Settings())  # Offline fixture substitutes encoder and disables neural ranker.


@pytest.mark.asyncio
async def test_readiness_requires_completed_warmup_and_shutdown_clears_it(monkeypatch):
    import app.main as main
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public = key.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo).decode()
    cfg = Settings(auth_public_key=public, auth_issuer="test-issuer", auth_audience="test-api")
    monkeypatch.setattr(main, "get_settings", lambda: cfg)
    ranker = SimpleNamespace(backend="flashrank")
    monkeypatch.setattr(main, "get_reranker", lambda name: ranker)
    events = []
    class Connection:
        async def fetchval(self, *args, **kwargs):
            return 1
    class Pool:
        @asynccontextmanager
        async def acquire(self, **kwargs):
            yield Connection()
    async def pool(): return Pool()
    async def schema(): events.append("schema")
    async def warmup(settings): events.append("warmup")
    async def close(): events.append("closed")
    async def cleanup(): return 0
    async def forbidden(): raise AssertionError("Normal startup must not run schema migrations")
    monkeypatch.setattr(main, "get_pool", pool)
    monkeypatch.setattr(main, "check_schema", schema)
    monkeypatch.setattr(main, "warmup_models", warmup)
    monkeypatch.setattr(main, "close_pool", close)
    monkeypatch.setattr(main, "cleanup_expired_sessions", cleanup)
    monkeypatch.setattr(main, "ensure_schema", forbidden)
    application = main.create_app()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application), base_url="http://test") as client:
        assert (await client.get("/ready")).status_code == 503
        async with application.router.lifespan_context(application):
            assert (await client.get("/ready")).status_code == 200
            assert events[:2] == ["schema", "warmup"]
            ranker.backend = "heuristic"
            assert (await client.get("/ready")).status_code == 503
            ranker.backend = "flashrank"
        assert (await client.get("/ready")).status_code == 503
    assert events[-1] == "closed"


def test_production_rejects_migrations_and_wildcard_hosts_or_origins():
    from pydantic import ValidationError
    for values in ({"run_migrations": True}, {"allowed_hosts": "*"},
                   {"allowed_origins": "https://frontend.example.test, *"}):
        with pytest.raises(ValidationError):
            Settings(environment="production", **values)
