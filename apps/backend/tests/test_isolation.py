"""Harness guarantees; deterministic substitutes verify contracts, not RAG quality."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest


def test_settings_ignore_ambient_service_configuration(monkeypatch):
    from app.config import Settings

    for name in (
        "DATABASE_URL", "VECTOR_DB_URL", "OPENROUTER_API_KEY",
        "EMBEDDING_API_URL", "EMBEDDING_API_KEY",
    ):
        monkeypatch.setenv(name, "synthetic-service-setting-must-not-be-used")

    settings = Settings(_env_file=None)
    assert settings.database_url is None
    assert settings.vector_db_url is None
    assert settings.openrouter_api_key is None
    assert settings.embedding_api_url is None
    assert settings.embedding_api_key is None


def test_settings_ignore_dotenv_files(tmp_path):
    from app.config import Settings

    sentinel = tmp_path / "synthetic.env"
    sentinel.write_text("DATABASE_URL=synthetic-dotenv-must-not-be-used\n", encoding="utf-8")
    assert Settings(_env_file=sentinel).database_url is None


@pytest.mark.parametrize("invocation", [1, 2])
def test_each_test_starts_with_empty_memory_storage(invocation):
    from app import db

    assert db._pool is None
    assert db._use_memory_fallback is True
    for table in ("sessions", "documents", "chunks", "embeddings", "chat_history"):
        storage = getattr(db._memory_store, table)
        assert not storage, table
        if isinstance(storage, dict):
            storage["synthetic-isolation-marker"] = {"invocation": invocation}
        else:
            storage.append({"invocation": invocation})


@pytest.mark.asyncio
@pytest.mark.parametrize("connector", ["connect", "create_pool"])
async def test_real_database_connectors_are_blocked_before_network(monkeypatch, connector):
    import socket
    import asyncpg

    dns_attempts = []

    def unexpected_dns(*args, **kwargs):
        dns_attempts.append(True)
        raise AssertionError("Ordinary database tests must not reach DNS")

    monkeypatch.setattr(socket, "getaddrinfo", unexpected_dns)
    with pytest.raises(RuntimeError, match="Ordinary tests cannot connect to a real database"):
        await getattr(asyncpg, connector)("postgresql://synthetic:synthetic@127.0.0.1:1/forge_hub_test")
    assert not dns_attempts


@pytest.mark.asyncio
async def test_external_http_is_blocked_before_dns(monkeypatch):
    import socket
    import httpx

    dns_attempts = []

    def unexpected_dns(*args, **kwargs):
        dns_attempts.append(True)
        raise AssertionError("External HTTP must be rejected before DNS")

    monkeypatch.setattr(socket, "getaddrinfo", unexpected_dns)
    async with httpx.AsyncClient(trust_env=False) as client:
        with pytest.raises(RuntimeError, match="Offline tests allow ASGI/mock transports, not external HTTP"):
            await client.get("https://network-guard.invalid/")
    assert not dns_attempts


def test_contract_encoder_is_deterministic_without_loading_a_model(monkeypatch):
    from app import embeddings
    from app.config import get_settings

    def forbidden_model_load(*args, **kwargs):
        raise AssertionError("Offline contract tests must not initialize a real model")

    monkeypatch.setattr(embeddings, "SentenceTransformer", forbidden_model_load)
    model = embeddings.get_model(get_settings())
    first = model.encode(["synthetic contract input"], convert_to_numpy=True, normalize_embeddings=True)
    second = model.encode(["synthetic contract input"], convert_to_numpy=True, normalize_embeddings=True)
    assert first.shape == (1, 768)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1.0)


def test_fallback_reranking_does_not_initialize_flashrank(monkeypatch):
    from app.gating import CrossEncoderReranker

    attempts = []

    def forbidden_ranker(*args, **kwargs):
        attempts.append(True)
        raise AssertionError("Offline contract tests must not initialize FlashRank")

    monkeypatch.setitem(sys.modules, "flashrank", SimpleNamespace(Ranker=forbidden_ranker))
    candidates = [{"content": "Synthetic contract evidence.", "dense_score": 0.8, "sparse_score": 0.2}]
    result = CrossEncoderReranker().rerank("contract evidence", candidates)
    assert result[0]["rerank_score"] is not None
    assert not attempts
def test_stdlib_control_socketpair_is_allowed_without_permitting_other_connections():
    import socket
    first, second = socket.socketpair()
    try:
        first.sendall(b"control")
        assert second.recv(7) == b"control"
    finally:
        first.close()
        second.close()
