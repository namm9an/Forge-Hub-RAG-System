"""Offline contract tests by default; PostgreSQL requires an explicit disposable DSN."""

from contextvars import ContextVar
import os
import re
import socket
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

import asyncpg
import httpx
import pytest
import pytest_asyncio
from pydantic_settings import BaseSettings
from pydantic_settings.sources import DotEnvSettingsSource

# Ensure backend package is in python search path regardless of where pytest is invoked
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

_session_patches = pytest.MonkeyPatch()
_real_connect = asyncpg.connect
_postgres_port = ContextVar("explicit_postgres_test_port", default=None)
_protocol_port = ContextVar("explicit_protocol_test_port", default=None)


def _test_database_url():
    dsn = os.environ.get("FORGE_HUB_TEST_DATABASE_URL", "")
    try:
        url = urlsplit(dsn)
        safe = (
            url.scheme in {"postgres", "postgresql"}
            and url.hostname in {"localhost", "127.0.0.1", "::1"}
            and re.fullmatch(r"forge_hub_test(?:_[a-zA-Z0-9_]+)?", unquote(url.path.lstrip("/")))
            and not url.query and not url.fragment
        )
        port = url.port or 5432
    except ValueError:
        safe = False
    if not safe:
        raise pytest.UsageError(
            "--run-postgres requires FORGE_HUB_TEST_DATABASE_URL pointing to an explicit "
            "loopback disposable database named forge_hub_test or forge_hub_test_<suffix>, "
            "without URL query parameters. Application DSNs are never used."
        )
    return dsn, port


def _guard_socket(original):
    def guarded(sock, address):
        caller = sys._getframe(1)
        # Windows asyncio uses the stdlib's own loopback socket pair for its control pipe.
        control_pair = caller.f_code.co_name in {"socketpair", "_fallback_socketpair"} and caller.f_code.co_filename == socket.__file__
        loopback = isinstance(address, tuple) and address[0] in {"localhost", "127.0.0.1", "::1"}
        explicit_postgres = loopback and address[1] == _postgres_port.get()
        explicit_protocol = loopback and address[1] == _protocol_port.get()
        if (control_pair and loopback) or explicit_postgres or explicit_protocol:
            return original(sock, address)
        raise RuntimeError("Offline tests block outbound sockets")
    return guarded


def _deny_http(*args, **kwargs):
    raise RuntimeError("Offline tests allow ASGI/mock transports, not external HTTP")


async def _deny_database(*args, **kwargs):
    raise RuntimeError("Ordinary tests cannot connect to a real database; use the guarded PostgreSQL fixture")


def pytest_addoption(parser):
    parser.addoption("--run-postgres", action="store_true", help="Run real PostgreSQL tests against a disposable test DSN")
    parser.addoption("--run-protocol", action="store_true", help="Run bounded raw HTTP checks on a new loopback-only server")


def pytest_configure(config):
    if config.getoption("--run-postgres"):
        _test_database_url()  # Explicit invalid requests fail even before test selection.

    def isolated_sources(cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings):
        return (init_settings,)

    # App imports instantiate settings during collection, before autouse fixtures run.
    # Dotenv sources may also read files while being constructed, before source selection.
    _session_patches.setattr(BaseSettings, "settings_customise_sources", classmethod(isolated_sources))
    _session_patches.setattr(DotEnvSettingsSource, "_read_env_files", lambda self: {})
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        _session_patches.setenv(key, "1")
    _session_patches.setenv("TOKENIZERS_PARALLELISM", "false")
    _session_patches.setattr(sys, "dont_write_bytecode", True)
    _session_patches.setattr(socket.socket, "connect", _guard_socket(socket.socket.connect))
    _session_patches.setattr(socket.socket, "connect_ex", _guard_socket(socket.socket.connect_ex))
    # HTTP guards fail before proxy/DNS handling; ASGI and explicit mock transports still work.
    _session_patches.setattr(httpx.HTTPTransport, "handle_request", _deny_http)
    _session_patches.setattr(httpx.AsyncHTTPTransport, "handle_async_request", _deny_http)
    _session_patches.setattr(asyncpg, "connect", _deny_database)
    _session_patches.setattr(asyncpg, "create_pool", _deny_database)


def pytest_unconfigure(config):
    _session_patches.undo()


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-postgres"):
        skip = pytest.mark.skip(reason="Real PostgreSQL is opt-in: --run-postgres and FORGE_HUB_TEST_DATABASE_URL")
        for item in items:
            if item.get_closest_marker("postgres"):
                item.add_marker(skip)
    if not config.getoption("--run-protocol"):
        for item in items:
            if item.get_closest_marker("protocol"):
                item.add_marker(pytest.mark.skip(reason="Raw loopback HTTP is opt-in: --run-protocol"))


@pytest.fixture
def loopback_protocol_port(request):
    if not request.config.getoption("--run-protocol"):
        pytest.skip("Raw loopback HTTP requires --run-protocol")
    return _protocol_port


class ContractEncoder:
    """Fixed normalized vectors test shapes/control flow only, never semantic retrieval quality."""

    max_seq_length = 384

    class Tokenizer:
        model_max_length = 384

        def encode(self, text, add_special_tokens=True, **kwargs):
            return list(range(len(re.findall(r"\w+|[^\w\s]", text)) + (2 if add_special_tokens else 0)))

        def num_special_tokens_to_add(self, pair=False):
            return 2

        def __call__(self, text, **kwargs):
            spans = [match.span() for match in re.finditer(r"\w+|[^\w\s]", text)]
            return {"input_ids": list(range(len(spans))), "offset_mapping": spans}

    tokenizer = Tokenizer()

    def encode(self, texts, **kwargs):
        import numpy as np

        if isinstance(texts, str):
            texts = [texts]
        vectors = np.zeros((len(texts), 768), dtype=np.float32)
        vectors[:, 0] = 1.0
        return vectors


@pytest.fixture(autouse=True)
def isolated_backend(monkeypatch, request):
    from app import app, db, embeddings, gating
    from app.config import get_settings

    get_settings.cache_clear()
    db._memory_store.reset()
    app.dependency_overrides.clear()
    # Original API contract tests predate authentication; security tests use real tokens or explicit local overrides.
    if request.path.name in {"test_gating.py", "test_retrieval.py", "test_upgrades.py"}:
        from app.auth import Principal, require_user
        app.dependency_overrides[require_user] = lambda: Principal(user_id="contract-user")
    monkeypatch.setattr(db, "_pool", None)
    monkeypatch.setattr(db, "_use_memory_fallback", True)
    monkeypatch.setattr(embeddings, "_model", ContractEncoder())
    monkeypatch.setattr(gating, "_global_reranker", None)
    monkeypatch.setattr(gating.CrossEncoderReranker, "_get_ranker", lambda self: None)
    yield
    db._memory_store.reset()
    app.dependency_overrides.clear()
    get_settings.cache_clear()


@pytest_asyncio.fixture(params=["runtime", "bootstrap"])
async def postgres_connection(request):
    """Independent real connection and rollback-only schema; no application pool/fallback."""
    import uuid
    from app.db import SCHEMA_SQL

    if not request.config.getoption("--run-postgres"):
        pytest.skip("Real PostgreSQL requires --run-postgres")
    dsn, port = _test_database_url()
    token = _postgres_port.set(port)
    try:
        connection = await _real_connect(dsn, timeout=10)
    finally:
        _postgres_port.reset(token)
    transaction = connection.transaction()
    try:
        await transaction.start()
        # Preserve pre-existing extension placement. Newly installed extensions also roll back.
        for extension in ("pgcrypto", "vector"):
            await connection.execute(f'CREATE EXTENSION IF NOT EXISTS "{extension}" WITH SCHEMA public')
        extension_schemas = await connection.fetch(
            "SELECT n.nspname FROM pg_extension e JOIN pg_namespace n ON n.oid = e.extnamespace "
            "WHERE e.extname IN ('pgcrypto', 'vector')"
        )
        schema = "forge_hub_test_" + uuid.uuid4().hex
        await connection.execute(f'CREATE SCHEMA "{schema}"')
        search_path = dict.fromkeys([schema, *(row["nspname"] for row in extension_schemas), "public"])
        quoted = ", ".join('"' + name.replace('"', '""') + '"' for name in search_path)
        await connection.execute("SET LOCAL search_path TO " + quoted)
        ddl = SCHEMA_SQL if request.param == "runtime" else (backend_dir.parents[1] / "scripts/bootstrap.sql").read_text(encoding="utf-8")
        await connection.execute(ddl)
        yield connection
    finally:
        try:
            if connection.is_in_transaction():
                await transaction.rollback()
        finally:
            await connection.close()
