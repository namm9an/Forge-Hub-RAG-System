"""Least-privilege checks run only on rollback-only disposable PostgreSQL fixtures."""

from contextlib import asynccontextmanager
from pathlib import Path
import uuid

import asyncpg
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient
import pytest
import pytest_asyncio

from app import app, db, sessions
from app.auth import Principal, require_user

pytestmark = [pytest.mark.postgres, pytest.mark.asyncio]


class FixturePool:
    def __init__(self, connection):
        self.connection = connection

    @asynccontextmanager
    async def acquire(self):
        yield self.connection


@pytest_asyncio.fixture
async def runtime_database(postgres_connection, monkeypatch):
    connection = postgres_connection
    monkeypatch.setattr(db, "_pool", FixturePool(connection))
    monkeypatch.setattr(db, "_use_memory_fallback", False)
    await db.ensure_schema()
    await connection.execute("CREATE SEQUENCE operator_counter")
    role = "forge_hub_rt_" + uuid.uuid4().hex
    script = (Path(__file__).resolve().parents[4] / "scripts/runtime_role.sql").read_text(encoding="utf-8")
    assert script.count("'forge_hub_runtime'") == 1
    await connection.execute(script.replace("'forge_hub_runtime'", "'" + role + "'"))
    before = await connection.fetchval("SELECT current_user")
    scope = connection.transaction()
    await scope.start()
    try:
        yield connection, role
    finally:
        # Roll back aborted statements and SET LOCAL ROLE before the outer fixture
        # rolls back the newly created role, grants, schemas and all synthetic data.
        await scope.rollback()
        await connection.execute("RESET ROLE")
        assert await connection.fetchval("SELECT current_user") == before


async def assume_runtime(connection, role):
    await connection.execute('SET LOCAL ROLE "' + role + '"')


async def test_privileged_database_identity_fails_readiness(postgres_connection, monkeypatch):
    monkeypatch.setattr(db, "_pool", FixturePool(postgres_connection))
    monkeypatch.setattr(db, "_use_memory_fallback", False)
    with pytest.raises(RuntimeError, match="least-privilege"):
        await db.check_runtime_privileges()


async def test_restricted_role_can_serve_private_apis_without_ddl(runtime_database):
    connection, role = runtime_database
    await assume_runtime(connection, role)
    await db.check_runtime_privileges()
    await db.check_schema()
    assert await db.consume_rate_limit("owner-a") is True
    app.dependency_overrides[require_user] = lambda: Principal("owner-a")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/sessions")
        assert response.status_code == 200
        sid = response.json()["session_id"]
        doc = await db.fetchval(
            "INSERT INTO documents(session_id,filename,file_size,total_pages) VALUES($1,$2,$3,$4) RETURNING id",
            sid, "private.pdf", 100, 1,
        )
        listed = await client.get("/api/documents", params={"session_id": sid})
        assert listed.status_code == 200
        assert [row["id"] for row in listed.json()["documents"]] == [str(doc)]
        app.dependency_overrides[require_user] = lambda: Principal("owner-b")
        assert (await client.get("/api/documents", params={"session_id": sid})).status_code == 404
        assert (await client.delete("/api/documents/" + str(doc))).status_code == 404
    for sql in (
        "CREATE TABLE runtime_must_not_create(id integer)",
        "ALTER TABLE sessions ADD COLUMN runtime_must_not_alter integer",
        "TRUNCATE sessions CASCADE",
        "INSERT INTO schema_migrations(version) VALUES('runtime-must-not-write')",
    ):
        with pytest.raises(asyncpg.InsufficientPrivilegeError):
            async with connection.transaction():
                await connection.execute(sql)


async def test_runtime_provisioning_leaves_operator_sequence_private(runtime_database):
    connection, role = runtime_database
    await assume_runtime(connection, role)
    assert not await connection.fetchval("SELECT has_sequence_privilege('operator_counter', 'USAGE, SELECT, UPDATE')")


@pytest.mark.parametrize("attribute", ["SUPERUSER", "BYPASSRLS", "CREATEDB", "CREATEROLE", "REPLICATION"])
async def test_unsafe_role_attributes_are_rejected(runtime_database, attribute):
    connection, role = runtime_database
    await connection.execute('ALTER ROLE "' + role + '" ' + attribute)
    await assume_runtime(connection, role)
    with pytest.raises(RuntimeError, match="least-privilege"):
        await db.check_runtime_privileges()


@pytest.mark.parametrize("membership", ["custom-createdb", "pg_read_all_data", "pg_read_server_files"])
async def test_membership_in_privileged_role_is_rejected(runtime_database, membership):
    connection, role = runtime_database
    privileged = membership
    if membership == "custom-createdb":
        privileged = "forge_hub_priv_" + uuid.uuid4().hex
        await connection.execute('CREATE ROLE "' + privileged + '" NOLOGIN CREATEDB')
    await connection.execute('GRANT "' + privileged + '" TO "' + role + '"')
    await assume_runtime(connection, role)
    with pytest.raises(RuntimeError, match="least-privilege"):
        await db.check_runtime_privileges()


async def test_table_ownership_is_rejected_even_without_elevated_attributes(runtime_database):
    connection, role = runtime_database
    await connection.execute('ALTER TABLE sessions OWNER TO "' + role + '"')
    await assume_runtime(connection, role)
    with pytest.raises(RuntimeError, match="least-privilege"):
        await db.check_runtime_privileges()


async def test_owner_role_membership_is_rejected(runtime_database):
    connection, role = runtime_database
    owner = "forge_hub_owner_" + uuid.uuid4().hex
    await connection.execute('CREATE ROLE "' + owner + '" NOLOGIN')
    await connection.execute('ALTER TABLE sessions OWNER TO "' + owner + '"')
    await connection.execute('GRANT "' + owner + '" TO "' + role + '"')
    await assume_runtime(connection, role)
    with pytest.raises(RuntimeError, match="least-privilege"):
        await db.check_runtime_privileges()


@pytest.mark.parametrize("excess", ["schema-create", "truncate", "grant-option", "migration-write", "migration-column-write", "column-grant-option"])
async def test_excess_runtime_privileges_are_rejected(runtime_database, excess):
    connection, role = runtime_database
    schema = await connection.fetchval("SELECT current_schema()")
    grants = {
        "schema-create": 'GRANT CREATE ON SCHEMA "' + schema + '" TO "' + role + '"',
        "truncate": 'GRANT TRUNCATE ON documents TO "' + role + '"',
        "grant-option": 'GRANT SELECT ON documents TO "' + role + '" WITH GRANT OPTION',
        "migration-write": 'GRANT INSERT ON schema_migrations TO "' + role + '"',
        "migration-column-write": 'GRANT INSERT(version) ON schema_migrations TO "' + role + '"',
        "column-grant-option": 'GRANT SELECT(filename) ON documents TO "' + role + '" WITH GRANT OPTION',
    }
    await connection.execute(grants[excess])
    await assume_runtime(connection, role)
    with pytest.raises(RuntimeError, match="least-privilege"):
        await db.check_runtime_privileges()


async def test_missing_required_dml_is_not_reported_ready(runtime_database):
    connection, role = runtime_database
    await connection.execute('REVOKE INSERT ON auth_rate_limits FROM "' + role + '"')
    await assume_runtime(connection, role)
    with pytest.raises(RuntimeError, match="least-privilege"):
        await db.check_runtime_privileges()


async def test_updatable_view_cannot_replace_an_expected_application_table(runtime_database):
    connection, role = runtime_database
    await connection.execute("ALTER TABLE documents RENAME TO original_documents")
    await connection.execute("CREATE VIEW documents AS SELECT * FROM original_documents")
    await connection.execute('GRANT SELECT,INSERT,UPDATE,DELETE ON documents TO "' + role + '"')
    await assume_runtime(connection, role)
    with pytest.raises(RuntimeError, match="least-privilege"):
        await db.check_runtime_privileges()
