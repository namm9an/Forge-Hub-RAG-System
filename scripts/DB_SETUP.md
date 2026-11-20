# Database setup

Use PostgreSQL 16 with `pgvector` and `pgcrypto`, Python 3.12 with the backend dependencies, and
`psql` for permission provisioning and the database smoke check. See the [backend setup](../apps/backend/README.md)
for the dependency lock and model readiness requirements.

## Apply migrations

For local development, set `FORGE_HUB_POSTGRES_PASSWORD` privately before running `docker compose up -d postgres`
from the repository root. The loopback database is `forge_hub`. Compose runs `bootstrap.sql` only when it
initializes an empty data directory. Run the versioned migrator even on a fresh Compose database to
record the expected migrations. Startup migrations default to off and production requires them off.

Before upgrading an existing database, confirm the target, review the migration files and verify a
backup and restore procedure. Set `VECTOR_DB_URL` to the migration connection in your operator
environment. This identity needs the privileges to create or upgrade the application objects and
install the two extensions if they are absent. The connection's search path must select the
application schema first and include the schemas that provide the extensions.

From the repository root, using the backend environment:

```sh
cd apps/backend
python -m app.migrate --apply
```

The migrator applies and records the versions in one transaction. After a failure, fix the cause
before retrying. Reverting a successfully applied data migration requires the reviewed restore plan.

## Provision and verify the runtime login

For production, use an operator authorized to manage roles and grant privileges on the migrated
schema. With `OPERATOR_DSN` set privately, run from the repository root:

```sh
psql "$OPERATOR_DSN" -X -v ON_ERROR_STOP=1 -f scripts/runtime_role.sql
```

The script provisions the `forge_hub_runtime` permission role. As the operator, grant that role to
a dedicated non-owner login that inherits its privileges. The login needs its own credential,
access to the database, and the same application schema search path. Keep it out of owner and
privileged roles. Configure the backend with this login's connection and `RUN_MIGRATIONS=false`.

Set `RUNTIME_DSN` privately to the connection the backend will use. For local development it can
reference the local Compose login.

```sh
psql "$RUNTIME_DSN" -X -v ON_ERROR_STOP=1 -f scripts/validate_db.sql
```

The read-only check requires both extensions, all application tables in the active schema, and the
expected migration records. After starting the backend, use `/ready` to confirm runtime permission
checks in production, PostgreSQL connectivity and model readiness.
