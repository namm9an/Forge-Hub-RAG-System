# Forge Hub backend

Frontend work is deferred. Existing clients must adopt the authenticated API contract before integration.

Use Python 3.12. The verified Windows dependency set is `requirements.win-py312.lock`;
`requirements.txt` contains supported bounds for resolution on other platforms.

```powershell
# From apps/backend, using a Python 3.12 environment:
python -m pip install -r requirements.win-py312.lock
```

Configure private environment values from the root `.env.example`. PostgreSQL and the trusted
RS256 issuer, audience and RSA public PEM are required. Never use a private signing key as the
verification key. Gemini and OpenRouter keys are optional; local extraction is labeled in responses.
The chosen account/login provider, MFA/recovery and key-rotation integration remain separate open gates.

Schema upgrades use reviewed migration credentials, with an operator backup/rollback plan:

```powershell
python -m app.migrate --apply
```

Do not run that command on an existing database merely to try the app. Local automated tests use
an independent disposable database. The raw bootstrap SQL alone does not record migration versions.
Production uses a dedicated restricted runtime login, the operator-only `scripts/runtime_role.sql`
permission role and `RUN_MIGRATIONS=false`; it must not use the migration/owner login.
See [database setup](../../scripts/DB_SETUP.md).

```powershell
# Direct local service; no forwarded headers are trusted in this example.
python -m uvicorn app.main:create_app --factory --http h11 --no-proxy-headers --port 8000
```

Startup checks schema versions, checks production runtime privileges and exercises actual encoder
and neural reranker inference before becoming ready. Prepare/download model assets in a controlled
environment before deployment. `/health` is process liveness; `/ready` is dependency readiness.
Memory mode is explicit development/test-only and never reports durable production readiness.

Every private API requires `Authorization: Bearer <token>`. Identity is the exact signed issuer+subject;
caller email/user/IP headers are not credentials. Session IDs must be explicit and owned. Upload
accepts matching query/form/header scopes and rejects conflicts. Deletion, history, search, stats,
embedding repair and cleanup follow the same ownership rules. Session inactivity expires its data.

Production requires explicit `ALLOWED_HOSTS` and browser origins. A reverse proxy must restrict
direct backend access and trust forwarding headers only from known hops. TLS, HTTP/2 reset budgets,
shared-cache behavior and ingress fairness require checks on the actual deployed chain.

From the repository root, run `python -m pytest apps/backend/tests` for the offline suite.
To run PostgreSQL checks, start a separate project using `docker-compose.test.yml` and set
`FORGE_HUB_TEST_DATABASE_URL` to its loopback `forge_hub_test` database. Then add `--run-postgres`.
The loopback HTTP/1 checks are separately opt-in using `--run-protocol`.
