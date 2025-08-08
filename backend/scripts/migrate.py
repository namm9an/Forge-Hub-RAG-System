#!/usr/bin/env python3
"""
Run database migrations on startup when RUN_MIGRATIONS=1.

- Uses asyncpg connection pool
- Creates a migrations table to track applied migrations
- Applies SQL files in backend/migrations in lexical order
- Idempotent: each migration runs inside a transaction and is recorded

Env vars used (from AppSettings):
- APP_DATABASE_URL
- APP_DB_POOL_MIN_SIZE (optional)
- APP_DB_POOL_MAX_SIZE (optional)
- RUN_MIGRATIONS
"""
from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import asyncpg  # type: ignore
import structlog

# Local import path safety when executed standalone
BASE_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = BASE_DIR
sys.path.insert(0, str(BASE_DIR / "src"))

from app.config.settings import get_settings  # type: ignore

logger = structlog.get_logger(__name__)


MIGRATIONS_DIR = BASE_DIR / "migrations"
MIGRATION_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS public.schema_migrations (
    id SERIAL PRIMARY KEY,
    filename TEXT NOT NULL UNIQUE,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT TIMEZONE('utc'::text, NOW())
);
"""


@dataclass
class Migration:
    filename: str
    path: Path


async def _ensure_pool(dsn: str, min_size: int, max_size: int) -> asyncpg.Pool:
    return await asyncpg.create_pool(dsn=dsn, min_size=min_size, max_size=max_size)


async def _ensure_migration_table(conn: asyncpg.Connection) -> None:
    await conn.execute(MIGRATION_TABLE_SQL)


async def _get_applied(conn: asyncpg.Connection) -> set[str]:
    rows = await conn.fetch("SELECT filename FROM public.schema_migrations")
    return {r["filename"] for r in rows}


def _load_migrations() -> List[Migration]:
    files = sorted([p for p in MIGRATIONS_DIR.glob("*.sql")])
    return [Migration(filename=p.name, path=p) for p in files]


async def _apply_migration(conn: asyncpg.Connection, mig: Migration) -> None:
    sql = mig.path.read_text(encoding="utf-8")
    # Wrap in transaction; sql file may include its own BEGIN/COMMIT which is fine
    async with conn.transaction():
        await conn.execute(sql)
        await conn.execute(
            "INSERT INTO public.schema_migrations (filename) VALUES ($1) ON CONFLICT (filename) DO NOTHING",
            mig.filename,
        )


async def run() -> int:
    # Skip unless explicitly enabled
    if os.getenv("RUN_MIGRATIONS", "0") not in ("1", "true", "True"):  # pragma: no cover
        logger.info("migrations_skipped", reason="RUN_MIGRATIONS not set to 1")
        return 0

    settings = get_settings()
    dsn = settings.DATABASE_URL
    if not dsn:
        logger.error("migrations_no_database_url")
        return 1

    min_size = max(1, int(os.getenv("APP_DB_POOL_MIN_SIZE", settings.DB_POOL_MIN_SIZE or 1)))
    max_size = max(min_size, int(os.getenv("APP_DB_POOL_MAX_SIZE", settings.DB_POOL_MAX_SIZE or 10)))

    pool = await _ensure_pool(dsn, min_size, max_size)
    try:
        async with pool.acquire() as conn:
            await _ensure_migration_table(conn)
            applied = await _get_applied(conn)

        migrations = _load_migrations()
        to_apply = [m for m in migrations if m.filename not in applied]
        if not to_apply:
            logger.info("migrations_none_pending")
            return 0

        logger.info("migrations_start", pending=len(to_apply))
        for mig in to_apply:
            try:
                async with pool.acquire() as conn:
                    await _apply_migration(conn, mig)
                logger.info("migration_applied", file=mig.filename)
            except Exception as e:  # pragma: no cover - stop at first failure
                logger.error("migration_failed", file=mig.filename, error=str(e))
                return 1

        logger.info("migrations_completed", applied=len(to_apply))
        return 0
    finally:
        await pool.close()


if __name__ == "__main__":
    code = asyncio.run(run())
    sys.exit(code)

