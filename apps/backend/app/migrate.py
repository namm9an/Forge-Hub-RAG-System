"""Operator-only schema upgrades; never invoked by a production request handler."""
import argparse
import asyncio


async def apply_migrations():
    from .db import ensure_schema, close_pool, get_pool
    try:
        if await get_pool() is None:
            raise RuntimeError("Schema migrations require PostgreSQL")
        await ensure_schema()
    finally:
        await close_pool()


def main():
    parser = argparse.ArgumentParser(description="Apply reviewed database migrations with a separate migration identity")
    parser.add_argument("--apply", action="store_true", help="Explicitly apply migrations to the configured database")
    args = parser.parse_args()
    if not args.apply:
        parser.error("No changes made. Review migration files/backups and use --apply with migration credentials.")
    asyncio.run(apply_migrations())


if __name__ == "__main__":
    main()
