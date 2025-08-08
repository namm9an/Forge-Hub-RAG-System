from fastapi import FastAPI, HTTPException
from fastapi.routing import APIRouter
import structlog
import time
from typing import Any, Dict
import os
import importlib.util
from pathlib import Path

from app.config.settings import get_settings
from app.api.routers.chat import router as chat_router
from app.services.supabase_client import get_supabase


logger = structlog.get_logger()


def _run_startup_migrations_if_enabled() -> None:
    """Dynamically import and run the migration script if RUN_MIGRATIONS=1.

    Uses importlib to load backend/scripts/migrate.py by path to avoid Pythonpath issues.
    """
    run_flag = os.getenv("RUN_MIGRATIONS", "0")
    if run_flag not in ("1", "true", "True"):
        return
    try:
        base_dir = Path(__file__).resolve().parents[3]  # points to backend/
        migrate_path = base_dir / "scripts" / "migrate.py"
        if not migrate_path.exists():
            logger.warning("migrate_script_missing", path=str(migrate_path))
            return
        spec = importlib.util.spec_from_file_location("_backend_migrate", str(migrate_path))
        if not spec or not spec.loader:
            logger.warning("migrate_spec_failed")
            return
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[assignment]
        # Run the async run() entrypoint
        import asyncio
        asyncio.get_event_loop()
        code = asyncio.run(mod.run())  # type: ignore[attr-defined]
        if code != 0:
            logger.error("startup_migrations_failed", code=code)
        else:
            logger.info("startup_migrations_completed")
    except Exception as e:  # pragma: no cover
        logger.error("startup_migrations_error", error=str(e))


def create_app() -> FastAPI:
    settings = get_settings()

    # Structured logging
    from app.observability.logging import configure_logging
    configure_logging(level="INFO" if not settings.DEBUG else "DEBUG")

    # FastAPI app with graceful shutdown lifespan
    from app.observability.middleware import (
        CorrelationIdMiddleware,
        ObservabilityMiddleware,
        GracefulShutdownLifespan,
    )
    app = FastAPI(title=settings.APP_NAME, version="0.1.0", lifespan=GracefulShutdownLifespan())

    # Problem+JSON exception mapping
    from app.observability.errors import http_exception_handler, unhandled_exception_handler
    from fastapi.exceptions import RequestValidationError
    from fastapi import HTTPException

    app.add_middleware(CorrelationIdMiddleware)
    app.add_middleware(ObservabilityMiddleware)

    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, lambda req, exc: http_exception_handler(req, HTTPException(status_code=422, detail="Unprocessable Entity")))
    app.add_exception_handler(Exception, unhandled_exception_handler)

    # Startup hook for migrations
    @app.on_event("startup")
    async def _startup() -> None:
        # Run sync wrapper to call the migration runner
        _run_startup_migrations_if_enabled()
        # Initialize OpenTelemetry if available
        try:
            from app.observability.telemetry import init_opentelemetry

            init_opentelemetry(app)
        except Exception:
            pass

    # Routers
    app.include_router(chat_router, prefix="")

    @app.get("/health", tags=["system"])  # Health check endpoint
    async def health():
        return {"status": "ok"}

    @app.get("/config", tags=["system"])  # Expose non-sensitive config for debugging
    async def config_sample():
        s = get_settings()
        # Only return safe fields
        return {
            "env": s.ENV,
            "debug": s.DEBUG,
            "host": s.HOST,
            "port": s.PORT,
            "db_pool_min": s.DB_POOL_MIN_SIZE,
            "db_pool_max": s.DB_POOL_MAX_SIZE,
        }

    @app.get("/metrics", tags=["system"])  # Basic in-process metrics
    async def metrics() -> Dict[str, Any]:
        # Minimal example; can be extended to Prometheus later
        return {
            "service": settings.APP_NAME,
            "env": settings.ENV,
            "timestamp": time.time(),
        }

    @app.get("/vector-stats", tags=["system"])  # Supabase RPC: get_vector_stats()
    async def vector_stats() -> Dict[str, Any]:
        try:
            client = get_supabase()
            if client is None:
                raise HTTPException(status_code=503, detail="Supabase not configured")
            from app.observability.middleware import increment_db_query_count
            resp = client.rpc("get_vector_stats", {}).execute()
            increment_db_query_count(1)
            data = getattr(resp, "data", None)
            return {"status": "ok", "data": data}
        except HTTPException:
            raise
        except Exception as e:
            logger.error("vector_stats_failed", error=str(e))
            raise HTTPException(status_code=500, detail="Failed to fetch vector stats")

    return app


app = create_app()
