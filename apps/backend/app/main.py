import asyncio
from contextlib import asynccontextmanager, suppress
from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from loguru import logger
from .config import get_settings
from .auth import Principal, require_user, validate_auth_configuration
from .documents import router as documents_router
from .embeddings import router as embeddings_router
from .chat import router as chat_router
from .retrieval import router as retrieval_router
from .sessions import router as sessions_router
from .db import get_pool, close_pool, ensure_schema, check_schema, check_runtime_privileges, cleanup_expired_sessions, owner_stats
from .schemas import StatsResponse
from .middleware import RequestLimits


def create_app() -> FastAPI:
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app):
        validate_auth_configuration(settings)
        async def maintenance():
            while True:
                try:
                    await cleanup_expired_sessions()
                except Exception as error:
                    logger.error("Expiry maintenance unavailable ({})", type(error).__name__)
                await asyncio.sleep(min(60, settings.session_timeout_seconds))
        worker = None
        try:
            await get_pool()
            if settings.run_migrations:
                await ensure_schema()
            else:
                await check_schema()
            if settings.environment == "production":
                await check_runtime_privileges()
            worker = asyncio.create_task(maintenance())
            yield
        finally:
            if worker is not None:
                worker.cancel()
                with suppress(asyncio.CancelledError):
                    await worker
            await close_pool()

    app = FastAPI(title="Forge Hub RAG System", version="1.0.0", lifespan=lifespan)

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in settings.allowed_origins.split(",")],
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Authorization", "Content-Type", "Accept", "X-Session-ID"],
    )
    if settings.environment == "production":
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=[host.strip() for host in settings.allowed_hosts.split(",") if host.strip()])
    app.add_middleware(RequestLimits, max_body_bytes=settings.max_total_bytes + 1024 * 1024,
                       max_inflight=settings.max_inflight_requests)

    @app.get("/health")
    async def health():
        return {"status": "alive"}

    @app.get("/ready")
    async def readiness():
        try:
            validate_auth_configuration(get_settings())
            if settings.environment == "production":
                await check_runtime_privileges()
            pool = await get_pool()
            if pool is None:
                return JSONResponse({"status": "development", "storage": "memory", "durable": False}, status_code=503)
            async with pool.acquire(timeout=3) as connection:
                await connection.fetchval("SELECT 1", timeout=3)
            return {"status": "ready", "storage": "postgres", "durable": True}
        except Exception:
            return JSONResponse({"status": "unavailable"}, status_code=503)

    @app.get("/api/stats", response_model=StatsResponse)
    async def stats(principal: Principal = Depends(require_user)):
        return StatsResponse(**await owner_stats(principal.user_id))

    @app.exception_handler(Exception)
    async def unhandled_error(request, exception):
        logger.error("API request failed ({})", type(exception).__name__)
        return JSONResponse({"detail": "The service could not complete the request"}, status_code=503,
                            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})

    # Routers
    app.include_router(documents_router)
    app.include_router(embeddings_router)
    app.include_router(retrieval_router)
    app.include_router(chat_router)
    app.include_router(sessions_router)

    return app

