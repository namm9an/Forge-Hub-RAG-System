from __future__ import annotations

import time
import uuid
from contextvars import ContextVar
from typing import Callable, Optional

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

logger = structlog.get_logger(__name__)

# Context variables used for logging and per-request metrics
request_id_ctx: ContextVar[str | None] = ContextVar("request_id", default=None)
_db_query_count_ctx: ContextVar[int] = ContextVar("db_query_count", default=0)


def reset_db_query_count() -> None:
    _db_query_count_ctx.set(0)


def increment_db_query_count(n: int = 1) -> None:
    try:
        current = _db_query_count_ctx.get()
        _db_query_count_ctx.set(current + n)
    except Exception:
        _db_query_count_ctx.set(n)


def get_db_query_count() -> int:
    return _db_query_count_ctx.get()


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Ensures each request has a correlation ID and binds it to logs.

    - Reads incoming X-Request-ID or generates a UUID4.
    - Exposes it via request.state.request_id and response header.
    - Binds to structlog contextvars for downstream logs.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:  # type: ignore[override]
        incoming = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = incoming
        token = request_id_ctx.set(incoming)
        structlog.contextvars.bind_contextvars(request_id=incoming)
        try:
            response = await call_next(request)
        finally:
            # Clear request_id from context to avoid leakage across tasks
            request_id_ctx.reset(token)
            structlog.contextvars.unbind_contextvars("request_id")
        response.headers["X-Request-ID"] = incoming
        return response


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """Measures request time, DB query count, and logs an access event."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:  # type: ignore[override]
        t0 = time.time()
        reset_db_query_count()
        try:
            response = await call_next(request)
            status = response.status_code
        except Exception:
            # Let exception handlers deal with response, but log timing here
            status = 500
            raise
        finally:
            latency_ms = int((time.time() - t0) * 1000)
            dbq = get_db_query_count()
            logger.info(
                "access_log",
                method=request.method,
                path=str(request.url.path),
                status=status,
                latency_ms=latency_ms,
                db_queries=dbq,
                client_ip=request.client.host if request.client else None,
                user_agent=request.headers.get("user-agent"),
            )
        # Expose metrics via headers
        response.headers["Server-Timing"] = f"app;dur={latency_ms}"
        response.headers["X-DB-Queries"] = str(dbq)
        return response


class GracefulShutdownLifespan:
    """ASGI lifespan manager to ensure graceful shutdown with retry for cleanup tasks."""

    def __init__(self, retries: int = 3, delay_sec: float = 0.5) -> None:
        self.retries = retries
        self.delay_sec = delay_sec

    async def __call__(self, app):  # FastAPI passes itself
        from contextlib import asynccontextmanager
        import asyncio

        @asynccontextmanager
        async def lifespan(_app):
            yield
            # On shutdown, attempt to flush logs/telemetry with retries
            for attempt in range(1, self.retries + 1):
                try:
                    # Flush structlog/standard logging handlers if any
                    for h in list(logger._logger.handlers) if hasattr(logger, "_logger") else []:  # type: ignore[attr-defined]
                        try:
                            h.flush()
                        except Exception:
                            pass
                    # Give OTEL exporter a chance to drain
                    await asyncio.sleep(self.delay_sec)
                    break
                except Exception:
                    if attempt == self.retries:
                        break
                    await asyncio.sleep(self.delay_sec)

        return lifespan
