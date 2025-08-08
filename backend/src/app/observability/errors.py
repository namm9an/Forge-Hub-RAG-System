from __future__ import annotations

import uuid
from typing import Any, Dict

import structlog
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import RequestResponseEndpoint

logger = structlog.get_logger(__name__)


def problem_json(
    *,
    request: Request,
    status: int,
    title: str,
    detail: str | None = None,
    type_: str = "about:blank",
    extra: Dict[str, Any] | None = None,
) -> JSONResponse:
    instance = str(request.url)
    rid = request.state.request_id if hasattr(request.state, "request_id") else None
    body: Dict[str, Any] = {
        "type": type_,
        "title": title,
        "status": status,
        "detail": detail,
        "instance": instance,
    }
    if rid:
        body["request_id"] = rid
    if extra:
        body.update(extra)
    return JSONResponse(status_code=status, content=body, media_type="application/problem+json")


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    title = exc.detail if isinstance(exc.detail, str) else "HTTP Exception"
    return problem_json(request=request, status=exc.status_code, title=title)


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("unhandled_exception", error=str(exc))
    return problem_json(
        request=request,
        status=500,
        title="Internal Server Error",
        detail="An unexpected error occurred",
        type_="urn:problem-type:internal-server-error",
    )
