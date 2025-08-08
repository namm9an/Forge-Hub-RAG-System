from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.services.rag_pipeline import generate_answer, record_history
from app.utils.rate_limit import rate_limit_tracking

logger = structlog.get_logger(__name__)

router = APIRouter()


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000)
    session_id: Optional[str] = Field(default=None, description="Client-generated UUID for session memory")
    top_k: Optional[int] = Field(default=12, ge=1, le=50)


class ChatResponse(BaseModel):
    request_id: str
    status: str
    message: str
    answer: str
    context: List[Dict[str, Any]]
    citations: List[Any]
    usage: Dict[str, Any]
    metadata: Dict[str, Any]
    error: Optional[Dict[str, Any]] = None


async def _stream_tokens(text: str, delay_sec: float = 0.0) -> AsyncGenerator[str, None]:
    # Simple token stream by words; replace with real model streaming as needed
    for tok in text.split():
        yield tok + " "
        if delay_sec:
            await asyncio.sleep(delay_sec)


@router.post("/v1/chat", response_model=ChatResponse, tags=["chat"])
async def chat_rest(
    payload: ChatRequest,
    request: Request,
    _: None = Depends(rate_limit_tracking(limit=60, window_seconds=60)),
) -> JSONResponse:
    started = time.time()
    request_id = str(uuid.uuid4())
    client_ip = request.client.host if request.client else "unknown"

    try:
        result = await generate_answer(
            question=payload.message.strip(), session_id=payload.session_id, k=payload.top_k or 12
        )
        answer = result.get("answer") or ""
        context = result.get("context", [])
        citations = result.get("citations", [])

        usage = {
            "prompt_chars": len(payload.message or ""),
            "completion_chars": len(answer),
            "total_chars": len(payload.message or "") + len(answer),
            "latency_ms": int((time.time() - started) * 1000),
        }
        meta = {
            "request_id": request_id,
            "ip": client_ip,
            "session_id": payload.session_id,
            "retrieval_top_k": payload.top_k,
        }

        return JSONResponse(
            status_code=200,
            content={
                "request_id": request_id,
                "status": "ok",
                "message": "success",
                "answer": answer,
                "context": context,
                "citations": citations,
                "usage": usage,
                "metadata": meta,
                "error": None,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("chat_rest_failed", error=str(e), request_id=request_id)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.websocket("/ws/chat")
async def chat_ws(websocket: WebSocket):
    # Accept connection after a connection-level rate check
    client_ip = websocket.client.host if websocket.client else "unknown"
    # Simple connection rate limit (60/min)
    limiter = rate_limit_tracking(limit=60, window_seconds=60)
    try:
        # Use the dependency function manually for WebSocket (no Depends support here)
        await limiter(WebSocketRequestShim(client_ip))
    except HTTPException as e:
        await websocket.close(code=4408)  # policy violation
        return

    await websocket.accept()
    await websocket.send_json({"type": "ready", "message": "connected"})

    try:
        while True:
            data = await websocket.receive_json()
            message = str(data.get("message", "")).strip()
            session_id = data.get("session_id")
            top_k = int(data.get("top_k", 12))

            if not message:
                await websocket.send_json({"type": "error", "error": {"code": "invalid_input", "message": "message is required"}})
                continue

            req_id = str(uuid.uuid4())
            t0 = time.time()

            # Rate limit per message as well
            try:
                await limiter(WebSocketRequestShim(client_ip))
            except HTTPException as e:
                await websocket.send_json({"type": "error", "error": {"code": "rate_limited", "message": "Too Many Requests"}})
                continue

            try:
                result = await generate_answer(question=message, session_id=session_id, k=top_k)
                answer = result.get("answer") or ""
                context = result.get("context", [])
                citations = result.get("citations", [])

                await websocket.send_json({
                    "type": "metadata",
                    "request_id": req_id,
                    "session_id": session_id,
                    "retrieval_top_k": top_k,
                })
                await websocket.send_json({"type": "context", "context": context, "citations": citations})

                # Stream tokens
                async for tok in _stream_tokens(answer):
                    await websocket.send_json({"type": "token", "delta": tok})
                latency_ms = int((time.time() - t0) * 1000)
                await websocket.send_json({
                    "type": "done",
                    "usage": {
                        "prompt_chars": len(message),
                        "completion_chars": len(answer),
                        "total_chars": len(message) + len(answer),
                        "latency_ms": latency_ms,
                    },
                })
            except Exception as e:
                logger.warning("chat_ws_error", error=str(e))
                await websocket.send_json({"type": "error", "error": {"code": "internal_error", "message": "Internal server error"}})
    except WebSocketDisconnect:
        return


class WebSocketRequestShim:
    """Shim object to pass client IP to rate limiter for WebSocket flows."""

    def __init__(self, client_ip: str) -> None:
        self.client = type("Client", (), {"host": client_ip})
