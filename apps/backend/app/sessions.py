"""Private, expiring sessions; identifiers are never proof of ownership."""

import json
import re
import uuid
from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, Query

from .auth import Principal, require_user
from .config import get_settings
from .db import execute, fetch, transaction, create_session_record
from .schemas import ChatHistoryResponse, ChatHistoryItem, Source

router = APIRouter(prefix="/api/sessions", tags=["sessions"])
_SESSION_ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")


async def authorize_session(session_id: str, user_id: str, *, create: bool = False) -> str:
    """Authorize the live session and return its immutable database row ID."""
    if not isinstance(session_id, str) or not _SESSION_ID.fullmatch(session_id) or session_id in {"default", "anonymous"}:
        raise HTTPException(status_code=422, detail="An explicit session identifier of 1-128 safe characters is required")
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    timeout = timedelta(seconds=get_settings().session_timeout_seconds)
    async with transaction():
        if create:
            await create_session_record(session_id, user_id)
        rows = await fetch(
            "SELECT id FROM sessions "
            "WHERE session_id=$1 AND user_id=$2 AND last_activity>NOW()-$3::interval FOR UPDATE",
            session_id, user_id, timeout,
        )
        if not rows:
            raise HTTPException(status_code=404, detail="Session not found")
        await execute(
            "UPDATE sessions SET last_activity=NOW() WHERE session_id=$1 AND user_id=$2",
            session_id, user_id,
        )
    return str(rows[0]["id"])


@router.post("")
async def create_session(user: Principal = Depends(require_user)):
    sid = str(uuid.uuid4())
    await authorize_session(sid, user.user_id, create=True)
    return {"session_id": sid}


@router.post("/refresh")
async def refresh_session(session_id: str = Query(...), user: Principal = Depends(require_user)):
    await authorize_session(session_id, user.user_id)
    return {"status": "ok"}


@router.post("/cleanup")
async def cleanup(user: Principal = Depends(require_user)):
    timeout = timedelta(seconds=get_settings().session_timeout_seconds)
    async with transaction():
        removed = await fetch(
            "DELETE FROM sessions WHERE user_id=$1 AND last_activity<=NOW()-$2::interval RETURNING session_id",
            user.user_id, timeout,
        )
    return {"deleted": len(removed)}


async def read_history(session_id: str, limit: int, user_id: str) -> ChatHistoryResponse:
    async with transaction():
        await authorize_session(session_id, user_id)
        rows = await fetch(
            "SELECT h.id, h.session_id, h.user_message, h.assistant_message, h.source_documents, "
            "h.processing_time_ms, h.retrieval_ms, h.generation_ms, h.created_at "
            "FROM chat_history h JOIN sessions s ON s.session_id=h.session_id "
            "WHERE h.session_id=$1 AND s.user_id=$3 ORDER BY h.created_at DESC,h.id DESC LIMIT $2",
            session_id, limit, user_id,
        )
    items = []
    for row in rows:
        try:
            values = row.get("source_documents") or []
            values = json.loads(values) if isinstance(values, str) else values
            sources = [Source(**source) for source in values or []]
        except (ValueError, TypeError):
            sources = []
        items.append(ChatHistoryItem(
            id=str(row["id"]), session_id=row["session_id"],
            user_message=row["user_message"], assistant_message=row["assistant_message"],
            source_documents=sources,
            processing_time_ms=row.get("processing_time_ms"),
            retrieval_ms=row.get("retrieval_ms"), generation_ms=row.get("generation_ms"),
            created_at=str(row["created_at"]),
        ))
    return ChatHistoryResponse(messages=items)


@router.get("/history", response_model=ChatHistoryResponse)
async def session_history(
    session_id: str = Query(...),
    limit: int = Query(50, ge=1, le=200),
    user: Principal = Depends(require_user),
):
    return await read_history(session_id, limit, user.user_id)


@router.delete("/{session_id}")
async def delete_session(session_id: str, user: Principal = Depends(require_user)):
    async with transaction():
        await authorize_session(session_id, user.user_id)
        await execute("DELETE FROM sessions WHERE session_id=$1 AND user_id=$2", session_id, user.user_id)
    return {"status": "ok", "deleted_session_id": session_id}
