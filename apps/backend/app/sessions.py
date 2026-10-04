from fastapi import APIRouter, Query
from .db import fetchval, execute, fetch, ensure_schema
from .schemas import ChatHistoryResponse, ChatHistoryItem, Source
import json

router = APIRouter(prefix="/api/sessions", tags=["sessions"]) 


@router.post("")
async def create_session():
    await ensure_schema()
    sid = await fetchval("INSERT INTO sessions (session_id) VALUES (gen_random_uuid()::text) RETURNING session_id")
    return {"session_id": sid}


@router.post("/refresh")
async def refresh_session(session_id: str = Query(...)):
    await ensure_schema()
    await execute("UPDATE sessions SET last_activity = NOW() WHERE session_id = $1", session_id)
    return {"status": "ok"}


@router.post("/cleanup")
async def cleanup(max_age_minutes: int = Query(60, ge=1, le=1440)):
    await ensure_schema()
    deleted = await fetchval("SELECT cleanup_expired_sessions($1::interval)", f"{max_age_minutes} minutes")
    return {"deleted": int(deleted or 0)}


@router.get("/history", response_model=ChatHistoryResponse)
async def session_history(session_id: str = Query(...), limit: int = Query(50, ge=1, le=200)):
    await ensure_schema()
    rows = await fetch(
        """
        SELECT id, session_id, user_message, assistant_message, source_documents, processing_time_ms, created_at
        FROM chat_history
        WHERE session_id = $1
        ORDER BY created_at DESC
        LIMIT $2
        """,
        session_id,
        limit,
    )
    items = []
    for r in rows:
        src = None
        if r["source_documents"]:
            try:
                src_list = r["source_documents"] if isinstance(r["source_documents"], list) else json.loads(r["source_documents"])  # type: ignore
                src = [Source(**s) for s in src_list]
            except Exception:
                src = None
        items.append(
            ChatHistoryItem(
                id=str(r["id"]),
                session_id=r["session_id"],
                user_message=r["user_message"],
                assistant_message=r["assistant_message"],
                source_documents=src,
                processing_time_ms=r["processing_time_ms"],
                created_at=str(r["created_at"]),
            )
        )
    return ChatHistoryResponse(messages=items)
