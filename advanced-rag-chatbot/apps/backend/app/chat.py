from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import StreamingResponse
from typing import List, AsyncGenerator
import httpx
from time import perf_counter
from loguru import logger
from sentence_transformers import SentenceTransformer
import numpy as np
import json
from .config import get_settings
from .db import fetch, execute, ensure_schema
from .schemas import ChatRequest, ChatResponse, Source, ChatHistoryResponse, ChatHistoryItem

router = APIRouter(prefix="/api/chat", tags=["chat"]) 

_model: SentenceTransformer | None = None


def get_model(settings) -> SentenceTransformer:
    global _model
    if _model is None:
      model_name = getattr(settings, "EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
      _model = SentenceTransformer(model_name)
    return _model


async def _llm_complete_openrouter(api_key: str, messages: List[dict], model: str) -> str:
    base_url = "https://openrouter.ai/api/v1"
    payload = {"model": model, "messages": messages}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=60.0, base_url=base_url) as client:
        resp = await client.post("/chat/completions", json=payload, headers=headers)
        if resp.status_code >= 400:
            logger.error("OpenRouter error {}: {}", resp.status_code, resp.text)
            raise HTTPException(status_code=502, detail="LLM service error")
        data = resp.json()
        return data["choices"][0]["message"]["content"]


def _as_pgvector(vec: List[float]) -> str:
    return "[" + ",".join(f"{v:.8f}" for v in vec) + "]"


@router.get("/history", response_model=ChatHistoryResponse)
async def get_history(session_id: str = Query(...), limit: int = Query(50, ge=1, le=200)):
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
    items: List[ChatHistoryItem] = []
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


@router.post("", response_model=ChatResponse)
async def chat(req: ChatRequest, settings=Depends(get_settings)):
    await ensure_schema()
    t0 = perf_counter()

    # 1) Embed query
    model = get_model(settings)
    q_vec = model.encode([req.query], convert_to_numpy=True, normalize_embeddings=True)[0].tolist()
    q_vec_pg = _as_pgvector(q_vec)

    # 2) Retrieve top-k similar chunks
    top_k = req.top_k or int(getattr(settings, "TOP_K_RESULTS", 5))
    threshold = req.threshold or float(getattr(settings, "SIMILARITY_THRESHOLD", 0.0))
    rows = await fetch(
        """
        SELECT c.id as chunk_id, c.content,
               1 - (e.embedding <=> q.vector) as similarity
        FROM embeddings e
        JOIN chunks c ON c.id = e.chunk_id,
             (SELECT $1::vector AS vector) AS q
        WHERE (1 - (e.embedding <=> q.vector)) >= $2
        ORDER BY e.embedding <=> q.vector ASC
        LIMIT $3
        """,
        q_vec_pg,
        threshold,
        top_k,
    )

    if not rows:
        raise HTTPException(status_code=404, detail="No relevant context found for the query.")

    sources: List[Source] = []
    context_snippets: List[str] = []
    for r in rows:
        sources.append(Source(id=str(r["chunk_id"]), score=float(r["similarity"])) )
        context_snippets.append(r["content"]) 

    system_prompt = (
        "You are an assistant that answers questions strictly using the provided context. "
        "Cite relevant parts concisely. If the answer is not in the context, say you don't know."
    )
    context_text = "\n\n".join(context_snippets)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {req.query}"},
    ]

    api_key = (
        get_settings().__dict__.get("OPENROUTER_API_KEY")
        or get_settings().__dict__.get("HORIZON_ALPHA_API_KEY")
        or get_settings().embedding_api_key
    )
    if not api_key:
        raise HTTPException(status_code=500, detail="OpenRouter API key not configured")

    answer = await _llm_complete_openrouter(api_key, messages, model="qwen/qwen3-coder:free")

    elapsed = int((perf_counter() - t0) * 1000)
    await execute(
        """
        INSERT INTO chat_history (session_id, user_message, assistant_message, source_documents, processing_time_ms)
        VALUES ($1, $2, $3, $4::jsonb, $5)
        """,
        req.session_id or "anonymous",
        req.query,
        answer,
        json.dumps([s.dict() for s in sources]),
        elapsed,
    )

    return ChatResponse(answer=answer, sources=sources, processing_time_ms=elapsed)


@router.post("/stream")
async def chat_stream(req: ChatRequest, settings=Depends(get_settings)):
    await ensure_schema()

    # Prepare context same as non-stream
    model = get_model(settings)
    q_vec = model.encode([req.query], convert_to_numpy=True, normalize_embeddings=True)[0].tolist()
    q_vec_pg = _as_pgvector(q_vec)
    top_k = req.top_k or int(getattr(settings, "TOP_K_RESULTS", 5))
    threshold = req.threshold or float(getattr(settings, "SIMILARITY_THRESHOLD", 0.0))
    rows = await fetch(
        """
        SELECT c.id as chunk_id, c.content,
               1 - (e.embedding <=> q.vector) as similarity
        FROM embeddings e
        JOIN chunks c ON c.id = e.chunk_id,
             (SELECT $1::vector AS vector) AS q
        WHERE (1 - (e.embedding <=> q.vector)) >= $2
        ORDER BY e.embedding <=> q.vector ASC
        LIMIT $3
        """,
        q_vec_pg,
        threshold,
        top_k,
    )
    if not rows:
        raise HTTPException(status_code=404, detail="No relevant context found for the query.")

    sources: List[Source] = []
    context_snippets: List[str] = []
    for r in rows:
        sources.append(Source(id=str(r["chunk_id"]), score=float(r["similarity"])) )
        context_snippets.append(r["content"]) 

    system_prompt = (
        "You are an assistant that answers questions strictly using the provided context. "
        "Cite relevant parts concisely. If the answer is not in the context, say you don't know."
    )
    context_text = "\n\n".join(context_snippets)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {req.query}"},
    ]

    api_key = (
        get_settings().__dict__.get("OPENROUTER_API_KEY")
        or get_settings().__dict__.get("HORIZON_ALPHA_API_KEY")
        or get_settings().embedding_api_key
    )
    if not api_key:
        raise HTTPException(status_code=500, detail="OpenRouter API key not configured")

    async def event_stream() -> AsyncGenerator[bytes, None]:
        base_url = "https://openrouter.ai/api/v1"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"model": "qwen/qwen3-coder:free", "messages": messages, "stream": True}
        full_text: List[str] = []
        start = perf_counter()
        async with httpx.AsyncClient(timeout=None, base_url=base_url) as client:
            async with client.stream("POST", "/chat/completions", json=payload, headers=headers) as resp:
                if resp.status_code >= 400:
                    txt = await resp.aread()
                    yield f"event: error\ndata: {txt.decode()}\n\n".encode()
                    return
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                        delta = obj.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            full_text.append(content)
                            yield content.encode()
                    except Exception:
                        continue
        # store history at end
        elapsed = int((perf_counter() - start) * 1000)
        try:
            await execute(
                """
                INSERT INTO chat_history (session_id, user_message, assistant_message, source_documents, processing_time_ms)
                VALUES ($1, $2, $3, $4::jsonb, $5)
                """,
                req.session_id or "anonymous",
                req.query,
                "".join(full_text),
                json.dumps([s.dict() for s in sources]),
                elapsed,
            )
        except Exception as e:
            logger.warning("Failed to store streamed chat history: {}", e)

    return StreamingResponse(event_stream(), media_type="text/plain")
