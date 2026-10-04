"""
Forge Hub RAG System — Gated Generation & Streaming Pipeline
Complies with:
- Rule D-1: Out-of-Domain Refusal Gate (Evidence Confidence Evaluator)
- Rule D-2: Sentence-Level Noise Pruning (Context Filtering)
- Rule D-3: Strict Grounded Prompting with Inline Citations
- Rule D-4: Streaming Token Delivery via SSE and Chunked Transfer Encoding
- Stage 2: Quantized Cross-Encoder Reranking
"""

import asyncio
import json
from time import perf_counter
from typing import List, AsyncGenerator, Optional
from fastapi import APIRouter, HTTPException, Depends, Query, Request
from fastapi.responses import StreamingResponse
import httpx
from loguru import logger

from .config import get_settings, Settings
from .db import fetch, execute, ensure_schema
from .schemas import (
    ChatRequest,
    ChatResponse,
    Source,
    ChatHistoryResponse,
    ChatHistoryItem,
    ConfidenceLevel,
)
from .embeddings import get_model, _as_pgvector
from .retrieval import _dense_search, _sparse_search, compute_rrf_fusion
from .gating import (
    EvidenceEvaluator,
    prune_and_format_context,
    get_reranker,
    REFUSAL_MESSAGE,
)

router = APIRouter(prefix="/api/chat", tags=["chat"])

STRICT_GROUNDED_SYSTEM_PROMPT = (
    "You are Forge Hub RAG, an enterprise technical intelligence assistant.\n"
    "Your answers must be strictly grounded ONLY in the provided context passages.\n"
    "Strict Rules:\n"
    "1. Cite sources inline using the exact format: [Doc: <filename>, Page: <page_number>, Chunk: <chunk_id>].\n"
    "2. Never extrapolate, hallucinate, or use external knowledge outside the provided context.\n"
    "3. If the context does not contain sufficient information to answer the question, state exactly: "
    "'The provided documents do not contain sufficient context to answer this question.'\n"
    "4. Keep answers concise, factual, and strictly faithful to the source material."
)


async def _llm_complete_openrouter(api_key: str, messages: List[dict], model: str) -> str:
    """Invokes OpenRouter API for full text generation."""
    base_url = "https://openrouter.ai/api/v1"
    payload = {"model": model, "messages": messages, "temperature": 0.1}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=60.0, base_url=base_url) as client:
        resp = await client.post("/chat/completions", json=payload, headers=headers)
        if resp.status_code >= 400:
            logger.error("OpenRouter error {}: {}", resp.status_code, resp.text)
            raise HTTPException(status_code=502, detail="LLM service error")
        data = resp.json()
        return data["choices"][0]["message"]["content"]


def _local_synthesize_grounded(query: str, sources: List[Source]) -> str:
    """
    Extractive fallback synthesis for offline testing and local evaluation
    when an external OpenRouter API key is unconfigured.
    Guarantees strict citation formatting (Rule D-3).
    """
    if not sources:
        return REFUSAL_MESSAGE
    top = sources[0]
    return (
        f"Based on the provided documentation: {top.content.strip()} "
        f"[Doc: {top.filename}, Page: {top.page_number}, Chunk: {top.id}]"
    )


async def _perform_hybrid_retrieval(
    query: str,
    session_id: Optional[str],
    top_k: int,
    settings: Settings,
):
    """Executes parallel dense + sparse retrieval and Reciprocal Rank Fusion."""
    model = get_model(settings)
    q_vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0].tolist()
    q_vec_pg = _as_pgvector(q_vec)

    candidate_k = max(top_k * 3, 20)

    # Parallel dispatch (Rule C-4)
    dense_future = _dense_search(q_vec_pg, session_id, candidate_k)
    sparse_future = _sparse_search(query, session_id, candidate_k)

    dense_candidates, sparse_candidates = await asyncio.gather(dense_future, sparse_future)

    # RRF Fusion (Rule C-3)
    fused_candidates = compute_rrf_fusion(
        dense_results=dense_candidates,
        sparse_results=sparse_candidates,
        k=settings.rrf_k,
        dense_weight=settings.dense_weight,
        sparse_weight=settings.sparse_weight,
        top_k=candidate_k,
    )
    return fused_candidates


@router.get("/history", response_model=ChatHistoryResponse)
async def get_history(session_id: str = Query(...), limit: int = Query(50, ge=1, le=200)):
    await ensure_schema()
    rows = await fetch(
        """
        SELECT id, session_id, user_message, assistant_message, source_documents,
               processing_time_ms, retrieval_ms, generation_ms, created_at
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
        if r.get("source_documents"):
            try:
                src_list = (
                    r["source_documents"]
                    if isinstance(r["source_documents"], list)
                    else json.loads(r["source_documents"])
                )
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
                processing_time_ms=r.get("processing_time_ms"),
                retrieval_ms=r.get("retrieval_ms"),
                generation_ms=r.get("generation_ms"),
                created_at=str(r["created_at"]),
            )
        )
    return ChatHistoryResponse(messages=items)


@router.post("", response_model=ChatResponse)
async def chat(req: ChatRequest, settings: Settings = Depends(get_settings)):
    """
    Gated Generation Pipeline with Latency Profiling and Citation Grounding.
    Includes Stage 2 Cross-Encoder Reranking and Complies with Rules D-1, D-2, D-3.
    """
    await ensure_schema()
    t_total_start = perf_counter()

    # 1. Two-Stage Hybrid Retrieval
    t_ret_start = perf_counter()
    candidates = await _perform_hybrid_retrieval(
        query=req.query,
        session_id=req.session_id,
        top_k=req.top_k or settings.top_k_results,
        settings=settings,
    )

    # Stage 2: Neural Cross-Encoder Reranker
    reranker = get_reranker(settings.reranker_model)
    candidates = reranker.rerank(
        query=req.query,
        candidates=candidates,
        top_k=req.top_k or settings.top_k_results,
    )
    retrieval_ms = int((perf_counter() - t_ret_start) * 1000)

    # 2. Evidence Confidence Gating (Rule D-1)
    evaluator = EvidenceEvaluator(default_threshold=settings.similarity_threshold)
    confidence, should_refuse, refusal_msg = evaluator.evaluate(
        query=req.query,
        candidates=candidates,
        threshold=req.threshold,
    )

    if should_refuse:
        # Refusal Gate triggered: return calibrated honest response without hallucinating
        total_time_ms = int((perf_counter() - t_total_start) * 1000)
        answer = refusal_msg

        await execute(
            """
            INSERT INTO chat_history (session_id, user_message, assistant_message, source_documents,
                                      processing_time_ms, retrieval_ms, generation_ms)
            VALUES ($1, $2, $3, $4::jsonb, $5, $6, $7)
            """,
            req.session_id or "default",
            req.query,
            answer,
            json.dumps([]),
            total_time_ms,
            retrieval_ms,
            0,
        )

        return ChatResponse(
            answer=answer,
            sources=[],
            confidence=ConfidenceLevel.LOW,
            processing_time_ms=total_time_ms,
            retrieval_ms=retrieval_ms,
            generation_ms=0,
            refusal_triggered=True,
        )

    # 3. Sentence-Level Noise Pruning & Citation Prompt Construction (Rules D-2, D-3)
    context_text, sources = prune_and_format_context(req.query, candidates)

    messages = [
        {"role": "system", "content": STRICT_GROUNDED_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context passages:\n{context_text}\n\nQuestion: {req.query}\n\nProvide an attributed answer citing [Doc: filename, Page: page_number, Chunk: chunk_id].",
        },
    ]

    # 4. Strict Synthesis & Generation Latency Profiling
    t_gen_start = perf_counter()
    api_key = settings.openrouter_api_key

    if api_key:
        try:
            answer = await _llm_complete_openrouter(api_key, messages, model=settings.llm_model)
        except Exception as e:
            logger.warning("External LLM generation failed ({}); falling back to local grounded synthesis.", e)
            answer = _local_synthesize_grounded(req.query, sources)
    else:
        answer = _local_synthesize_grounded(req.query, sources)

    generation_ms = int((perf_counter() - t_gen_start) * 1000)
    total_time_ms = int((perf_counter() - t_total_start) * 1000)

    # 5. Persist Chat History with Full Latency Breakdown
    await execute(
        """
        INSERT INTO chat_history (session_id, user_message, assistant_message, source_documents,
                                  processing_time_ms, retrieval_ms, generation_ms)
        VALUES ($1, $2, $3, $4::jsonb, $5, $6, $7)
        """,
        req.session_id or "default",
        req.query,
        answer,
        json.dumps([s.model_dump() for s in sources]),
        total_time_ms,
        retrieval_ms,
        generation_ms,
    )

    return ChatResponse(
        answer=answer,
        sources=sources,
        confidence=confidence,
        processing_time_ms=total_time_ms,
        retrieval_ms=retrieval_ms,
        generation_ms=generation_ms,
        refusal_triggered=False,
    )


@router.post("/stream")
async def chat_stream(
    req: ChatRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
):
    """
    Streaming Generation supporting both Server-Sent Events (SSE) and Chunked Transfer Encoding (Rule D-4).
    Streams raw tokens without buffering the entire response.
    """
    await ensure_schema()
    t_total_start = perf_counter()

    # 1. Two-Stage Hybrid Retrieval
    t_ret_start = perf_counter()
    candidates = await _perform_hybrid_retrieval(
        query=req.query,
        session_id=req.session_id,
        top_k=req.top_k or settings.top_k_results,
        settings=settings,
    )

    # Stage 2: Neural Cross-Encoder Reranker
    reranker = get_reranker(settings.reranker_model)
    candidates = reranker.rerank(
        query=req.query,
        candidates=candidates,
        top_k=req.top_k or settings.top_k_results,
    )
    retrieval_ms = int((perf_counter() - t_ret_start) * 1000)

    # 2. Evidence Confidence Gating (Rule D-1)
    evaluator = EvidenceEvaluator(default_threshold=settings.similarity_threshold)
    confidence, should_refuse, refusal_msg = evaluator.evaluate(
        query=req.query,
        candidates=candidates,
        threshold=req.threshold,
    )

    accept_header = request.headers.get("accept", "")
    is_sse = (
        "text/event-stream" in accept_header
        or (req.stream_format and req.stream_format.lower() == "sse")
        or request.query_params.get("format") == "sse"
    )

    stream_headers = {
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }

    async def refusal_stream() -> AsyncGenerator[bytes, None]:
        if is_sse:
            yield f"data: {json.dumps({'token': refusal_msg, 'refusal': True})}\n\n".encode("utf-8")
            yield b"data: [DONE]\n\n"
        else:
            yield refusal_msg.encode("utf-8")

        total_elapsed = int((perf_counter() - t_total_start) * 1000)
        try:
            await execute(
                """
                INSERT INTO chat_history (session_id, user_message, assistant_message, source_documents,
                                          processing_time_ms, retrieval_ms, generation_ms)
                VALUES ($1, $2, $3, $4::jsonb, $5, $6, $7)
                """,
                req.session_id or "default",
                req.query,
                refusal_msg,
                json.dumps([]),
                total_elapsed,
                retrieval_ms,
                0,
            )
        except Exception as e:
            logger.warning("Failed storing refusal stream in history: {}", e)

    if should_refuse:
        return StreamingResponse(
            refusal_stream(),
            media_type="text/event-stream" if is_sse else "text/plain; charset=utf-8",
            headers=stream_headers,
        )

    # 3. Context Filtering & Prompt Construction (Rules D-2, D-3)
    context_text, sources = prune_and_format_context(req.query, candidates)
    messages = [
        {"role": "system", "content": STRICT_GROUNDED_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context passages:\n{context_text}\n\nQuestion: {req.query}\n\nProvide an attributed answer citing [Doc: filename, Page: page_number, Chunk: chunk_id].",
        },
    ]

    api_key = settings.openrouter_api_key

    async def token_stream() -> AsyncGenerator[bytes, None]:
        full_tokens: List[str] = []
        t_gen_start = perf_counter()

        if api_key:
            base_url = "https://openrouter.ai/api/v1"
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {"model": settings.llm_model, "messages": messages, "stream": True, "temperature": 0.1}

            try:
                async with httpx.AsyncClient(timeout=60.0, base_url=base_url) as client:
                    async with client.stream("POST", "/chat/completions", json=payload, headers=headers) as resp:
                        if resp.status_code >= 400:
                            err_txt = await resp.aread()
                            logger.error("OpenRouter streaming error {}: {}", resp.status_code, err_txt.decode())
                            fallback_ans = _local_synthesize_grounded(req.query, sources)
                            if is_sse:
                                yield f"data: {json.dumps({'token': fallback_ans})}\n\n".encode("utf-8")
                                yield b"data: [DONE]\n\n"
                            else:
                                yield fallback_ans.encode("utf-8")
                            full_tokens.append(fallback_ans)
                            return

                        async for line in resp.aiter_lines():
                            if not line or not line.startswith("data:"):
                                continue
                            data_chunk = line[5:].strip()
                            if data_chunk == "[DONE]":
                                break
                            try:
                                obj = json.loads(data_chunk)
                                delta = obj.get("choices", [{}])[0].get("delta", {})
                                content = delta.get("content")
                                if content:
                                    full_tokens.append(content)
                                    if is_sse:
                                        yield f"data: {json.dumps({'token': content})}\n\n".encode("utf-8")
                                    else:
                                        yield content.encode("utf-8")
                            except Exception:
                                continue
            except Exception as e:
                logger.warning("Streaming API error ({}); using local grounded synthesis", e)
                fallback_ans = _local_synthesize_grounded(req.query, sources)
                if is_sse:
                    yield f"data: {json.dumps({'token': fallback_ans})}\n\n".encode("utf-8")
                else:
                    yield fallback_ans.encode("utf-8")
                full_tokens.append(fallback_ans)
        else:
            # Local token streaming simulation for testing/offline environments
            fallback_ans = _local_synthesize_grounded(req.query, sources)
            words = fallback_ans.split(" ")
            for idx, word in enumerate(words):
                token = word if idx == len(words) - 1 else word + " "
                full_tokens.append(token)
                if is_sse:
                    yield f"data: {json.dumps({'token': token})}\n\n".encode("utf-8")
                else:
                    yield token.encode("utf-8")
                await asyncio.sleep(0.01)

        if is_sse:
            yield b"data: [DONE]\n\n"

        gen_ms = int((perf_counter() - t_gen_start) * 1000)
        tot_ms = int((perf_counter() - t_total_start) * 1000)
        complete_text = "".join(full_tokens)

        try:
            await execute(
                """
                INSERT INTO chat_history (session_id, user_message, assistant_message, source_documents,
                                          processing_time_ms, retrieval_ms, generation_ms)
                VALUES ($1, $2, $3, $4::jsonb, $5, $6, $7)
                """,
                req.session_id or "default",
                req.query,
                complete_text,
                json.dumps([s.model_dump() for s in sources]),
                tot_ms,
                retrieval_ms,
                gen_ms,
            )
        except Exception as e:
            logger.warning("Failed saving streamed chat history: {}", e)

    return StreamingResponse(
        token_stream(),
        media_type="text/event-stream" if is_sse else "text/plain; charset=utf-8",
        headers=stream_headers,
    )
