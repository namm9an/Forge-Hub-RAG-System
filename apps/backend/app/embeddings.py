import asyncio
from contextlib import ExitStack
from functools import partial
import threading
from typing import List, Optional
import weakref

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from loguru import logger
import numpy as np
from sentence_transformers import SentenceTransformer

from .auth import Principal, require_user
from .config import get_settings, Settings
from .db import execute, fetch, transaction
from .sessions import authorize_session

router = APIRouter(prefix="/api/embeddings", tags=["embeddings"])
_model: Optional[SentenceTransformer] = None
_model_lock = threading.Lock()
_work_limits = weakref.WeakKeyDictionary()


def _blocking_workers():
    return max(1, min(int(getattr(get_settings(), "embedding_workers", 2)), 8))


def _work_limit(limits, loop, capacity):
    reference = limits.get(loop)
    limiter = reference() if reference else None
    if limiter is None:
        limiter = asyncio.Semaphore(capacity)
        limits[loop] = weakref.ref(limiter)
    return limiter


async def _run_blocking(prepare):
    loop = asyncio.get_running_loop()
    with ExitStack() as acquired:
        limiter = _work_limit(_work_limits, loop, _blocking_workers())
        await limiter.acquire()
        acquired.callback(limiter.release)
        work = prepare()
        if work is None:
            return None
        future = loop.run_in_executor(None, work)
        held = acquired.pop_all()

    def finished(result):
        held.close()
        if not result.cancelled():
            result.exception()  # Observe failures even after a request disconnects.

    future.add_done_callback(finished)
    return await asyncio.shield(future)


async def run_blocking(function, *args, **kwargs):
    """Bound blocking work, retaining its slot if the awaiting request is cancelled."""
    return await _run_blocking(lambda: partial(function, *args, **kwargs))


async def run_reranking(function, *args, **kwargs):
    return await run_blocking(function, *args, **kwargs)


def get_model(settings) -> SentenceTransformer:
    """Synchronous compatibility API; production callers use get_model_async."""
    global _model
    with _model_lock:
        if _model is None:
            logger.info("Loading configured embedding model")
            _model = SentenceTransformer(settings.embedding_model)
    return _model


async def get_model_async(settings) -> SentenceTransformer:
    return await run_blocking(get_model, settings)


def _as_pgvector(vec: List[float]) -> str:
    values = np.asarray(vec, dtype=np.float64)
    if values.shape != (768,) or not np.isfinite(values).all():
        raise ValueError("Expected one finite 768-dimensional vector")
    return "[" + ",".join(f"{v:.8f}" for v in values) + "]"


def _encode_texts(model: SentenceTransformer, texts: List[str], batch_size: int = 16) -> List[List[float]]:
    output = model.encode(
        texts, batch_size=batch_size, convert_to_numpy=True,
        show_progress_bar=False, normalize_embeddings=True,
    )
    vectors = np.asarray(output, dtype=np.float64)
    if vectors.shape != (len(texts), 768) or not np.isfinite(vectors).all():
        raise ValueError("Embedding model returned invalid vector dimensions or values")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4, rtol=1e-4):
        raise ValueError("Embedding model must return normalized nonzero vectors")
    return vectors.tolist()


async def _embed_texts(model: SentenceTransformer, texts: List[str], batch_size: int = 16) -> List[List[float]]:
    return await run_blocking(_encode_texts, model, texts, batch_size) if texts else []


async def embed_query(model: SentenceTransformer, query: str) -> List[float]:
    return (await _embed_texts(model, [query]))[0]


@router.post("/generate")
async def generate_all_missing(
    session_id: Optional[str] = Query(None),
    x_session_id: Optional[str] = Header(None),
    settings: Settings = Depends(get_settings),
    principal: Principal = Depends(require_user),
):
    identifiers = [value for value in (session_id, x_session_id) if value is not None]
    if not identifiers or any(not value.strip() for value in identifiers):
        raise HTTPException(status_code=422, detail="An explicit session_id is required")
    if len(set(identifiers)) != 1:
        raise HTTPException(status_code=400, detail="Conflicting session identifiers")
    sid = identifiers[0]
    async with transaction():
        await authorize_session(sid, principal.user_id)
        rows = await fetch(
            """
            SELECT c.id, c.content
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            LEFT JOIN embeddings e ON e.chunk_id = c.id
            WHERE e.id IS NULL AND d.session_id = $1
            ORDER BY c.created_at ASC
            LIMIT 2000
            """,
            sid,
        )
        if not rows:
            return {"status": "ok", "generated": 0, "session_id": sid}
        try:
            model = await get_model_async(settings)
        except Exception as exc:
            logger.warning("Scoped embedding generation failed: {}", type(exc).__name__)
            raise HTTPException(status_code=503, detail="Embedding generation is unavailable") from exc
        for start in range(0, len(rows), 16):
            batch = rows[start:start + 16]
            try:
                vectors = await _embed_texts(model, [row["content"] for row in batch])
            except Exception as exc:
                logger.warning("Scoped embedding generation failed: {}", type(exc).__name__)
                raise HTTPException(status_code=503, detail="Embedding generation is unavailable") from exc
            for row, vector in zip(batch, vectors):
                await execute(
                    """
                    INSERT INTO embeddings (chunk_id, embedding, model_version)
                    VALUES ($1, $2::vector, $3)
                    ON CONFLICT (chunk_id) DO NOTHING
                    """,
                    row["id"], _as_pgvector(vector), settings.embedding_model,
                )
    return {"status": "ok", "generated": len(rows), "session_id": sid}
