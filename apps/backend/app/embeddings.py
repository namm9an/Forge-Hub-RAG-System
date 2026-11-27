import asyncio
from collections import OrderedDict
from contextlib import ExitStack
from functools import partial
from hashlib import sha256
from itertools import chain
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
_reranking_limits = weakref.WeakKeyDictionary()
_query_batchers = weakref.WeakKeyDictionary()
_query_cache_lock = threading.Lock()
_query_cache_model = None
_query_embeddings = OrderedDict()
_QUERY_CACHE_SIZE = 1024


def _blocking_workers():
    return max(1, min(int(getattr(get_settings(), "embedding_workers", 2)), 8))


def _work_limit(limits, loop, capacity):
    reference = limits.get(loop)
    limiter = reference() if reference else None
    if limiter is None:
        limiter = asyncio.Semaphore(capacity)
        limits[loop] = weakref.ref(limiter)
    return limiter


async def _run_blocking(prepare, *, reranking=False):
    loop = asyncio.get_running_loop()
    rank_workers = getattr(get_settings(), "reranker_workers", 1)
    with ExitStack() as acquired:
        if reranking:
            lane = _work_limit(_reranking_limits, loop, rank_workers)
            await lane.acquire()
            acquired.callback(lane.release)
        limiter = _work_limit(_work_limits, loop, max(_blocking_workers(), rank_workers))
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
    """Wait for reranking capacity before reserving a shared CPU worker."""
    return await _run_blocking(lambda: partial(function, *args, **kwargs), reranking=True)


def get_model(settings) -> SentenceTransformer:
    """Synchronous compatibility API; production callers use get_model_async."""
    global _model
    with _model_lock:
        if _model is None:
            import torch

            torch.set_num_threads(1)
            if torch.get_num_interop_threads() != 1:
                torch.set_num_interop_threads(1)
            logger.info("Loading configured embedding model")
            _model = SentenceTransformer(settings.embedding_model)
    return _model


async def get_model_async(settings) -> SentenceTransformer:
    model = _model
    return model if model is not None else await run_blocking(get_model, settings)


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


def _index_text(content, heading):
    return f"{heading}\n{content}" if heading else content


class _QueryBatcher:
    """Pack waiting queries into bounded CPU calls without caching query data."""

    def __init__(self, model):
        self.model = model
        self.pending = OrderedDict()
        self.workers = set()

    async def encode(self, query):
        loop = asyncio.get_running_loop()
        result = loop.create_future()
        self.pending[result] = query
        if len(self.workers) < _blocking_workers():
            worker = loop.create_task(self.drain())
            self.workers.add(worker)
        try:
            return await result
        finally:
            self.pending.pop(result, None)

    async def drain(self):
        batch = []
        try:
            while self.pending:
                batch = []
                while self.pending and len(batch) < 8:
                    result, query = self.pending.popitem(last=False)
                    if not result.done():
                        batch.append((query, result))
                if batch:
                    await self.resolve(batch)
        except asyncio.CancelledError:
            for result in chain((result for _, result in batch), self.pending):
                result.cancel()
            self.pending.clear()
            raise
        finally:
            self.workers.discard(asyncio.current_task())

    async def resolve(self, batch):
        def prepare():
            # The batch can lose all its clients while it waits for a CPU slot.
            batch[:] = [(query, result) for query, result in batch if not result.done()]
            if batch:
                return partial(_encode_texts, self.model, [query for query, _ in batch])
            return None

        try:
            vectors = await _run_blocking(prepare)
        except Exception as error:
            if len(batch) > 1:
                # Preserve independent request outcomes when one input or the larger batch fails.
                for query, result in batch:
                    if not result.done():
                        await self.resolve([(query, result)])
            elif not batch[0][1].done():
                batch[0][1].set_exception(error)
            return
        if vectors is None:
            return
        for (_, result), vector in zip(batch, vectors):
            if not result.done():
                result.set_result(vector)


async def embed_query(model: SentenceTransformer, query: str, *, cache_owner: Optional[str] = None) -> List[float]:
    global _query_cache_model
    key = (cache_owner, sha256(query.encode()).digest())
    with _query_cache_lock:
        if _query_cache_model is not model:
            _query_embeddings.clear()
            _query_cache_model = model
        cached = _query_embeddings.get(key)
        if cached is not None:
            _query_embeddings.move_to_end(key)
            return list(cached)

    loop = asyncio.get_running_loop()
    batcher = _query_batchers.get(loop)
    if batcher is None or batcher.model is not model:
        batcher = _query_batchers[loop] = _QueryBatcher(model)
    vector = await batcher.encode(query)
    with _query_cache_lock:
        if _query_cache_model is model:
            _query_embeddings[key] = tuple(vector)
            _query_embeddings.move_to_end(key)
            if len(_query_embeddings) > _QUERY_CACHE_SIZE:
                _query_embeddings.popitem(last=False)
    return vector


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
            SELECT c.id, c.content, c.heading
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
                vectors = await _embed_texts(model, [_index_text(row["content"], row["heading"]) for row in batch])
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
