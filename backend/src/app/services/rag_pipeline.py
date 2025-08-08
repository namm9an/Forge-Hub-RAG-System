from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import structlog

from app.config.settings import get_settings

# Third-party, available per pyproject
import asyncpg  # type: ignore
import httpx  # type: ignore
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder, SentenceTransformer  # type: ignore


logger = structlog.get_logger(__name__)


# -----------------------------
# Configuration and Constants
# -----------------------------
settings = get_settings()

ELASTIC_URL: Optional[str] = os.getenv("ELASTICSEARCH_URL") or os.getenv("ELASTIC_URL")
ELASTIC_INDEX: str = os.getenv("ELASTICSEARCH_INDEX", "document_chunks")
OPENAI_API_KEY: Optional[str] = settings.OPENAI_API_KEY
OPENAI_RERANK_MODEL: str = os.getenv("OPENAI_RERANK_MODEL", "gpt-4o-mini")
PG_DSN: Optional[str] = settings.DATABASE_URL

# Fusion weights (normalized later if some modalities are unavailable)
WEIGHT_ELASTIC = 0.40
WEIGHT_PGVECTOR = 0.30
WEIGHT_BM25 = 0.30

# Conversation memory TTL (seconds)
SESSION_TTL = 60 * 60

# Embedding model
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CROSS_ENCODER_MODEL = os.getenv(
    "CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# Retrieval parameters
DEFAULT_K = 12
RERANK_TOP_K = 10
FINAL_TOP_K = 6


# -----------------------------
# Utilities
# -----------------------------
class _TTLMemory:
    """Simple in-process TTL memory keyed by session UUID.

    Designed as an ephemeral, anonymous conversation memory without JWT.
    """

    def __init__(self, ttl_seconds: int) -> None:
        self._ttl = ttl_seconds
        self._store: Dict[str, Tuple[float, List[Dict[str, str]]]] = {}

    def _now(self) -> float:
        return time.time()

    def _cleanup(self) -> None:
        now = self._now()
        expired = [sid for sid, (ts, _) in self._store.items() if now - ts > self._ttl]
        for sid in expired:
            self._store.pop(sid, None)

    def record(self, session_id: str, role: str, content: str) -> None:
        self._cleanup()
        ts, hist = self._store.get(session_id, (self._now(), []))
        hist.append({"role": role, "content": content})
        self._store[session_id] = (self._now(), hist)

    def get(self, session_id: str) -> List[Dict[str, str]]:
        self._cleanup()
        item = self._store.get(session_id)
        if not item:
            return []
        ts, hist = item
        # bump timestamp on access
        self._store[session_id] = (self._now(), hist)
        return hist


_memory = _TTLMemory(SESSION_TTL)


class _DB:
    pool: Optional[asyncpg.Pool] = None

    @classmethod
    async def get_pool(cls) -> asyncpg.Pool:
        if cls.pool is None:
            if not PG_DSN:
                raise RuntimeError("DATABASE_URL is not configured")
            cls.pool = await asyncpg.create_pool(dsn=PG_DSN, min_size=1, max_size=10)
        return cls.pool


# Embedding model and cross-encoder are expensive to load; keep singletons
_embedding_model: Optional[SentenceTransformer] = None
_cross_encoder: Optional[CrossEncoder] = None


def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        logger.info("loading_embedding_model", model=EMBEDDING_MODEL_NAME)
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        logger.info("loading_cross_encoder", model=CROSS_ENCODER_MODEL)
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    return _cross_encoder


# -----------------------------
# Data models
# -----------------------------
@dataclass
class RetrievedHit:
    doc_id: str
    chunk_id: Optional[str]
    score: float
    content: str
    metadata: Dict[str, Any]
    source: str  # elastic|pgvector|bm25|fusion|rerank


# -----------------------------
# Embedding cache (embedding_cache table)
# -----------------------------
async def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


async def get_or_compute_embedding(text: str) -> List[float]:
    """Check embedding_cache by content_hash. If found, return cached vector; else compute and upsert.

    Returns an embedding as a Python list[float].
    """
    content_hash = await _sha256(text)
    pool = await _DB.get_pool()
    async with pool.acquire() as conn:
        # Try cache hit
        row = await conn.fetchrow(
            """
            SELECT embedding::text, id FROM public.embedding_cache WHERE content_hash=$1
            """,
            content_hash,
        )
        if row:
            # embedding is stored as vector type; fetch as text then parse
            emb_text: str = row["embedding::text"] if "embedding::text" in row else row[0]
            try:
                # Expect format like '[0.1, 0.2, ...]' or '(...)' depending on driver; normalize
                raw = emb_text.strip()
                if raw[0] in "([" and raw[-1] in ")]":
                    raw = raw[1:-1]
                vec = [float(x) for x in raw.split(",") if x.strip()]
                await conn.execute(
                    "UPDATE public.embedding_cache SET usage_count = usage_count + 1, last_used = TIMEZONE('utc'::text, NOW()) WHERE content_hash=$1",
                    content_hash,
                )
                logger.debug("embedding_cache_hit", hash=content_hash)
                return vec
            except Exception as e:  # fallback: recompute
                logger.warning("embedding_cache_parse_failed_recompute", error=str(e))
        # Miss: compute
        model = _get_embedding_model()
        vec = model.encode([text])[0].tolist()  # type: ignore[attr-defined]
        # Upsert
        await conn.execute(
            """
            INSERT INTO public.embedding_cache (content_hash, content_preview, embedding)
            VALUES ($1, $2, $3)
            ON CONFLICT (content_hash) DO UPDATE SET
                usage_count = public.embedding_cache.usage_count + 1,
                last_used = TIMEZONE('utc'::text, NOW())
            """,
            content_hash,
            text[:200],
            vec,
        )
        logger.debug("embedding_cache_miss_inserted", hash=content_hash)
        return vec


# -----------------------------
# Retrieval modalities
# -----------------------------
async def _search_elastic_knn(query: str, k: int) -> List[RetrievedHit]:
    if not ELASTIC_URL:
        return []
    emb = await get_or_compute_embedding(query)
    url = f"{ELASTIC_URL.rstrip('/')}/{ELASTIC_INDEX}/_search"
    body = {
        "size": k,
        "knn": {
            "field": "embedding",
            "query_vector": emb,
            "k": k,
            "num_candidates": max(k * 5, 50),
            "similarity": "cosine",
        },
        "_source": ["chunk_id", "document_id", "content", "metadata"],
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, json=body)
            resp.raise_for_status()
            data = resp.json()
            hits = []
            for h in data.get("hits", {}).get("hits", []):
                src = h.get("_source", {})
                score = float(h.get("_score", 0.0))
                hits.append(
                    RetrievedHit(
                        doc_id=str(src.get("document_id")),
                        chunk_id=str(src.get("chunk_id")) if src.get("chunk_id") else None,
                        score=score,
                        content=str(src.get("content", "")),
                        metadata=src.get("metadata", {}) or {},
                        source="elastic",
                    )
                )
            return hits
    except Exception as e:
        logger.warning("elastic_knn_failed", error=str(e))
        return []


async def _search_elastic_bm25(query: str, k: int) -> List[RetrievedHit]:
    if not ELASTIC_URL:
        return []
    url = f"{ELASTIC_URL.rstrip('/')}/{ELASTIC_INDEX}/_search"
    body = {
        "size": k,
        "query": {"match": {"content": {"query": query, "operator": "and"}}},
        "_source": ["chunk_id", "document_id", "content", "metadata"],
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, json=body)
            resp.raise_for_status()
            data = resp.json()
            hits = []
            for h in data.get("hits", {}).get("hits", []):
                src = h.get("_source", {})
                score = float(h.get("_score", 0.0))
                hits.append(
                    RetrievedHit(
                        doc_id=str(src.get("document_id")),
                        chunk_id=str(src.get("chunk_id")) if src.get("chunk_id") else None,
                        score=score,
                        content=str(src.get("content", "")),
                        metadata=src.get("metadata", {}) or {},
                        source="bm25",
                    )
                )
            return hits
    except Exception as e:
        logger.warning("elastic_bm25_failed", error=str(e))
        return []


async def _search_pgvector(query: str, k: int) -> List[RetrievedHit]:
    emb = await get_or_compute_embedding(query)
    pool = await _DB.get_pool()
    sql = (
        """
        SELECT dc.document_id::text, dc.id::text as chunk_id, dc.content, dc.metadata, 
               1 - (e.embedding <=> $1) as similarity
        FROM public.embeddings e
        JOIN public.document_chunks dc ON e.chunk_id = dc.id
        WHERE e.status = 'completed'
        ORDER BY e.embedding <-> $1
        LIMIT $2
        """
    )
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(sql, emb, k)
    except Exception as e:
        logger.warning("pgvector_search_failed", error=str(e))
        return []
    hits: List[RetrievedHit] = []
    for r in rows:
        hits.append(
            RetrievedHit(
                doc_id=str(r["document_id"]),
                chunk_id=str(r["chunk_id"]),
                score=float(r["similarity"] or 0.0),
                content=str(r["content"] or ""),
                metadata=r["metadata"] or {},
                source="pgvector",
            )
        )
    return hits


async def _search_postgres_fulltext(query: str, k: int) -> List[RetrievedHit]:
    """Fallback BM25-like using Postgres full-text search with ts_rank."""
    pool = await _DB.get_pool()
    sql = (
        """
        SELECT dc.document_id::text, dc.id::text as chunk_id, dc.content, dc.metadata,
               ts_rank_cd(to_tsvector('english', dc.content), plainto_tsquery('english', $1)) AS rank
        FROM public.document_chunks dc
        WHERE to_tsvector('english', dc.content) @@ plainto_tsquery('english', $1)
        ORDER BY rank DESC
        LIMIT $2
        """
    )
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(sql, query, k)
    except Exception as e:
        logger.warning("postgres_fulltext_failed", error=str(e))
        return []
    hits: List[RetrievedHit] = []
    for r in rows:
        hits.append(
            RetrievedHit(
                doc_id=str(r["document_id"]),
                chunk_id=str(r["chunk_id"]),
                score=float(r["rank"] or 0.0),
                content=str(r["content"] or ""),
                metadata=r["metadata"] or {},
                source="bm25",
            )
        )
    return hits


def _fuse_rankings(
    lists: List[Tuple[str, List[RetrievedHit]]],
    weights: Dict[str, float],
    top_k: int,
) -> List[RetrievedHit]:
    """Reciprocal rank fusion with weighted scores.

    Score(item) = sum_i weight_i / (k + rank_i)
    """
    k_const = 60.0
    # Build rank maps
    rank_maps: Dict[str, Dict[str, int]] = {}
    id_to_hit: Dict[str, RetrievedHit] = {}
    for name, hits in lists:
        rank_map: Dict[str, int] = {}
        for idx, h in enumerate(hits):
            key = f"{h.doc_id}:{h.chunk_id or ''}"
            id_to_hit[key] = h
            rank_map[key] = idx + 1
        rank_maps[name] = rank_map

    fused_scores: Dict[str, float] = {}
    for name, rank_map in rank_maps.items():
        w = weights.get(name, 0.0)
        for key, rank in rank_map.items():
            fused_scores[key] = fused_scores.get(key, 0.0) + w / (k_const + rank)

    # Build results sorted by fused score desc
    sorted_items = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    results: List[RetrievedHit] = []
    for key, score in sorted_items[:top_k * 2]:  # keep buffer for reranking
        base = id_to_hit[key]
        results.append(
            RetrievedHit(
                doc_id=base.doc_id,
                chunk_id=base.chunk_id,
                score=score,
                content=base.content,
                metadata={**(base.metadata or {}), "fused_score": score},
                source="fusion",
            )
        )
    return results[: top_k * 2]


async def _crossencoder_rerank(query: str, hits: List[RetrievedHit], top_k: int) -> List[RetrievedHit]:
    if not hits:
        return []
    model = _get_cross_encoder()
    pairs = [(query, h.content) for h in hits]
    try:
        scores = model.predict(pairs).tolist()  # type: ignore[attr-defined]
    except Exception as e:
        logger.warning("crossencoder_predict_failed", error=str(e))
        return hits[:top_k]
    rescored = []
    for h, s in zip(hits, scores):
        rescored.append(
            RetrievedHit(
                doc_id=h.doc_id,
                chunk_id=h.chunk_id,
                score=float(s),
                content=h.content,
                metadata={**(h.metadata or {}), "crossencoder_score": float(s)},
                source="rerank",
            )
        )
    rescored.sort(key=lambda x: x.score, reverse=True)
    return rescored[:top_k]


async def _openai_rerank(query: str, hits: List[RetrievedHit], top_k: int) -> List[RetrievedHit]:
    if not OPENAI_API_KEY or not hits:
        return hits[:top_k]
    # Lightweight scoring prompt to avoid large costs
    # We ask the model to output JSON array of floats [0..1] of same length as inputs
    system = "You are a ranking model. Score relevance between 0 and 1 for each candidate passage to the user query. Respond ONLY with a JSON array of numbers."
    passages = [h.content[:1000] for h in hits]
    user = json.dumps({"query": query, "passages": passages})
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": OPENAI_RERANK_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
    }
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"].strip()
            scores = json.loads(content)
            if not isinstance(scores, list):
                raise ValueError("Invalid score format")
            rescored = []
            for h, s in zip(hits, scores):
                v = float(s)
                rescored.append(
                    RetrievedHit(
                        doc_id=h.doc_id,
                        chunk_id=h.chunk_id,
                        score=v,
                        content=h.content,
                        metadata={**(h.metadata or {}), "openai_rerank": v},
                        source="rerank",
                    )
                )
            rescored.sort(key=lambda x: x.score, reverse=True)
            return rescored[:top_k]
    except Exception as e:
        logger.warning("openai_rerank_failed", error=str(e))
        return hits[:top_k]


# -----------------------------
# Public API
# -----------------------------
async def retrieve_documents(query: str, k: int = DEFAULT_K) -> List[Document]:
    """Hybrid retrieval with weighted fusion and multi-stage reranking.

    1) Retrieve via three modalities (elastic cosine, pgvector ANN, keyword BM25)
    2) Fuse with 40/30/30 weighted reciprocal rank fusion
    3) Re-rank with CrossEncoder, then optional OpenAI rerank
    Returns LangChain Document objects.
    """
    # Run modalities concurrently
    tasks: List[asyncio.Task] = []
    tasks.append(asyncio.create_task(_search_elastic_knn(query, k)))
    tasks.append(asyncio.create_task(_search_pgvector(query, k)))
    # BM25: try elastic, fallback to postgres fulltext
    if ELASTIC_URL:
        tasks.append(asyncio.create_task(_search_elastic_bm25(query, k)))
    else:
        tasks.append(asyncio.create_task(_search_postgres_fulltext(query, k)))

    elastic_hits, pg_hits, bm25_hits = await asyncio.gather(*tasks)

    # Normalize weights based on availability
    available = {
        "elastic": WEIGHT_ELASTIC if elastic_hits else 0.0,
        "pgvector": WEIGHT_PGVECTOR if pg_hits else 0.0,
        "bm25": WEIGHT_BM25 if bm25_hits else 0.0,
    }
    total_w = sum(available.values()) or 1.0
    weights = {k: v / total_w for k, v in available.items()}

    fused = _fuse_rankings(
        [("elastic", elastic_hits), ("pgvector", pg_hits), ("bm25", bm25_hits)],
        weights=weights,
        top_k=k,
    )

    # Stage 1 rerank: CrossEncoder
    stage1 = await _crossencoder_rerank(query, fused, top_k=RERANK_TOP_K)
    # Stage 2 rerank: OpenAI
    stage2 = await _openai_rerank(query, stage1, top_k=FINAL_TOP_K)

    docs: List[Document] = []
    for h in stage2[:FINAL_TOP_K]:
        metadata = {**(h.metadata or {}), "source": h.source, "score": h.score, "doc_id": h.doc_id, "chunk_id": h.chunk_id}
        docs.append(Document(page_content=h.content, metadata=metadata))
    return docs


async def generate_answer(
    question: str,
    session_id: Optional[str] = None,
    k: int = DEFAULT_K,
    llm: Optional[Any] = None,
) -> Dict[str, Any]:
    """Generate an answer using retrieved documents and lightweight conversation memory.

    Parameters:
      - question: user question
      - session_id: UUID string identifying anonymous session (1-hour TTL). If provided,
        prior turns are included as context and updated via record_history.
      - k: retrieval depth
      - llm: optional LangChain-compatible LLM (e.g., ChatOpenAI). If None, returns
        retrieved contexts without LLM generation to allow caller to plug its own chain.

    Returns JSON-serializable dict with fields: answer, context, citations, session_id
    """
    history = _memory.get(session_id) if session_id else []

    docs = await retrieve_documents(question, k=k)

    # Build a simple prompt if LLM provided
    answer_text = ""
    if llm is not None:
        context_snippets = "\n\n".join(
            [f"[Doc {i+1}] {d.page_content}" for i, d in enumerate(docs)]
        )
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in history]) if history else ""
        prompt = (
            "You are a helpful assistant. Use the provided context to answer the question.\n"
            "Cite sources using [Doc N] when relevant. If unsure, say you don't know.\n\n"
            f"Conversation History:\n{history_text}\n\n"
            f"Context:\n{context_snippets}\n\n"
            f"Question: {question}\nAnswer:"
        )
        try:
            # LangChain interface: llm.invoke or llm.predict
            if hasattr(llm, "invoke"):
                answer_text = llm.invoke(prompt)  # type: ignore
                if isinstance(answer_text, dict) and "content" in answer_text:
                    answer_text = answer_text["content"]
                elif not isinstance(answer_text, str):
                    answer_text = str(answer_text)
            elif hasattr(llm, "predict"):
                answer_text = llm.predict(prompt)  # type: ignore
            else:
                answer_text = ""
        except Exception as e:
            logger.warning("llm_invoke_failed", error=str(e))
            answer_text = ""

    citations = [d.metadata.get("doc_id") for d in docs]

    # Record the turn if session provided
    if session_id:
        record_history(session_id, role="user", content=question)
        if answer_text:
            record_history(session_id, role="assistant", content=answer_text)

    return {
        "answer": answer_text,
        "context": [
            {
                "content": d.page_content,
                "metadata": d.metadata,
            }
            for d in docs
        ],
        "citations": citations,
        "session_id": session_id,
    }


def record_history(session_id: str, role: str, content: str) -> None:
    """Record a conversation turn into in-memory 1-hour TTL session store.

    session_id should be a UUID string generated by the caller. This store is
    anonymous and JWT-less per requirements.
    """
    if not session_id:
        return
    _memory.record(session_id, role=role, content=content)


__all__ = [
    "retrieve_documents",
    "generate_answer",
    "record_history",
]

