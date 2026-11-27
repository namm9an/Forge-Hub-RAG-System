import asyncio
import re
from time import perf_counter
from typing import List, Dict, Any, Optional
from uuid import UUID
from fastapi import APIRouter, Query, Depends, HTTPException

from .config import get_settings, Settings
from .auth import Principal, require_user
from .db import fetch, get_pool
from .embeddings import get_model_async, embed_query, _as_pgvector
from .schemas import HybridSearchRequest, HybridSearchResponse, SearchResultItem
from .sessions import authorize_session

router = APIRouter(prefix="/api/search", tags=["search"])


async def _dense_search(
    query_vector_pg: str,
    session_id: str,
    candidate_k: int,
    *,
    user_id: str,
) -> List[Dict[str, Any]]:
    """
    Leg 1: Dense Vector Similarity Search using pgvector cosine distance.
    Evaluates 1 - (e.embedding <=> q_vector).
    """
    if not session_id or not user_id:
        raise ValueError("Dense retrieval requires an owned session")
    query = """
    SELECT c.id as chunk_id, c.content, c.page_number, c.document_id, d.filename,
           1 - (e.embedding <=> $1::vector) as dense_score
    FROM embeddings e
    JOIN chunks c ON c.id = e.chunk_id
    JOIN documents d ON d.id = c.document_id
    JOIN sessions s ON s.session_id = d.session_id
    WHERE d.session_id = $2 AND s.user_id = $4
    ORDER BY e.embedding <=> $1::vector ASC
    LIMIT $3
    """
    return await fetch(query, query_vector_pg, session_id, candidate_k, user_id)


async def _sparse_search(
    query_text: str,
    session_id: str,
    candidate_k: int,
    *,
    user_id: str,
) -> List[Dict[str, Any]]:
    """
    Leg 2: Sparse Lexical Full-Text Search using PostgreSQL ts_rank_cd on GIN index.
    Retry empty ordinary questions with English-normalized OR terms. Explicit
    web-search operators retain their native semantics without relaxation.
    """
    if not session_id or not user_id:
        raise ValueError("Sparse retrieval requires an owned session")
    if len(query_text) > 4000:
        raise ValueError("Sparse queries cannot exceed 4000 characters")
    query = """
    SELECT c.id as chunk_id, c.content, c.page_number, c.document_id, d.filename,
           ts_rank_cd(c.content_tsv, websearch_to_tsquery('english', $1)) as sparse_score
    FROM chunks c
    JOIN documents d ON d.id = c.document_id
    JOIN sessions s ON s.session_id = d.session_id
    WHERE c.content_tsv @@ websearch_to_tsquery('english', $1)
      AND d.session_id = $2 AND s.user_id = $4
    ORDER BY sparse_score DESC
    LIMIT $3
    """
    rows = await fetch(query, query_text, session_id, candidate_k, user_id)
    if rows or '"' in query_text or re.search(r"\bor\b|(?<!\w)-\S", query_text, re.IGNORECASE):
        return rows

    # Normalize/stem with the same English dictionary as content_tsv. Values
    # remain bound; quote every normalized lexeme before assembling tsquery.
    # No meaningful lexemes produces NULL, which matches no rows.
    fallback = """
    WITH fallback_query AS (
        SELECT string_agg(quote_literal(lexeme), ' | ')::tsquery AS terms
        FROM unnest(tsvector_to_array(to_tsvector('english', $1))) AS normalized(lexeme)
    )
    SELECT c.id as chunk_id, c.content, c.page_number, c.document_id, d.filename,
           ts_rank_cd(c.content_tsv, f.terms) as sparse_score
    FROM chunks c
    JOIN documents d ON d.id = c.document_id
    JOIN sessions s ON s.session_id = d.session_id
    CROSS JOIN fallback_query f
    WHERE c.content_tsv @@ f.terms
      AND d.session_id = $2 AND s.user_id = $4
    ORDER BY sparse_score DESC
    LIMIT $3
    """
    return await fetch(fallback, query_text, session_id, candidate_k, user_id)


def compute_rrf_fusion(
    dense_results: List[Dict[str, Any]],
    sparse_results: List[Dict[str, Any]],
    k: int = 60,
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4,
    top_k: int = 5,
) -> List[SearchResultItem]:
    """
    Mathematical Reciprocal Rank Fusion (RRF) (Rule C-3):
    RRF_Score(d) = (dense_weight / (k + dense_rank)) + (sparse_weight / (k + sparse_rank))
    """
    fused_scores: Dict[str, float] = {}
    dense_ranks: Dict[str, int] = {}
    sparse_ranks: Dict[str, int] = {}
    dense_scores: Dict[str, float] = {}
    sparse_scores: Dict[str, float] = {}
    chunk_metadata: Dict[str, Dict[str, Any]] = {}

    # 1. Rank Dense Results (1-indexed)
    for rank, item in enumerate(dense_results, start=1):
        cid = str(item["chunk_id"])
        dense_ranks[cid] = rank
        dense_scores[cid] = float(item.get("dense_score", 0.0))
        chunk_metadata[cid] = item
        fused_scores[cid] = fused_scores.get(cid, 0.0) + (dense_weight / (k + rank))

    # 2. Rank Sparse Results (1-indexed)
    for rank, item in enumerate(sparse_results, start=1):
        cid = str(item["chunk_id"])
        sparse_ranks[cid] = rank
        sparse_scores[cid] = float(item.get("sparse_score", 0.0))
        if cid not in chunk_metadata:
            chunk_metadata[cid] = item
        fused_scores[cid] = fused_scores.get(cid, 0.0) + (sparse_weight / (k + rank))

    # 3. Sort Candidates by Fused RRF Score Descending
    sorted_chunks = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

    results: List[SearchResultItem] = []
    for cid, rrf_score in sorted_chunks[:top_k]:
        meta = chunk_metadata[cid]
        results.append(
            SearchResultItem(
                chunk_id=cid,
                document_id=str(meta.get("document_id", "")),
                filename=str(meta.get("filename", "unknown.pdf")),
                page_number=int(meta.get("page_number", 1)),
                score=round(rrf_score, 6),
                dense_score=round(dense_scores.get(cid, 0.0), 4) if cid in dense_scores else None,
                sparse_score=round(sparse_scores.get(cid, 0.0), 4) if cid in sparse_scores else None,
                dense_rank=dense_ranks.get(cid),
                sparse_rank=sparse_ranks.get(cid),
                content=str(meta.get("content", "")),
            )
        )

    return results


async def retrieve_candidates(
    query: str, session_id: str, settings: Settings, candidate_k: int, *, user_id: str,
    dense_weight: Optional[float] = None, sparse_weight: Optional[float] = None,
    rrf_k: Optional[int] = None,
) -> List[SearchResultItem]:
    """One owner-scoped retrieval path shared by search and chat."""
    model = await get_model_async(settings)
    query_vector = _as_pgvector(await embed_query(model, query, cache_owner=user_id))
    dense, sparse = await asyncio.gather(
        _dense_search(query_vector, session_id, candidate_k, user_id=user_id),
        _sparse_search(query, session_id, candidate_k, user_id=user_id),
    )
    return compute_rrf_fusion(
        dense, sparse, k=rrf_k if rrf_k is not None else settings.rrf_k,
        dense_weight=dense_weight if dense_weight is not None else settings.dense_weight,
        sparse_weight=sparse_weight if sparse_weight is not None else settings.sparse_weight,
        top_k=candidate_k,
    )


async def _semantic_duplicate_pairs(chunk_ids, session_id: str, user_id: str):
    # The memory adapter has no semantic-model contract, including fixed test encoders.
    if await get_pool() is None:
        return set()
    rows = await fetch(
        """
        WITH owned_candidates AS MATERIALIZED (
            SELECT e.chunk_id, e.embedding, e.model_version
            FROM embeddings e
            JOIN chunks c ON c.id = e.chunk_id
            JOIN documents d ON d.id = c.document_id
            JOIN sessions s ON s.session_id = d.session_id
            WHERE c.id = ANY($1::uuid[]) AND d.session_id = $2 AND s.user_id = $3
        )
        SELECT a.chunk_id AS left_id, b.chunk_id AS right_id
        FROM owned_candidates a
        JOIN owned_candidates b ON a.chunk_id < b.chunk_id AND a.model_version = b.model_version
        WHERE 1 - (a.embedding <=> b.embedding) >= $4::double precision
        """,
        [UUID(chunk_id) for chunk_id in chunk_ids], session_id, user_id, 0.97,
    )
    return {frozenset((str(row["left_id"]), str(row["right_id"]))) for row in rows}


async def deduplicate_ranked_candidates(candidates, *, top_k: int, session_id: str, user_id: str):
    """Keep the best-ranked representative and fill remaining slots with distinct parents."""
    if top_k == 1 or len(candidates) < 2:
        return candidates[:top_k]
    chunk_ids = [str(item["chunk_id"] if isinstance(item, dict) else item.chunk_id) for item in candidates]
    semantic_pairs = await _semantic_duplicate_pairs(chunk_ids, session_id, user_id)
    kept = []
    for item, chunk_id in zip(candidates, chunk_ids):
        content = item["content"] if isinstance(item, dict) else item.content
        text = " ".join(content.casefold().split())
        words = re.findall(r"\w+", text)
        shingles = {tuple(words[start:start + 5]) for start in range(len(words) - 4)}
        if any(
            (text and text == other_text)
            or frozenset((chunk_id, other_id)) in semantic_pairs
            or (shingles and other_shingles and len(shingles & other_shingles) / len(shingles | other_shingles) >= 0.8)
            for _, other_id, other_text, other_shingles in kept
        ):
            continue
        kept.append((item, chunk_id, text, shingles))
        if len(kept) == top_k:
            break
    return [item for item, _, _, _ in kept]


@router.post("/hybrid", response_model=HybridSearchResponse)
async def hybrid_search_post(
    req: HybridSearchRequest,
    settings: Settings = Depends(get_settings),
    principal: Principal = Depends(require_user),
):
    """
    Production Hybrid Search using Reciprocal Rank Fusion (RRF).
    Complies with Rules C-1, C-2, C-3, C-4, and C-5.
    """
    await authorize_session(req.session_id, principal.user_id)
    if len(req.query) > settings.max_query_chars:
        raise HTTPException(status_code=422, detail="Query exceeds the configured limit")
    if req.dense_weight + req.sparse_weight <= 0:
        raise HTTPException(status_code=422, detail="At least one retrieval weight must be positive")
    t0 = perf_counter()
    candidate_k = max(req.top_k * 3, 20)
    fused_results = (await retrieve_candidates(
        req.query, req.session_id, settings, candidate_k, user_id=principal.user_id,
        dense_weight=req.dense_weight, sparse_weight=req.sparse_weight, rrf_k=req.rrf_k,
    ))[:req.top_k]

    elapsed_ms = int((perf_counter() - t0) * 1000)

    return HybridSearchResponse(
        query=req.query,
        total_results=len(fused_results),
        results=fused_results,
        retrieval_ms=elapsed_ms,
    )


@router.get("/hybrid", response_model=HybridSearchResponse)
async def hybrid_search_get(
    q: str = Query(..., min_length=1, max_length=4000, pattern=r"\S", description="Search query string"),
    session_id: str = Query(..., min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$", description="Owned session ID"),
    top_k: int = Query(5, ge=1, le=50),
    settings: Settings = Depends(get_settings),
    principal: Principal = Depends(require_user),
):
    """Convenience GET endpoint wrapping the hybrid search POST contract."""
    req = HybridSearchRequest(
        query=q,
        session_id=session_id,
        top_k=top_k,
        dense_weight=settings.dense_weight,
        sparse_weight=settings.sparse_weight,
        rrf_k=settings.rrf_k,
    )
    return await hybrid_search_post(req, settings=settings, principal=principal)
