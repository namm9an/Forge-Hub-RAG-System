import asyncio
from time import perf_counter
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Query, Depends, HTTPException
from loguru import logger

from .config import get_settings, Settings
from .db import fetch, ensure_schema
from .embeddings import get_model, _as_pgvector
from .schemas import HybridSearchRequest, HybridSearchResponse, SearchResultItem

router = APIRouter(prefix="/api/search", tags=["search"])


async def _dense_search(
    query_vector_pg: str,
    session_id: Optional[str],
    candidate_k: int,
) -> List[Dict[str, Any]]:
    """
    Leg 1: Dense Vector Similarity Search using pgvector cosine distance.
    Evaluates 1 - (e.embedding <=> q_vector).
    """
    query = """
    SELECT c.id as chunk_id, c.content, c.page_number, c.document_id, d.filename,
           1 - (e.embedding <=> $1::vector) as dense_score
    FROM embeddings e
    JOIN chunks c ON c.id = e.chunk_id
    JOIN documents d ON d.id = c.document_id
    WHERE ($2::text IS NULL OR d.session_id = $2)
    ORDER BY e.embedding <=> $1::vector ASC
    LIMIT $3
    """
    return await fetch(query, query_vector_pg, session_id, candidate_k)


async def _sparse_search(
    query_text: str,
    session_id: Optional[str],
    candidate_k: int,
) -> List[Dict[str, Any]]:
    """
    Leg 2: Sparse Lexical Full-Text Search using PostgreSQL ts_rank_cd on GIN index.
    Upgraded to websearch_to_tsquery to allow natural language multi-term queries
    to match without strict boolean AND collapse.
    """
    query = """
    SELECT c.id as chunk_id, c.content, c.page_number, c.document_id, d.filename,
           ts_rank_cd(c.content_tsv, websearch_to_tsquery('english', $1)) as sparse_score
    FROM chunks c
    JOIN documents d ON d.id = c.document_id
    WHERE c.content_tsv @@ websearch_to_tsquery('english', $1)
      AND ($2::text IS NULL OR d.session_id = $2)
    ORDER BY sparse_score DESC
    LIMIT $3
    """
    return await fetch(query, query_text, session_id, candidate_k)


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


@router.post("/hybrid", response_model=HybridSearchResponse)
async def hybrid_search_post(
    req: HybridSearchRequest,
    settings: Settings = Depends(get_settings),
):
    """
    Production Hybrid Search using Reciprocal Rank Fusion (RRF).
    Complies with Rules C-1, C-2, C-3, C-4, and C-5.
    """
    await ensure_schema()
    t0 = perf_counter()

    # 1. Embed query vector
    model = get_model(settings)
    q_vec = model.encode([req.query], convert_to_numpy=True, normalize_embeddings=True)[0].tolist()
    q_vec_pg = _as_pgvector(q_vec)

    candidate_k = max(req.top_k * 3, 20)

    # 2. Parallel concurrent execution of Dense and Sparse retrieval (Rule C-4)
    dense_future = _dense_search(q_vec_pg, req.session_id, candidate_k)
    sparse_future = _sparse_search(req.query, req.session_id, candidate_k)

    dense_candidates, sparse_candidates = await asyncio.gather(dense_future, sparse_future)

    # 3. Apply Reciprocal Rank Fusion (Rule C-3)
    fused_results = compute_rrf_fusion(
        dense_results=dense_candidates,
        sparse_results=sparse_candidates,
        k=req.rrf_k,
        dense_weight=req.dense_weight,
        sparse_weight=req.sparse_weight,
        top_k=req.top_k,
    )

    elapsed_ms = int((perf_counter() - t0) * 1000)

    return HybridSearchResponse(
        query=req.query,
        total_results=len(fused_results),
        results=fused_results,
        retrieval_ms=elapsed_ms,
    )


@router.get("/hybrid", response_model=HybridSearchResponse)
async def hybrid_search_get(
    q: str = Query(..., min_length=1, description="Search query string"),
    session_id: Optional[str] = Query(None, description="Optional session ID"),
    top_k: int = Query(5, ge=1, le=50),
    settings: Settings = Depends(get_settings),
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
    return await hybrid_search_post(req, settings=settings)
