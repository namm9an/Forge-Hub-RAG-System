from fastapi import APIRouter, Query, Depends
from typing import List, Tuple
from .db import fetch
from .config import get_settings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

router = APIRouter(prefix="/api/search", tags=["search"]) 


def _overlap_score(query: str, text: str) -> float:
    q_tokens = set(query.lower().split())
    t_tokens = set(text.lower().split())
    if not q_tokens or not t_tokens:
        return 0.0
    return len(q_tokens & t_tokens) / max(1, len(q_tokens))


async def _fetch_corpus() -> List[Tuple[str, str]]:
    rows = await fetch("SELECT id, content FROM chunks")
    return [(str(r['id']), r['content']) for r in rows]


@router.get("/hybrid")
async def hybrid_search(
    q: str = Query(..., min_length=1),
    top_k: int = Query(5, ge=1, le=50),
    dense_weight: float = Query(0.4, ge=0.0, le=1.0),
    tfidf_weight: float = Query(0.3, ge=0.0, le=1.0),
    overlap_weight: float = Query(0.3, ge=0.0, le=1.0),
    settings=Depends(get_settings),
):
    # 1) Get corpus
    corpus = await _fetch_corpus()  # [(chunk_id, content)]
    if not corpus:
        return {"results": []}

    ids = [cid for cid, _ in corpus]
    texts = [t for _, t in corpus]

    # 2) TF-IDF + cosine
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts + [q])
    query_vec = tfidf_matrix[-1]
    doc_matrix = tfidf_matrix[:-1]
    sparse_scores = cosine_similarity(query_vec, doc_matrix)[0]  # shape: (N,)

    # 3) Overlap score
    overlap_scores = [ _overlap_score(q, t) for t in texts ]

    # 4) Dense score (optional quick proxy: reuse sparse for now since dense embedding of all chunks on each request is expensive)
    # In production, consider precomputing embeddings of all chunks to avoid real-time encoding.
    # Here we'll reuse sparse_scores as a proxy to keep endpoint responsive.
    dense_scores = sparse_scores

    # 5) Combine
    results = []
    for i, cid in enumerate(ids):
        score = dense_weight * float(dense_scores[i]) + tfidf_weight * float(sparse_scores[i]) + overlap_weight * float(overlap_scores[i])
        results.append({"chunk_id": cid, "content": texts[i], "score": score})
    results.sort(key=lambda x: x["score"], reverse=True)

    return {"results": results[:top_k]}
