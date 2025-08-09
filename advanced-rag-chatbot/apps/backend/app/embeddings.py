from fastapi import APIRouter, HTTPException, Depends
from typing import List
from loguru import logger
from sentence_transformers import SentenceTransformer
import numpy as np
from .config import get_settings
from .db import fetch, execute, ensure_schema

router = APIRouter(prefix="/api/embeddings", tags=["embeddings"]) 

_model: SentenceTransformer | None = None


def get_model(settings) -> SentenceTransformer:
    global _model
    if _model is None:
        model_name = getattr(settings, "EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
        logger.info("Loading embedding model: {}", model_name)
        _model = SentenceTransformer(model_name)
    return _model


def _as_pgvector(vec: List[float]) -> str:
    return "[" + ",".join(f"{v:.8f}" for v in vec) + "]"


async def _embed_texts(model: SentenceTransformer, texts: List[str], batch_size: int = 16) -> List[List[float]]:
    if not texts:
        return []
    emb = model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
    return emb.tolist()


@router.post("/generate")
async def generate_all_missing(settings=Depends(get_settings)):
    await ensure_schema()
    model = get_model(settings)

    # Find chunks without embeddings
    rows = await fetch(
        """
        SELECT c.id, c.content
        FROM chunks c
        LEFT JOIN embeddings e ON e.chunk_id = c.id
        WHERE e.id IS NULL
        ORDER BY c.created_at ASC
        LIMIT 2000
        """
    )

    if not rows:
        return {"status": "ok", "generated": 0}

    texts = [r["content"] for r in rows]
    ids = [r["id"] for r in rows]

    vectors = await _embed_texts(model, texts, batch_size=int(getattr(settings, "BATCH_SIZE", 16)))

    # Insert embeddings
    created = 0
    for chunk_id, vec in zip(ids, vectors):
        await execute(
            """
            INSERT INTO embeddings (chunk_id, embedding, model_version)
            VALUES ($1, $2::vector, $3)
            ON CONFLICT (chunk_id) DO NOTHING
            """,
            chunk_id,
            _as_pgvector(vec),
            getattr(settings, "EMBEDDING_MODEL", "all-mpnet-base-v2"),
        )
        created += 1

    return {"status": "ok", "generated": created}
