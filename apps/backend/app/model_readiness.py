"""Prove local inference works before advertising the RAG service as ready."""
from .embeddings import get_model_async, _embed_texts, run_reranking
from .gating import get_reranker


async def warmup_models(settings):
    model = await get_model_async(settings)
    if getattr(model, "tokenizer", None) is None:
        raise RuntimeError("Embedding tokenizer is unavailable")
    await _embed_texts(model, ["Backend readiness verification."])
    reranker = get_reranker(settings.reranker_model)
    await run_reranking(reranker.rerank, query="readiness", candidates=[
        {"content": "Backend readiness verification.", "dense_score": 1.0, "sparse_score": 0.0},
    ])
    if reranker.backend != "flashrank":
        raise RuntimeError("Neural reranker is unavailable")
