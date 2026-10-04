"""
Forge Hub RAG System — Evidence Gating & Noise Pruning Engine
Complies with:
- Rule D-1: Out-of-Domain Refusal Gate (Evidence Confidence Evaluator)
- Rule D-2: Sentence-Level Noise Pruner (Context Filtering)
- Stage 2: Quantized Cross-Encoder Reranking (FlashRank with semantic fallback)
"""

import re
from typing import List, Dict, Any, Tuple, Optional, Union
from loguru import logger
from .schemas import ConfidenceLevel, SearchResultItem, Source

REFUSAL_MESSAGE = "The provided documents do not contain sufficient context to answer this question."


def split_into_sentences(text: str) -> List[str]:
    """
    Splits chunk text into sentences preserving complete sentence integrity.
    Uses regex for sentence boundary detection on punctuation (.!?).
    """
    if not text or not text.strip():
        return []
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in raw if s.strip()]
    return sentences if sentences else [text.strip()]


def score_sentence_relevance(sentence: str, query: str) -> float:
    """
    Scores lexical and term coverage between an individual sentence and the query.
    Computes Jaccard/overlap score weighted by word length and query coverage.
    """
    s_tokens = set(re.findall(r"\w+", sentence.lower()))
    q_tokens = set(re.findall(r"\w+", query.lower()))

    # Remove generic English stop words for sharper discrimination
    stop_words = {
        "a", "an", "the", "in", "on", "of", "to", "for", "with", "at", "by", "from",
        "and", "or", "is", "are", "was", "were", "be", "been", "this", "that", "it",
        "what", "how", "why", "when", "where", "which", "who", "does", "do", "did"
    }
    q_content = q_tokens - stop_words
    if not q_content:
        q_content = q_tokens

    if not q_content or not s_tokens:
        return 0.0

    overlap = q_content.intersection(s_tokens)
    if not overlap:
        return 0.0

    # Overlap ratio relative to content query terms
    coverage = len(overlap) / len(q_content)
    # Density in sentence
    density = len(overlap) / len(s_tokens)
    return round((0.7 * coverage) + (0.3 * density), 4)


def prune_chunk_sentences(
    content: str,
    query: str,
    min_sentence_relevance: float = 0.10,
) -> Tuple[str, int, int]:
    """
    Rule D-2: Sentence-Level Noise Pruner.
    Splits retrieved chunk into sentences, scores relevance against query,
    and prunes irrelevant sentences to reduce context noise.
    Returns: (pruned_text, original_count, retained_count)
    """
    sentences = split_into_sentences(content)
    if len(sentences) <= 1:
        return content, len(sentences), len(sentences)

    scored_sentences = [
        (s, score_sentence_relevance(s, query))
        for s in sentences
    ]

    # Retain sentences meeting threshold
    retained = [s for s, score in scored_sentences if score >= min_sentence_relevance]

    # Guardrail: if all sentences scored below threshold, keep the top 1 most relevant
    if not retained:
        best_sentence = max(scored_sentences, key=lambda x: x[1])[0]
        retained = [best_sentence]

    pruned_text = " ".join(retained)
    return pruned_text, len(sentences), len(retained)


class CrossEncoderReranker:
    """
    Stage 2 Neural Cross-Encoder Reranker using FlashRank (quantized model)
    with graceful fallback to high-fidelity semantic scoring when FlashRank
    package is not installed.
    Supports both SearchResultItem instances and dictionary representations.
    """
    def __init__(self, model_name: str = "ms-marco-TinyBERT-L-2-v2"):
        self.model_name = model_name
        self._ranker = None
        self._flashrank_checked = False

    def _get_ranker(self):
        if not self._flashrank_checked:
            self._flashrank_checked = True
            try:
                from flashrank import Ranker
                self._ranker = Ranker(model_name=self.model_name)
                logger.info("Initialized FlashRank cross-encoder with model {}", self.model_name)
            except Exception as e:
                logger.info("FlashRank not available ({}); fallback semantic scoring active.", e)
                self._ranker = None
        return self._ranker

    def _compute_fallback_score(self, item: Any, query: str) -> float:
        content = getattr(item, "content", "") if not isinstance(item, dict) else item.get("content", "")
        raw_dense = getattr(item, "dense_score", None) if not isinstance(item, dict) else item.get("dense_score")
        raw_sparse = getattr(item, "sparse_score", None) if not isinstance(item, dict) else item.get("sparse_score")

        dense = max(float(raw_dense), 0.0) if raw_dense is not None else 0.0
        sparse = max(float(raw_sparse), 0.0) if raw_sparse is not None else 0.0

        sentences = split_into_sentences(content)
        best_sent = max([score_sentence_relevance(s, query) for s in sentences], default=0.0)

        # Composite semantic relevance score
        return round(0.55 * dense + 0.30 * best_sent + 0.15 * min(sparse, 1.0), 4)

    def rerank(
        self,
        query: str,
        candidates: List[Union[SearchResultItem, Dict[str, Any]]],
        top_k: Optional[int] = None,
    ) -> List[Union[SearchResultItem, Dict[str, Any]]]:
        if not candidates:
            return []

        # Shallow copy to avoid mutating external caller's order unexpectedly
        items = list(candidates)

        ranker = self._get_ranker()
        if ranker is not None:
            try:
                from flashrank import RerankRequest
                passages = [
                    {"id": str(i), "text": getattr(c, "content", "") if not isinstance(c, dict) else c.get("content", "")}
                    for i, c in enumerate(items)
                ]
                rerank_req = RerankRequest(query=query, passages=passages)
                results = ranker.rerank(rerank_req)

                score_map = {int(r["id"]): float(r["score"]) for r in results if "id" in r and "score" in r}
                for i, c in enumerate(items):
                    if i in score_map:
                        sc = round(score_map[i], 4)
                    else:
                        sc = self._compute_fallback_score(c, query)

                    if isinstance(c, dict):
                        c["rerank_score"] = sc
                    else:
                        c.rerank_score = sc

                def sort_key(x):
                    sc = getattr(x, "rerank_score", None) if not isinstance(x, dict) else x.get("rerank_score")
                    base = getattr(x, "score", 0.0) if not isinstance(x, dict) else x.get("score", 0.0)
                    return (sc if sc is not None else -1.0, base)

                items.sort(key=sort_key, reverse=True)
                return items[:top_k] if top_k else items
            except Exception as e:
                logger.warning("FlashRank execution error ({}); falling back to semantic scoring.", e)

        # Fallback semantic scoring: combines dense similarity, sentence-level relevance, and lexical frequency
        for c in items:
            sc = self._compute_fallback_score(c, query)
            if isinstance(c, dict):
                c["rerank_score"] = sc
            else:
                c.rerank_score = sc

        def sort_key(x):
            sc = getattr(x, "rerank_score", None) if not isinstance(x, dict) else x.get("rerank_score")
            base = getattr(x, "score", 0.0) if not isinstance(x, dict) else x.get("score", 0.0)
            return (sc if sc is not None else -1.0, base)

        items.sort(key=sort_key, reverse=True)
        return items[:top_k] if top_k else items


_global_reranker: Optional[CrossEncoderReranker] = None


def get_reranker(model_name: str = "ms-marco-TinyBERT-L-2-v2") -> CrossEncoderReranker:
    global _global_reranker
    if _global_reranker is None:
        _global_reranker = CrossEncoderReranker(model_name=model_name)
    return _global_reranker


class EvidenceEvaluator:
    """
    Rule D-1: Evidence Confidence Evaluator.
    Determines whether retrieved context is sufficient and relevant to answer the query.
    If evidence confidence is low (out-of-domain or irrelevant), triggers calibrated refusal.
    Stage 2: Accurately factors in neural cross-encoder rerank_score to prevent false refusals
    on semantic paraphrases.
    """

    def __init__(self, default_threshold: float = 0.35):
        self.default_threshold = default_threshold

    def evaluate(
        self,
        query: str,
        candidates: List[Union[SearchResultItem, Dict[str, Any]]],
        threshold: Optional[float] = None,
    ) -> Tuple[ConfidenceLevel, bool, str]:
        """
        Evaluates candidate chunks against query.
        Returns:
            (ConfidenceLevel, should_refuse, refusal_message_or_reason)
        """
        eff_threshold = threshold if threshold is not None else self.default_threshold

        # 1. Zero candidate chunks retrieved
        if not candidates:
            logger.info("Evidence Gate: 0 candidates found for query '{}' -> Refusal triggered.", query)
            return ConfidenceLevel.LOW, True, REFUSAL_MESSAGE

        # 2. Check maximum dense, sparse, and rerank scores
        raw_dense = [getattr(c, "dense_score", None) if not isinstance(c, dict) else c.get("dense_score") for c in candidates]
        dense_scores = [float(s) for s in raw_dense if s is not None]

        raw_sparse = [getattr(c, "sparse_score", None) if not isinstance(c, dict) else c.get("sparse_score") for c in candidates]
        sparse_scores = [float(s) for s in raw_sparse if s is not None]

        raw_rerank = [getattr(c, "rerank_score", None) if not isinstance(c, dict) else c.get("rerank_score") for c in candidates]
        rerank_scores = [float(s) for s in raw_rerank if s is not None]

        max_dense = max(dense_scores) if dense_scores else 0.0
        max_sparse = max(sparse_scores) if sparse_scores else 0.0
        max_rerank = max(rerank_scores) if rerank_scores else 0.0

        # 3. Check term overlap across all candidate contents
        q_tokens = set(re.findall(r"\w+", query.lower()))
        stop_words = {"what", "is", "the", "how", "why", "when", "where", "which", "who", "are", "a", "an", "in", "of", "for", "to"}
        content_q_tokens = q_tokens - stop_words or q_tokens

        all_text = " ".join(
            (getattr(c, "content", "") if not isinstance(c, dict) else c.get("content", "")).lower()
            for c in candidates
        )
        terms_matched = sum(1 for term in content_q_tokens if term in all_text)
        term_match_ratio = terms_matched / max(1, len(content_q_tokens))

        logger.debug(
            "Evidence Gate metrics: max_dense={:.4f}, max_sparse={:.4f}, max_rerank={:.4f}, term_match_ratio={:.2f}, threshold={:.2f}",
            max_dense, max_sparse, max_rerank, term_match_ratio, eff_threshold,
        )

        # 4. Out-of-Domain / Insufficient Evidence Condition:
        # If dense similarity is below threshold AND sparse keyword score is negligible AND rerank score is low,
        # OR if zero non-stopword query terms appear in retrieved context and dense & rerank scores are both low
        if (max_dense < eff_threshold and max_sparse < 0.10 and max_rerank < 0.40) or (term_match_ratio == 0.0 and max_dense < 0.50 and max_rerank < 0.50):
            logger.info(
                "Evidence Gate: Insufficient confidence (max_dense={:.4f} < {:.2f}, max_rerank={:.4f}) -> Refusal triggered.",
                max_dense, eff_threshold, max_rerank,
            )
            return ConfidenceLevel.LOW, True, REFUSAL_MESSAGE

        # 5. High Confidence vs Ambiguous
        if max_dense >= 0.55 or max_rerank >= 0.55 or (max_dense >= 0.40 and max_sparse >= 0.15):
            return ConfidenceLevel.HIGH, False, ""

        return ConfidenceLevel.AMBIGUOUS, False, ""


def prune_and_format_context(
    query: str,
    candidates: List[Union[SearchResultItem, Dict[str, Any]]],
    min_sentence_relevance: float = 0.10,
) -> Tuple[str, List[Source]]:
    """
    Applies Sentence-Level Noise Pruning (Rule D-2) to all candidates
    and formats context with inline citation headers (Rule D-3):
    [Doc: filename, Page: page_number, Chunk: chunk_id].
    """
    formatted_passages: List[str] = []
    sources: List[Source] = []

    for c in candidates:
        chunk_id = getattr(c, "chunk_id", None) or (c.get("chunk_id", "") if isinstance(c, dict) else "")
        doc_id = getattr(c, "document_id", None) or (c.get("document_id", "") if isinstance(c, dict) else "")
        filename = getattr(c, "filename", None) or (c.get("filename", "unknown.pdf") if isinstance(c, dict) else "unknown.pdf")
        page_no = getattr(c, "page_number", None) or (c.get("page_number", 1) if isinstance(c, dict) else 1)
        score = getattr(c, "score", None) or (c.get("score", 0.0) if isinstance(c, dict) else 0.0)
        dense_score = getattr(c, "dense_score", None) if not isinstance(c, dict) else c.get("dense_score")
        sparse_score = getattr(c, "sparse_score", None) if not isinstance(c, dict) else c.get("sparse_score")
        rerank_score = getattr(c, "rerank_score", None) if not isinstance(c, dict) else c.get("rerank_score")
        content = getattr(c, "content", None) or (c.get("content", "") if isinstance(c, dict) else "")

        pruned_content, orig_count, ret_count = prune_chunk_sentences(
            content, query, min_sentence_relevance=min_sentence_relevance
        )

        citation_tag = f"[Doc: {filename}, Page: {page_no}, Chunk: {chunk_id}]"
        passage = f"Passage {citation_tag}:\n{pruned_content}"
        formatted_passages.append(passage)

        sources.append(
            Source(
                id=str(chunk_id),
                document_id=str(doc_id),
                filename=str(filename),
                page_number=int(page_no),
                score=score,
                dense_score=dense_score,
                sparse_score=sparse_score,
                rerank_score=rerank_score,
                content=pruned_content,
            )
        )

    context_str = "\n\n".join(formatted_passages)
    return context_str, sources
