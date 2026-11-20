"""
Forge Hub RAG System — Evidence Gating & Noise Pruning Engine
Complies with:
- Rule D-1: Out-of-Domain Refusal Gate (Evidence Confidence Evaluator)
- Rule D-2: Sentence-Level Noise Pruner (Context Filtering)
- Stage 2: Quantized Cross-Encoder Reranking (FlashRank with semantic fallback)
"""

import json
import math
import re
import threading
from typing import List, Dict, Any, Tuple, Optional, Union
from loguru import logger
from .schemas import ConfidenceLevel, SearchResultItem, Source

REFUSAL_MESSAGE = "The provided documents do not contain sufficient context to answer this question."

_METRIC = re.compile(r"\b[a-z][\w.-]*@\d+\b", re.I)
_CHOICE = re.compile(r"^(?:and\s+)?(?:which|what (?:minimum|maximum))\s+(?:minimum\s+|maximum\s+)?", re.I)
_BOUND_QUESTION = re.compile(r"\b(?:minimum|maximum|guarantee\w*|at least|at most)\b", re.I)
_METRIC_VALUE = re.compile(
    r"\s*(?:(?:is|was|of|equals|exceeds|(?:must|should) be)\s+|[:=]\s*)?"
    r"(?:(?:at least|at most)\s+|[<>]=?\s*)?[+-]?\d", re.I,
)
_UNSPECIFIED_FIELD = re.compile(
    r"\b(\w+)\s+(?:(?:is|was|remains)\s+)?"
    r"(?:unspecified|undocumented|unknown|not (?:specified|documented|known|provided))\b", re.I,
)
_PROTOCOL = re.compile(r"\b(?:TLS|SSL|HTTPS?|SSH|TCP|UDP|QUIC|DNS|SMTP|IPsec|gRPC|AMQP|MQTT)\b", re.I)
_TIME_UNIT = r"(?:milliseconds?|seconds?|minutes?|hours?|days?|weeks?|months?|years?)"
_AMOUNT = r"\b(?:\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|a|an|half)[\s-]*"
_DURATION_QUESTION = re.compile(rf"^(?:and\s+)?how (?:long|many {_TIME_UNIT})\b", re.I)
_DURATION = re.compile(rf"{_AMOUNT}{_TIME_UNIT}\b|\b(?:indefinitely|permanently)\b", re.I)
_LENGTH = re.compile(
    rf"{_AMOUNT}(?:mm|cm|km|(?:kilo)?met(?:er|re)s?|inch(?:es)?|foot|feet|yards?)\b", re.I,
)
_RETENTION = re.compile(r"\b(?:retain\w*|retention|keep\w*|kept|stored?|storing|expir\w*|ttl)\b", re.I)
_QUESTION_FILLER = frozenset(
    "a an and are can could do does have in is it its may of on should the their them they to was we were will would".split()
)
_DURATION_FILLER = _QUESTION_FILLER | {"take", "takes"}


def _terms(text: str) -> set[str]:
    words = {word.removesuffix("s") for word in re.findall(r"\w+", text.casefold())}
    return words | {word[:-1] for word in words if word.endswith("ed")}


def _supports_metrics(question, excerpts, requested, require_all):
    needs_value = _BOUND_QUESTION.search(question)
    found = {
        metric[0].casefold()
        for excerpt in excerpts
        for sentence in split_into_sentences(excerpt)
        for metric in _METRIC.finditer(sentence)
        if not needs_value or _METRIC_VALUE.match(sentence[metric.end():])
    }
    return requested <= found if require_all else bool(requested & found)


def _corroborates_question(question: str, excerpt: str) -> bool:
    # shortcut: explicit fields and quantity cues catch missing facts, add an
    # independently validated answerability model for unrestricted paraphrases.
    choice = _CHOICE.match(question)
    if choice:
        phrase = re.findall(r"\b[^\W\d_]+\b", question[choice.end():])
        if phrase and phrase[0].istitle() and len(phrase) > 2:
            phrase = phrase[1:]
        fields = _terms(" ".join(phrase[:2]))
        if fields:
            topic = _terms(" ".join(phrase[2:])) - _QUESTION_FILLER
            return any(
                not (fields & _terms(" ".join(_UNSPECIFIED_FIELD.findall(sentence))))
                and (fields & _terms(sentence) or (
                    "protocol" in fields and _PROTOCOL.search(sentence) and topic & _terms(sentence)
                ))
                for sentence in split_into_sentences(excerpt)
            )

    duration = _DURATION_QUESTION.match(question)
    if duration:
        words = re.findall(r"\w+", question[duration.end():].casefold())
        anchors = _terms(" ".join(word for word in words if word not in _DURATION_FILLER))
        retention = bool(_RETENTION.search(question))
        if retention:
            anchors = {word for word in anchors if not _RETENTION.fullmatch(word)}
        allows_length = not retention and duration[0].casefold().endswith("how long")
        return any(
            (not anchors or anchors & _terms(sentence))
            and (_DURATION.search(sentence) or (allows_length and _LENGTH.search(sentence)))
            and (not retention or _RETENTION.search(sentence))
            for sentence in split_into_sentences(excerpt)
        )
    return True


class NeuralRerankUnavailable(RuntimeError):
    pass


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

    # An exact-word score cannot choose between semantically paraphrased sentences.
    if not retained:
        retained = sentences

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
        self._lock = threading.Lock()
        self.backend = "heuristic"

    def _get_ranker(self):
        if not self._flashrank_checked:
            self._flashrank_checked = True
            try:
                from flashrank import Ranker
                self._ranker = Ranker(model_name=self.model_name)
                logger.info("Initialized FlashRank cross-encoder with model {}", self.model_name)
            except Exception as e:
                logger.info("FlashRank unavailable ({}); heuristic ranking active.", type(e).__name__)
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
        *,
        require_neural: bool = False,
    ) -> List[Union[SearchResultItem, Dict[str, Any]]]:
        # Model initialization and inference share one lock across worker threads.
        with self._lock:
            return self._rerank(query, candidates, top_k, require_neural)

    def _rerank(self, query, candidates, top_k, require_neural):
        if not candidates:
            return []

        # Shallow copy to avoid mutating external caller's order unexpectedly
        items = [dict(c) if isinstance(c, dict) else c.model_copy(deep=True) for c in candidates]

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
                if (len(results) != len(items) or set(score_map) != set(range(len(items)))
                        or any(not math.isfinite(value) or not 0 <= value <= 1 for value in score_map.values())):
                    raise ValueError("Incomplete or invalid neural reranker scores")
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
                self.backend = "flashrank"
                return items[:top_k] if top_k else items
            except Exception as e:
                logger.warning("FlashRank execution failed ({}); heuristic ranking active.", type(e).__name__)

        self.backend = "heuristic"
        if require_neural:
            raise NeuralRerankUnavailable("Neural retrieval is unavailable")
        # This weighted heuristic is not cross-encoder or semantic-quality evidence.
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
_reranker_init_lock = threading.Lock()


def get_reranker(model_name: str = "ms-marco-TinyBERT-L-2-v2") -> CrossEncoderReranker:
    global _global_reranker
    with _reranker_init_lock:
        if _global_reranker is None or _global_reranker.model_name != model_name:
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
        candidates: List[Union[SearchResultItem, Source, Dict[str, Any]]],
        threshold: Optional[float] = None,
        *,
        question: Optional[str] = None,
        require_all_metrics: bool = True,
    ) -> Tuple[ConfidenceLevel, bool, str]:
        """
        Evaluates candidate chunks against query.
        Returns:
            (ConfidenceLevel, should_refuse, refusal_message_or_reason)
        """
        eff_threshold = threshold if threshold is not None else self.default_threshold

        # 1. Zero candidate chunks retrieved
        if not candidates:
            logger.info("Evidence Gate: no candidates; refusal triggered.")
            return ConfidenceLevel.LOW, True, REFUSAL_MESSAGE

        # 2. Check maximum dense, sparse, and rerank scores
        try:
            raw_dense = [getattr(c, "dense_score", None) if not isinstance(c, dict) else c.get("dense_score") for c in candidates]
            dense_scores = [float(s) for s in raw_dense if s is not None]
            raw_sparse = [getattr(c, "sparse_score", None) if not isinstance(c, dict) else c.get("sparse_score") for c in candidates]
            sparse_scores = [float(s) for s in raw_sparse if s is not None]
            raw_rerank = [getattr(c, "rerank_score", None) if not isinstance(c, dict) else c.get("rerank_score") for c in candidates]
            rerank_scores = [float(s) for s in raw_rerank if s is not None]
        except (TypeError, ValueError, OverflowError):
            return ConfidenceLevel.LOW, True, REFUSAL_MESSAGE
        if not all(math.isfinite(value) for value in dense_scores + sparse_scores + rerank_scores):
            return ConfidenceLevel.LOW, True, REFUSAL_MESSAGE

        max_dense = max(dense_scores) if dense_scores else 0.0
        max_sparse = max(sparse_scores) if sparse_scores else 0.0
        max_rerank = max(rerank_scores) if rerank_scores else 0.0

        # 3. Check term overlap across all candidate contents
        q_tokens = set(re.findall(r"\w+", query.lower()))
        stop_words = {"what", "is", "the", "how", "why", "when", "where", "which", "who", "are", "a", "an", "in", "of", "for", "to"}
        content_q_tokens = q_tokens - stop_words or q_tokens

        excerpts = [(c.get("content") if isinstance(c, dict) else c.content) or "" for c in candidates]
        question = (query if question is None else question).strip()
        metrics = set(_METRIC.findall(question.casefold()))
        corroborated = (
            _supports_metrics(question, excerpts, metrics, require_all_metrics) if metrics
            else any(_corroborates_question(question, excerpt) for excerpt in excerpts)
        )
        if not corroborated:
            return ConfidenceLevel.LOW, True, REFUSAL_MESSAGE
        all_text = " ".join(excerpts).lower()
        terms_matched = len(content_q_tokens.intersection(re.findall(r"\w+", all_text)))
        term_match_ratio = terms_matched / max(1, len(content_q_tokens))

        logger.debug(
            "Evidence Gate metrics: max_dense={:.4f}, max_sparse={:.4f}, max_rerank={:.4f}, term_match_ratio={:.2f}, threshold={:.2f}",
            max_dense, max_sparse, max_rerank, term_match_ratio, eff_threshold,
        )

        # 4. Out-of-Domain / Insufficient Evidence Condition:
        # FTS rank is not confidence: incidental OR matches cannot override weak semantic evidence.
        lexical_exact = max_sparse > 0 and bool(content_q_tokens) and terms_matched == len(content_q_tokens)
        if (max_dense < eff_threshold and max_rerank < 0.40 and not lexical_exact) or (term_match_ratio == 0.0 and max_dense < 0.50 and max_rerank < 0.50):
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
    max_context_chars: int = 24000,
) -> Tuple[str, List[Source]]:
    """
    Pack whole admitted chunks into a bounded, explicitly untrusted evidence payload.

    The compatibility name is retained, but exact-word sentence pruning is not a
    semantic filter: it can delete the answer after the reranker admitted it.
    Drop whole chunks that do not fit instead of silently truncating evidence.
    """
    formatted_passages: List[str] = []
    sources: List[Source] = []
    used = 0

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

        if not content or not chunk_id or int(page_no) < 1:
            continue
        if re.search(r"[\x00-\x1f\[\]]", str(filename)) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", str(chunk_id)):
            continue
        citation_tag = f"[Doc: {filename}, Page: {page_no}, Chunk: {chunk_id}]"
        passage = json.dumps({"citation": citation_tag, "text": content}, ensure_ascii=False)
        cost = len(passage) + (2 if formatted_passages else 0)
        if used + cost > max_context_chars:
            continue
        formatted_passages.append(passage)
        used += cost

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
                content=content,
            )
        )

    context_str = "\n\n".join(formatted_passages)
    return context_str, sources
