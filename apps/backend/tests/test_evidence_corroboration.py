from app.gating import EvidenceEvaluator, REFUSAL_MESSAGE
from scripts.evaluate_golden import load_golden
import math
import pytest


def test_large_lexical_rank_with_only_incidental_overlap_does_not_defeat_refusal():
    candidate = {"content": "Network route selection uses replicated routing tables.",
                 "dense_score": 0.05, "sparse_score": 10.0, "rerank_score": 0.001}
    _, refusal, message = EvidenceEvaluator().evaluate(
        "What is the fastest route for a giraffe to reach Mars?", [candidate])
    assert refusal is True
    assert message == REFUSAL_MESSAGE


def test_exact_technical_identifier_can_be_supported_lexically():
    candidate = {"content": "ERR_STORAGE_42 means the durable storage device is unavailable.",
                 "dense_score": 0.05, "sparse_score": 0.2, "rerank_score": 0.001}
    _, refusal, _ = EvidenceEvaluator().evaluate("ERR_STORAGE_42", [candidate])
    assert refusal is False


@pytest.mark.parametrize("case_id, page", [
    ("eval_025", 1), ("eval_026", 3), ("eval_027", 4),
    ("eval_028", 5), ("eval_029", 7), ("eval_030", 8),
])
def test_high_topic_relevance_cannot_supply_missing_answer(case_id, page):
    golden = load_golden()
    query = next(case["query"] for case in golden["cases"] if case["id"] == case_id)
    candidate = {"content": golden["pages"][page - 1], "dense_score": 0.9, "rerank_score": 0.99}
    assert EvidenceEvaluator().evaluate(query, [candidate])[1:] == (True, REFUSAL_MESSAGE)


@pytest.mark.parametrize("question, evidence", [
    ("Which encryption algorithm protects mail?", "Mail encryption uses AES-256."),
    ("What minimum Recall@10 is guaranteed?", "Guaranteed Recall@10 is 0.9."),
    ("What minimum Recall@10 is guaranteed?", "Recall@10 must be at least 0.90."),
    ("How long can the service hold a write lease?", "A write lease expires after 30 seconds."),
    ("How many days are audit events retained?", "The service retains audit events for 48 hours."),
    ("And how many days does it retain them?", "The archive retains logs for 42 days."),
    ("How long does it take?", "Rebuilding the database requires 45 seconds."),
    ("Which protocol protects archive replication?", "Archive replication uses mutual TLS."),
    ("How long is the Ethernet cable?", "The Ethernet cable is 5 meters long."),
    ("How long do we retain archived logs?", "Archive records are stored for 30 days."),
    ("Which compression format is used when the client protocol is unknown?",
     "Archive compression is Zstandard when the client protocol is unknown."),
])
def test_evidence_that_bears_the_requested_answer_remains_eligible(question, evidence):
    candidate = {"content": evidence, "dense_score": 0.9, "rerank_score": 0.99}
    assert EvidenceEvaluator().evaluate(question, [candidate])[1] is False


@pytest.mark.parametrize("question, evidence", [
    ("And which encryption algorithm protects the archive?", "The archive has a write-ahead log."),
    ("Which encryption algorithm protects the archive?", "Archive encryption is mandatory but the algorithm is unspecified."),
    ("How long is an archive session retained?", "An archive session sends a heartbeat every 10 seconds."),
    ("What minimum Recall@10 is guaranteed?", "The dashboard shows Recall@10."),
    ("What minimum Recall@10 is guaranteed?", "Recall@10 is not tracked, and latency is 12 ms."),
])
def test_related_field_or_value_without_answer_is_refused(question, evidence):
    candidate = {"content": evidence, "dense_score": 0.9, "rerank_score": 0.99}
    assert EvidenceEvaluator().evaluate(question, [candidate])[1:] == (True, REFUSAL_MESSAGE)


def test_preceding_question_does_not_impose_its_own_answer_fields():
    question = "What is its recovery phrase?"
    query = question + "\nPrevious user questions: What is minimum Recall@5?"
    candidate = {"content": "The recovery phrase is violet heron.", "dense_score": 0.9, "rerank_score": 0.99}
    assert EvidenceEvaluator().evaluate(query, [candidate], question=question)[1] is False


@pytest.mark.parametrize("field", ["dense_score", "sparse_score", "rerank_score"])
def test_nonfinite_evidence_cannot_be_admitted(field):
    candidate = {"content": "Network routing supports replicated server clusters.",
                 "dense_score": 0.8, "sparse_score": 0.2, "rerank_score": 0.8}
    candidate[field] = float("nan")
    _, refusal, _ = EvidenceEvaluator().evaluate("network routing", [candidate])
    assert refusal is True


@pytest.mark.parametrize("output", [[{"id": "0", "score": float("nan")}], [], [{"id": "0", "score": 1.1}]])
def test_invalid_neural_outputs_are_never_labeled_flashrank(output, monkeypatch):
    from app.gating import CrossEncoderReranker
    class BrokenRanker:
        def rerank(self, request): return output
    ranker = CrossEncoderReranker()
    monkeypatch.setattr(ranker, "_get_ranker", lambda: BrokenRanker())
    result = ranker.rerank("network", [{"content": "network routing", "dense_score": 0.4, "sparse_score": 0.0}])
    assert ranker.backend == "heuristic"
    assert all(math.isfinite(item["rerank_score"]) for item in result)


def test_neural_only_mode_rejects_missing_inference_before_returning_candidates(monkeypatch):
    from app.gating import CrossEncoderReranker, NeuralRerankUnavailable
    ranker = CrossEncoderReranker()
    monkeypatch.setattr(ranker, "_get_ranker", lambda: None)
    with pytest.raises(NeuralRerankUnavailable):
        ranker.rerank("network", [{"content": "network routing"}], require_neural=True)
