"""Evaluate labeled synthetic facts over the real local HTTP and retrieval path."""

import argparse
import asyncio
import json
import os
from pathlib import Path
import tempfile

# The installed model libraries read offline mode at import time.
if __name__ == "__main__":
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")

from scripts.evaluate_workloads import (
    REPO_ROOT, TestIdentities, assert_owned_sources, build_settings,
    isolated_http, upload, validate_target,
)
from app.gating import REFUSAL_MESSAGE


GOLDEN = REPO_ROOT / "apps/backend/tests/fixtures/golden_rag_eval.json"


def normalized(text):
    return " ".join(text.casefold().split())


def load_golden(path=GOLDEN):
    data = json.loads(path.read_text(encoding="utf-8"))
    pages, cases = data["pages"], data["cases"]
    positives = sum(bool(case["facts"]) for case in cases)
    if len(cases) != 30 or positives != 20 or len({case["id"] for case in cases}) != 30:
        raise ValueError("Golden evaluation requires 20 answerable and 10 distinct refusal cases")
    for case in cases:
        if not case["query"].strip():
            raise ValueError(f"Empty question in {case['id']}")
        for fact in case["facts"]:
            page, text = fact["page"], normalized(fact["text"])
            if not (1 <= page <= len(pages) and text and normalized(pages[page - 1]).count(text) == 1):
                raise ValueError(f"Invalid annotated fact in {case['id']}")
    return data


def score_case(case, payload, document_id):
    assert_owned_sources(payload, {document_id})
    sources, answer, facts = payload["sources"], normalized(payload["answer"]), case["facts"]
    if not facts:
        return {"refusal": payload["refusal_triggered"] is True and not sources
                and answer == normalized(REFUSAL_MESSAGE)}

    def supports(source, fact):
        return source["page_number"] == fact["page"] and normalized(fact["text"]) in normalized(source["content"])

    relevant = sum(any(supports(source, fact) for fact in facts) for source in sources)
    hits = sum(any(supports(source, fact) for source in sources[:5]) for fact in facts)
    return {
        "precision": relevant / len(sources) if sources else 0.0,
        "facts_found_at_5": hits,
        "facts_expected": len(facts),
        "answer_facts_found": sum(normalized(fact["text"]) in answer for fact in facts),
    }


async def run(dsn, output):
    data = load_golden()
    identities = TestIdentities()
    settings = build_settings(dsn, identities.public_key, 8).model_copy(update={"max_history_turns": 0})
    records = []
    async with isolated_http(settings, 180) as (client, _, inference):
        user = await upload(client, identities, "golden-reader", data["pages"], "golden-runbook.pdf")
        before = inference.copy()
        for case in data["cases"]:
            response = await client.post(
                "/api/chat", headers=identities.headers(user["subject"]),
                json={"query": case["query"], "session_id": user["session_id"], "top_k": 5},
            )
            record = {"id": case["id"], "status_code": response.status_code}
            if response.status_code == 200:
                try:
                    payload = response.json()
                    backend, retrieval_ms = payload["generation_backend"], payload["retrieval_ms"]
                    record.update(score_case(case, payload, user["document_id"]))
                    record.update(generation_backend=backend, retrieval_ms=retrieval_ms)
                    record["sources"] = payload["sources"]
                except (KeyError, TypeError, ValueError) as error:
                    record["response_error"] = type(error).__name__
            records.append(record)
        observed = dict(inference - before)

    positive = [record for record in records if "precision" in record]
    facts_expected = sum(len(case["facts"]) for case in data["cases"])
    precision = sum(record["precision"] for record in positive) / 20
    recall = sum(record["facts_found_at_5"] for record in positive) / facts_expected
    refusals = sum(record.get("refusal") is True for record in records) / 10
    report = {
        "scope": "Real loopback HTTP, PostgreSQL, MPNet and FlashRank. Synthetic manual fact/page labels. Local extraction only.",
        "metrics": {
            "context_precision": precision, "fact_recall_at_5": recall, "negative_refusal": refusals,
            "extractive_answer_fact_coverage": sum(record["answer_facts_found"] for record in positive) / facts_expected,
            "provider_faithfulness": "NOT MEASURED", "provider_answer_relevance": "NOT MEASURED",
        },
        "acceptance": {
            "context_precision": "PASS" if precision >= 0.80 else "FAIL",
            "fact_recall_at_5": "PASS" if recall >= 0.85 else "FAIL",
            "negative_refusal": "PASS" if refusals == 1.0 else "FAIL",
            "http_and_model_path": "PASS" if (
                all(record["status_code"] == 200 and record.get("generation_backend") in {"refusal", "extractive_fallback"}
                    for record in records) and observed.get("flashrank") and set(observed) == {"flashrank"}
            ) else "FAIL",
            "provider_faithfulness": "NOT MEASURED", "provider_answer_relevance": "NOT MEASURED",
        },
        "observed_reranker_calls": observed, "cases": records, "backend_release_gate": "INCOMPLETE",
    }
    path = output / "golden_results.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2))
    print(f"Results: {path}")
    return 2 if "FAIL" in report["acceptance"].values() else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=os.environ.get("FORGE_HUB_TEST_DATABASE_URL", ""))
    args = parser.parse_args()
    try:
        validate_target(args.database_url)
    except ValueError as error:
        parser.error(str(error))
    output = Path(tempfile.mkdtemp(prefix="forge-hub-golden-"))
    return asyncio.run(run(args.database_url, output))


if __name__ == "__main__":
    raise SystemExit(main())
