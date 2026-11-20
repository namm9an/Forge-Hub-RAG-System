"""Offline tests of workload safety, independent gold data and honest scoring.

These are harness contract tests, never measurements of concurrency or model quality.
"""
import math
from contextlib import ExitStack, asynccontextmanager
import uuid

import httpx
import pytest

from scripts.evaluate_workloads import (
    NEEDLE_FACTS, REFERENCE_FACTS, assert_owned_sources, build_settings, contains_fact, corpus_layout,
    loopback_network, needle_corpus, parser, percentiles, summarize_requests, validate_target,
)


@pytest.mark.parametrize("dsn", ["", "postgresql://localhost/forge_hub", "postgresql://example.test/forge_hub_test",
    "postgresql://127.0.0.1/forge_hub_test?options=unsafe", "postgresql://localhost/forge_hub_test#other",
    "postgresql://localhost:invalid/forge_hub_test", "postgresql://localhost:99999/forge_hub_test"])
def test_workload_refuses_non_disposable_or_malformed_target(dsn):
    with pytest.raises(ValueError):
        validate_target(dsn)


def test_explicit_loopback_database_and_individual_workloads():
    dsn = "postgresql://synthetic:synthetic@127.0.0.1:15432/forge_hub_test_workloads"
    assert validate_target(dsn) == dsn
    options = parser().parse_args(["--database-url", dsn, "--workload", "load30", "--capacity", "64"])
    assert options.workload == "load30" and options.capacity == 64


def test_sustained_profiles_match_the_documented_load_and_run_length():
    from collections import Counter
    from scripts.evaluate_workloads import SUSTAINED_PROFILES

    expected = {
        "sustained-read": (50, 10, 60, 100, {"search": 1}),
        "sustained-mixed": (30, 30, 300, 45, {"stream": 6, "search": 3, "upload": 1}),
        "sustained-upload": (25, 25, 120, 15, {"upload": 1}),
    }
    assert {name: (p.users, p.namespaces, p.seconds, p.target, Counter(p.operations))
            for name, p in SUSTAINED_PROFILES.items()} == expected
    assert parser().parse_args(["--workload", "sustained-all"]).workload == "sustained-all"


def test_sustained_metrics_use_success_only_and_close_at_strict_throughput_limit():
    from scripts.evaluate_workloads import SUSTAINED_PROFILES, summarize_sustained

    profile = SUSTAINED_PROFILES["sustained-mixed"]
    records = [
        dict(operation="search", status_code=200, latency_ms=110, retrieval_ms=120, owned=True, fact_correct=True),
        dict(operation="stream", status_code=200, latency_ms=1440, ttft_ms=220, completion_ms=1400,
             owned=True, fact_correct=True),
        dict(operation="search", status_code=503, latency_ms=1),
        dict(operation="stream", status_code=200, latency_ms=5, error="IncompleteStream"),
    ]
    summary = summarize_sustained(profile, records, 2 / profile.target)
    assert summary["completed"] == 2
    assert summary["throughput_status"] == "FAIL"
    assert summary["status_counts"] == {"200": 3, "503": 1}
    assert summary["metrics"]["search_retrieval_ms"]["percentiles"]["p50"] == 120
    assert summary["metrics"]["stream_first_token_ms"]["status"] == "PASS"
    assert summary["metrics"]["stream_completion_ms"]["status"] == "PASS"
    assert summary["metrics"]["http_5xx_percent"]["status"] == "FAIL"


@pytest.mark.asyncio
async def test_stream_measurement_requires_token_persisted_completion_and_owned_sources():
    from scripts.evaluate_workloads import read_chat_stream

    metadata = 'event: metadata\ndata: {"sources":[{"document_id":"own"}]}\n\n'
    token = 'data: {"token":"violet heron"}\n\n'
    complete = ('event: complete\ndata: {"complete":true,"sources":[{"document_id":"own"}],'
                '"generation_backend":"extractive_fallback","refusal":false}\n\ndata: [DONE]\n\n')
    result = {"owned": True}
    await read_chat_stream(httpx.Response(200, text=metadata + token + complete), {"own"}, "violet heron", result, 0)
    assert result["owned"] and result["fact_correct"]
    assert 0 <= result["ttft_ms"] <= result["completion_ms"]
    with pytest.raises(ValueError):
        await read_chat_stream(httpx.Response(200, text=metadata + token), {"own"}, "violet heron", {}, 0)
    with pytest.raises(ValueError):
        await read_chat_stream(httpx.Response(200, text=token + complete), {"own"}, "violet heron", {}, 0)
    with pytest.raises(ValueError):
        await read_chat_stream(httpx.Response(200, text=metadata.replace('"own"', '"other"') + token + complete),
                               {"own"}, "violet heron", {}, 0)


@pytest.mark.asyncio
async def test_sustained_upload_checks_vectors_and_keeps_cleanup_out_of_the_http_mix():
    from scripts.evaluate_workloads import sustained_request

    document = uuid.uuid4()
    requests, queries = [], []

    class TestConnection:
        @asynccontextmanager
        async def acquire(self):
            yield self

        @asynccontextmanager
        async def transaction(self):
            yield

        async def fetchval(self, sql, document_id, session_id):
            assert document_id == document and session_id == "disposable"
            queries.append(sql.split()[0])
            return 1 if sql.startswith("SELECT") else document

        async def execute(self, sql, session_id):
            assert session_id == "disposable"
            queries.append(sql.split()[0])

    def respond(request):
        requests.append((request.method, request.url.path))
        return httpx.Response(200, json={"session_id": "disposable",
            "documents": [{"document_id": str(document), "chunks_created": 1}]})

    user = {"session_id": "primary", "upload_session": "disposable",
            "headers": {}, "subject": "owner", "pdf": b"synthetic"}
    async with httpx.AsyncClient(base_url="http://test", transport=httpx.MockTransport(respond)) as client:
        row = await sustained_request(client, TestConnection(), user, "upload", 0)
    assert row["owned"] and row["fact_correct"]
    assert requests == [("POST", "/api/documents/upload")]
    assert queries == ["SELECT", "DELETE", "UPDATE"]


def test_corpus_contains_exactly_500_unique_pages_at_independently_declared_depths():
    pages, cases = needle_corpus()
    assert len(pages) == len(set(pages)) == 500
    assert [(case["chunk_index"], case["page_number"]) for case in cases] == [(25, 26), (250, 251), (474, 475)]
    assert [case["depth"] for case in cases] == [0.05, 0.50, 0.95]
    assert sum("retired draft" in page for page in pages) == 15
    for fact, case in zip(NEEDLE_FACTS, cases):
        assert case["answer"] == fact.answer
        assert fact.subject in pages[case["chunk_index"]]
        assert sum(fact.answer in page for page in pages) == 1
        assert sum(fact.subject in page and "retired draft" in page for page in pages) == 5
    with pytest.raises(ValueError):
        needle_corpus(499)


def test_multiturn_gold_values_are_distinct_across_private_subjects():
    assert len(REFERENCE_FACTS) == 2
    assert len({row[0] for row in REFERENCE_FACTS}) == len({row[1] for row in REFERENCE_FACTS}) == len({row[2] for row in REFERENCE_FACTS}) == 2


@pytest.mark.parametrize("values", [[math.nan], [math.inf], [-1]])
def test_invalid_latency_samples_are_not_silently_scored(values):
    with pytest.raises(ValueError):
        percentiles(values)


def test_percentiles_use_all_successful_samples_and_empty_is_unmeasured():
    assert percentiles([]) is None
    assert percentiles([10, 20, 30, 40]) == pytest.approx({"p50": 25.0, "p90": 37.0, "p95": 38.5, "p99": 39.7})


def test_pdf_whitespace_and_numeric_substrings_do_not_corrupt_gold_scoring():
    assert contains_fact("COPPER\nheron 317", "copper heron 317")
    assert contains_fact("retains audit logs for 42\n days.", "42 days")
    assert not contains_fact("retains for 142 days", "42 days")
    assert not contains_fact("violet heron 3179", "violet heron 317")


@pytest.mark.parametrize("defect", ["missing_chunk", "duplicate_hash", "wrong_page", "wrong_index", "missing_vector"])
def test_actual_corpus_validation_rejects_underfilled_or_misplaced_ingestion(defect):
    rows = [{"chunk_index": i, "page_number": i + 1, "content_hash": str(i)} for i in range(500)]
    assert corpus_layout(rows, 500)["layout_status"] == "PASS"
    vectors = 500
    if defect == "missing_chunk":
        rows.pop()
    elif defect == "duplicate_hash":
        rows[20]["content_hash"] = rows[19]["content_hash"]
    elif defect == "wrong_page":
        rows[20]["page_number"] = 500
    elif defect == "wrong_index":
        rows[20]["chunk_index"] = 500
    else:
        vectors = 499
    assert corpus_layout(rows, vectors)["layout_status"] == "FAIL"


def successful_record(latency=100):
    return {"status_code": 200, "latency_ms": latency, "owned": True,
            "fact_correct": True, "generation_backend": "extractive_fallback"}


@pytest.mark.parametrize("code", [401, 429, 503])
def test_fast_rejection_is_failure_and_never_improves_success_latency(code):
    summary = summarize_requests([successful_record(1000), {"status_code": code, "latency_ms": 1}], 2)
    assert summary["status"] == "FAIL" and summary["errors"] == 1
    assert summary["completed"] == 1
    assert summary["successful_http_latency_ms"]["p50"] == 1000
    assert summary["status_counts"][str(code)] == 1


def test_only_rejections_cannot_pass_or_get_zero_latency():
    summary = summarize_requests([{"status_code": 503, "latency_ms": 1}], 1, 100)
    assert summary["status"] == "FAIL"
    assert summary["successful_http_latency_ms"] is None
    assert summary["p95_budget_status"] == "NOT MEASURED"


def test_transport_and_invalid_payload_failures_are_counted():
    rows = [{"status_code": None, "latency_ms": 2, "error": "ConnectError"},
            {"status_code": 200, "latency_ms": 3, "error": "KeyError"}]
    summary = summarize_requests(rows, 1)
    assert summary["completed"] == 0 and summary["errors"] == 2 and summary["status"] == "FAIL"


def test_wrong_owner_wrong_fact_or_provider_call_cannot_pass():
    for field, value in [("owned", False), ("fact_correct", False), ("generation_backend", "provider")]:
        row = successful_record()
        row[field] = value
        assert summarize_requests([row], 1)["status"] == "FAIL"


def test_strict_latency_budget_equal_to_observation_fails():
    assert summarize_requests([successful_record(100)], 1, 100)["p95_budget_status"] == "FAIL"
    assert summarize_requests([successful_record(99)], 1, 100)["p95_budget_status"] == "PASS"
    assert summarize_requests([successful_record()], 1)["p95_budget_status"] == "NOT MEASURED"
    assert summarize_requests([successful_record(99.9999)], 1, 100)["p95_budget_status"] == "PASS"


@pytest.mark.parametrize("payload", [{}, {"sources": None}, {"sources": [{"document_id": "foreign"}]}, {"sources": [{}]}])
def test_ownership_assertion_rejects_foreign_missing_or_malformed_sources(payload):
    with pytest.raises(ValueError):
        assert_owned_sources(payload, {"owned"})


def test_valid_owned_sources_and_refusal_empty_sources():
    assert_owned_sources({"sources": [{"document_id": "owned"}]}, {"owned"})
    assert_owned_sources({"sources": []}, {"owned"})


def test_settings_ignore_application_secrets_and_label_nondefault_capacity(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "synthetic-must-not-be-used")
    monkeypatch.setenv("GEMINI_API_KEY", "synthetic-must-not-be-used")
    monkeypatch.setenv("DATABASE_URL", "postgresql://remote/production")
    settings = build_settings("postgresql://localhost/forge_hub_test", "synthetic-public-key", 64)
    assert settings.database_url == "postgresql://localhost/forge_hub_test"
    assert settings.openrouter_api_key is settings.gemini_api_key is None
    assert settings.max_inflight_requests == 64 and settings.embedding_workers == 2
    assert settings.requests_per_minute == 120
    with pytest.raises(ValueError):
        build_settings("postgresql://localhost/forge_hub_test", "synthetic-public-key", 50)


def test_synthetic_rate_comparison_retains_auth_and_rejects_invalid_budget():
    from scripts.evaluate_workloads import parser

    args = parser().parse_args(["--workload", "sustained-read", "--comparison-requests-per-minute", "1200"])
    target, key = "postgresql://localhost/forge_hub_test", "synthetic-public-key"
    settings = build_settings(target, key, args.capacity,
                              comparison_requests_per_minute=args.comparison_requests_per_minute)
    assert settings.requests_per_minute == 1200
    assert settings.auth_public_key == key
    assert settings.max_inflight_requests == 8
    assert parser().parse_args([]).comparison_requests_per_minute is None
    with pytest.raises(ValueError):
        build_settings(target, key, 8, comparison_requests_per_minute=10001)


def test_workload_http_guard_blocks_external_sync_request_before_dns(monkeypatch):
    import socket
    calls = []
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: calls.append(True))
    with ExitStack() as stack:
        loopback_network(stack, {15432, 18080})
        with httpx.Client(trust_env=False) as client, pytest.raises(RuntimeError, match="blocks non-target HTTP"):
            client.get("https://provider-disabled.invalid/test")
    assert calls == []


@pytest.mark.asyncio
async def test_workload_http_guard_blocks_external_async_request_before_dns(monkeypatch):
    import socket
    calls = []
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: calls.append(True))
    with ExitStack() as stack:
        loopback_network(stack, {15432, 18080})
        async with httpx.AsyncClient(trust_env=False) as client:
            with pytest.raises(RuntimeError, match="blocks non-target HTTP"):
                await client.get("https://provider-disabled.invalid/test")
    assert calls == []
