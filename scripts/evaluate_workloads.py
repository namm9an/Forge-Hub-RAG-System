#!/usr/bin/env python3
"""Opt-in real HTTP/model workloads against a uniquely scoped disposable database.

No external inference: answer checks measure labeled extractive fallback, not
semantic answer quality. Run one workload at a time to avoid repeating ingestion.
"""
import argparse
import asyncio
from collections import Counter
from contextlib import ExitStack, asynccontextmanager, suppress
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib
import json
import logging
import math
import os
from pathlib import Path
import re
import socket
import sys
import tempfile
from time import perf_counter, time
from urllib.parse import urlsplit
from unittest.mock import patch
import uuid

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if __name__ == "__main__":
    # Hub libraries cache their offline policy during import; local models must already exist.
    for flag in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        os.environ[flag] = "1"
    os.environ.setdefault("LOGURU_LEVEL", "WARNING")

from scripts.benchmark_rag import generate_benchmark_pdf, validate_test_database

import asyncpg
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
import httpx
import jwt
import numpy as np
import uvicorn

from app.config import Settings, get_settings


@dataclass(frozen=True)
class NeedleFact:
    depth: float
    subject: str
    answer: str


# Gold facts are declared before retrieval; never inferred from returned sources.
NEEDLE_FACTS = (
    NeedleFact(0.05, "Solstice", "copper heron 317"),
    NeedleFact(0.50, "Meridian", "violet badger 641"),
    NeedleFact(0.95, "Equinox", "silver otter 829"),
)
REFERENCE_FACTS = (
    ("Aster", "42 days", "violet heron"),
    ("Boreal", "73 days", "amber otter"),
)
WORKLOADS = ("load30", "load50", "needle", "multiturn")


@dataclass(frozen=True)
class SustainedProfile:
    users: int
    namespaces: int
    seconds: int
    target: int
    operations: tuple[str, ...]


SUSTAINED_PROFILES = {
    "sustained-read": SustainedProfile(50, 10, 60, 100, ("search",)),
    "sustained-mixed": SustainedProfile(30, 30, 300, 45,
        ("stream", "search", "stream", "stream", "search", "stream", "stream", "search", "stream", "upload")),
    "sustained-upload": SustainedProfile(25, 25, 120, 15, ("upload",)),
}


def validate_target(dsn):
    validate_test_database(dsn)
    port = urlsplit(dsn).port or 5432
    if not 1 <= port <= 65535:
        raise ValueError("Invalid disposable database port")
    return dsn


def needle_corpus(count=500):
    if count != 500:
        raise ValueError("The needle workload requires exactly 500 unique chunks")
    pages = [
        f"Inventory ledger record {i:04d} belongs to transport unit {i:04d}. "
        f"Its assigned storage slot is {10000 + i}. This entry records routine inventory handling."
        for i in range(count)
    ]
    expected = []
    for subject_index, fact in enumerate(NEEDLE_FACTS):
        index = round(fact.depth * (count - 1))
        pages[index] = f"The approved current recovery phrase for the {fact.subject} archive is {fact.answer}."
        for revision in range(5):
            distractor_index = 10 + revision * 95 + subject_index
            pages[distractor_index] = (
                f"A retired draft for the {fact.subject} archive listed recovery phrase "
                f"bronze kestrel {9000 + distractor_index}. That draft is obsolete and must not be used."
            )
        expected.append({"depth": fact.depth, "chunk_index": index, "page_number": index + 1,
                         "subject": fact.subject, "answer": fact.answer})
    return pages, expected


def percentiles(values):
    if not values:
        return None
    if any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("Latency samples must be finite and nonnegative")
    return {f"p{p}": float(np.percentile(values, p)) for p in (50, 90, 95, 99)}


def contains_fact(text, fact):
    normalized = " ".join(text.casefold().split())
    expected = " ".join(fact.casefold().split())
    return bool(re.search(r"(?<!\w)" + re.escape(expected) + r"(?!\w)", normalized))


def corpus_layout(rows, vectors):
    valid = (len(rows) == 500 and len({row["content_hash"] for row in rows}) == 500 and vectors == 500
             and all(row["chunk_index"] == i and row["page_number"] == i + 1 for i, row in enumerate(rows)))
    return {"actual_chunks": len(rows), "unique_content_hashes": len({row["content_hash"] for row in rows}),
            "actual_embeddings": vectors, "layout_status": "PASS" if valid else "FAIL"}


def summarize_requests(records, elapsed, p95_budget_ms=None):
    if elapsed <= 0 or not math.isfinite(elapsed):
        raise ValueError("Elapsed measurement time must be positive and finite")
    accepted = [r for r in records if r.get("status_code") == 200 and not r.get("error")]
    checks = {"all_requests_completed": bool(records) and len(accepted) == len(records),
              "ownership": bool(accepted) and all(r.get("owned") is True for r in accepted),
              "gold_fact": bool(accepted) and all(r.get("fact_correct") is True for r in accepted),
              "provider_disabled": bool(accepted) and all(r.get("generation_backend") in
                  {"extractive_fallback", "refusal"} for r in accepted)}
    latency = percentiles([r["latency_ms"] for r in accepted])
    latency_status = "NOT MEASURED"
    if p95_budget_ms is not None and latency is not None:
        latency_status = "PASS" if latency["p95"] < p95_budget_ms else "FAIL"
    return {
        "requests": len(records), "completed": len(accepted),
        "status_counts": dict(Counter(str(r.get("status_code") or "transport_error") for r in records)),
        "errors": sum(bool(r.get("error")) or r.get("status_code") != 200 for r in records),
        "isolation_violations": sum(r.get("owned") is False for r in records),
        "completed_requests_per_second": round(len(accepted) / elapsed, 3),
        "successful_http_latency_ms": latency,
        "all_attempt_latency_ms": percentiles([r["latency_ms"] for r in records]),
        "checks": {key: "PASS" if value else "FAIL" for key, value in checks.items()},
        "p95_budget_ms": p95_budget_ms, "p95_budget_status": latency_status,
        "status": "PASS" if all(checks.values()) and latency_status != "FAIL" else "FAIL",
    }


def assert_owned_sources(payload, document_ids):
    sources = payload.get("sources")
    if not isinstance(sources, list):
        raise ValueError("Response has no source list")
    if any(not isinstance(source, dict) or source.get("document_id") not in document_ids for source in sources):
        raise ValueError("Response includes a foreign or unidentified document")


def build_settings(dsn, public_key, capacity, *, comparison_requests_per_minute=None):
    if capacity not in (8, 64):
        raise ValueError("Use default capacity 8 or explicit comparison capacity 64")

    def init_only(cls, settings_cls, init_settings, **sources):
        return (init_settings,)

    rate_budget = {}
    if comparison_requests_per_minute is not None:
        rate_budget["requests_per_minute"] = comparison_requests_per_minute
    # Avoid .env, account credentials and unrelated application environment settings.
    with patch.object(Settings, "settings_customise_sources", classmethod(init_only)):
        return Settings(_env_file=None, database_url=validate_target(dsn), vector_db_url=None,
            storage_mode="postgres", environment="test", run_migrations=False,
            auth_public_key=public_key, auth_issuer="forge-workload-test", auth_audience="forge-workload-api",
            openrouter_api_key=None, gemini_api_key=None, embedding_api_url=None, embedding_api_key=None,
            max_inflight_requests=capacity, embedding_workers=2, **rate_budget)


def loopback_network(stack, ports):
    """Permit only the run's HTTP listener, disposable PostgreSQL and stdlib control pipes."""
    def wrap(original):
        def connect(sock, address):
            frame = sys._getframe(1)
            control_pipe = frame.f_code.co_name in {"socketpair", "_fallback_socketpair"} and frame.f_code.co_filename == socket.__file__
            local = isinstance(address, tuple) and address[0] in {"127.0.0.1", "::1", "localhost"}
            if local and (control_pipe or address[1] in ports):
                return original(sock, address)
            raise RuntimeError("Workload runner blocks non-target network connections")
        return connect
    stack.enter_context(patch.object(socket.socket, "connect", wrap(socket.socket.connect)))
    stack.enter_context(patch.object(socket.socket, "connect_ex", wrap(socket.socket.connect_ex)))
    original_sync = httpx.HTTPTransport.handle_request
    original_async = httpx.AsyncHTTPTransport.handle_async_request

    def check_http(request):
        if request.url.host not in {"127.0.0.1", "localhost", "::1"} or request.url.port not in ports:
            raise RuntimeError("Workload runner blocks non-target HTTP and model downloads")

    def local_sync(transport, request):
        check_http(request)
        return original_sync(transport, request)

    async def local_async(transport, request):
        check_http(request)
        return await original_async(transport, request)

    stack.enter_context(patch.object(httpx.HTTPTransport, "handle_request", local_sync))
    stack.enter_context(patch.object(httpx.AsyncHTTPTransport, "handle_async_request", local_async))
    stack.enter_context(patch.dict(os.environ, {
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
    }))


@asynccontextmanager
async def isolated_http(settings, startup_timeout):
    """Own all resources; never point the application at a caller-supplied schema or HTTP URL."""
    schema = "forge_hub_eval_" + uuid.uuid4().hex
    connection = pool = server = task = None
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(256)
    listener.setblocking(False)
    port = listener.getsockname()[1]
    try:
        with ExitStack() as stack:
            loopback_network(stack, {port, urlsplit(settings.database_url).port or 5432})
            connection = await asyncpg.connect(validate_target(settings.database_url), timeout=10)
            for extension in ("pgcrypto", "vector"):
                await connection.execute(f'CREATE EXTENSION IF NOT EXISTS "{extension}" WITH SCHEMA public')
            await connection.execute(f'CREATE SCHEMA "{schema}"')
            pool = await asyncpg.create_pool(settings.database_url, min_size=1, max_size=10,
                server_settings={"search_path": f'"{schema}",public'})
            database = importlib.import_module("app.db")
            stack.enter_context(patch.object(database, "_pool", pool))
            stack.enter_context(patch.object(database, "_use_memory_fallback", False))
            for name in ("db", "documents", "sessions", "embeddings", "main"):
                module = importlib.import_module("app." + name)
                stack.enter_context(patch.object(module, "get_settings", lambda: settings))
            await database.ensure_schema()
            gating = importlib.import_module("app.gating")
            reranker = gating.get_reranker(settings.reranker_model, settings.reranker_max_length)
            observed_inference = Counter()
            original_rerank = reranker._rerank

            def observed_rerank(*args, **kwargs):
                # Called under the real reranker's lock; record the actual backend, not a replacement result.
                result = original_rerank(*args, **kwargs)
                observed_inference[reranker.backend] += 1
                return result

            stack.enter_context(patch.object(reranker, "_rerank", observed_rerank))
            app = importlib.import_module("app.main").create_app()
            app.dependency_overrides[get_settings] = lambda: settings
            server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port,
                loop="asyncio", http="h11", http2=False, ws="none", lifespan="on",
                proxy_headers=False, access_log=False, log_config=None, log_level="error",
                timeout_graceful_shutdown=10))
            task = asyncio.create_task(server.serve(sockets=[listener]))
            deadline = perf_counter() + startup_timeout
            while not server.started:
                if task.done():
                    await task
                    raise RuntimeError("HTTP startup failed; check local models and disposable database setup")
                if perf_counter() >= deadline:
                    raise TimeoutError("HTTP/model startup exceeded the configured deadline")
                await asyncio.sleep(0.05)
            try:
                async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", trust_env=False,
                    timeout=httpx.Timeout(300, connect=5, pool=5),
                    limits=httpx.Limits(max_connections=64, max_keepalive_connections=64)) as client:
                    ready = await client.get("/ready")
                    if ready.status_code != 200 or ready.json().get("storage") != "postgres":
                        raise RuntimeError("Actual model/PostgreSQL readiness did not pass")
                    embeddings = importlib.import_module("app.embeddings")
                    if not isinstance(embeddings._model, embeddings.SentenceTransformer):
                        raise RuntimeError("Real SentenceTransformer encoder is required")
                    yield client, pool, observed_inference
            finally:
                server.should_exit = True
                await asyncio.wait_for(asyncio.shield(task), timeout=20)
    finally:
        if task is not None and not task.done():
            if server is not None:
                server.force_exit = True
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        listener.close()
        if pool is not None:
            try:
                await asyncio.wait_for(pool.close(), timeout=10)
            except asyncio.TimeoutError:
                pool.terminate()
        if connection is not None:
            try:
                # Exact generated identifier only, as in benchmark_rag's isolated lifecycle.
                if not re.fullmatch(r"forge_hub_eval_[0-9a-f]{32}", schema):
                    raise RuntimeError("Unsafe generated schema identifier")
                await connection.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
            finally:
                await connection.close()


class TestIdentities:
    def __init__(self):
        self.key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self.public_key = self.key.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo).decode()

    def headers(self, subject):
        now = int(time())
        token = jwt.encode(dict(sub=subject, iss="forge-workload-test", aud="forge-workload-api",
            iat=now, exp=now + 3600), self.key, algorithm="RS256")
        return {"Authorization": "Bearer " + token}


async def upload(client, identities, subject, pages, name):
    headers = identities.headers(subject)
    created = await client.post("/api/sessions", headers=headers)
    if created.status_code != 200:
        raise RuntimeError(f"Session setup failed with HTTP {created.status_code}")
    sid = created.json()["session_id"]
    response = await client.post("/api/documents/upload", headers=headers,
        params={"session_id": sid}, files={"files": (name, generate_benchmark_pdf(pages), "application/pdf")})
    if response.status_code != 200:
        raise RuntimeError(f"Upload setup failed with HTTP {response.status_code}")
    document = response.json()["documents"][0]
    return {"subject": subject, "session_id": sid, "document_id": document["document_id"],
            "reported_chunks": document["chunks_created"]}


async def chat_once(client, identities, user, question, answer, gate=None):
    if gate is not None:
        await gate.wait()
    started = perf_counter()
    result = {"user": user["subject"], "status_code": None}
    try:
        response = await client.post("/api/chat", headers=identities.headers(user["subject"]),
            json={"session_id": user["session_id"], "query": question, "top_k": 5})
        result["status_code"] = response.status_code
        if response.status_code == 200:
            payload = response.json()
            try:
                assert_owned_sources(payload, {user["document_id"]})
                result["owned"] = True
            except ValueError:
                result["owned"] = False
            result.update(fact_correct=contains_fact(payload["answer"], answer) and not payload["refusal_triggered"],
                generation_backend=payload["generation_backend"], retrieval_query=payload.get("retrieval_query"),
                source_ids=[source["id"] for source in payload["sources"]],
                retrieval_ms=payload.get("retrieval_ms"), generation_ms=payload.get("generation_ms"))
    except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError) as error:
        result["error"] = type(error).__name__
    result["latency_ms"] = round((perf_counter() - started) * 1000, 3)
    return result


async def check_cross_owner(client, identities, users):
    checks = []
    for index, user in enumerate(users):
        other = users[(index + 1) % len(users)]
        for route in ("/api/documents", "/api/chat/history"):
            response = await client.get(route, headers=identities.headers(user["subject"]),
                                        params={"session_id": other["session_id"]})
            checks.append({"user": user["subject"], "route": route, "status_code": response.status_code,
                           "denied": response.status_code == 404})
    return checks


async def read_chat_stream(response, document_ids, answer, result, started):
    event = ""
    tokens = []
    metadata = complete = done = False
    async for line in response.aiter_lines():
        if line.startswith("event: "):
            event = line[7:]
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            done = True
            break
        payload = json.loads(data)
        if event in {"metadata", "complete"}:
            try:
                assert_owned_sources(payload, document_ids)
                if not payload["sources"]:
                    raise ValueError("Expected owned source evidence")
            except ValueError:
                result["owned"] = False
                raise
        if event == "metadata":
            metadata = True
        elif event == "complete":
            complete = (payload.get("complete") is True and not payload.get("refusal")
                        and payload.get("generation_backend") == "extractive_fallback")
            result["completion_ms"] = (perf_counter() - started) * 1000
        elif event == "error":
            raise ValueError("Stream reported an error")
        elif not event:
            token = payload["token"]
            if not metadata or not isinstance(token, str) or not token or complete:
                raise ValueError("Invalid stream token")
            if not tokens:
                result["ttft_ms"] = (perf_counter() - started) * 1000
            tokens.append(token)
        event = ""
    if not (tokens and complete and done):
        raise ValueError("Incomplete streamed response")
    result.update(owned=True, fact_correct=contains_fact("".join(tokens), answer))


def summarize_sustained(profile, records, elapsed):
    if elapsed <= 0 or not math.isfinite(elapsed):
        raise ValueError("Elapsed measurement time must be positive and finite")
    accepted = [r for r in records if r.get("status_code") == 200 and not r.get("error")
                and r.get("owned") is True and r.get("fact_correct") is True]
    statuses = Counter(r.get("status_code") for r in records)
    http_per_second = {}
    for r in records:
        if r.get("status_code") is not None:
            http_per_second.setdefault(r.get("started_second", 0), Counter())[r["status_code"] // 100] += 1
    metrics = {}
    for name, values, targets, required in (
        ("search_retrieval_ms", [r["retrieval_ms"] for r in accepted if r["operation"] == "search"],
         (120, 320, 480, 850), "search" in profile.operations),
        ("stream_first_token_ms", [r["ttft_ms"] for r in accepted if r["operation"] == "stream"],
         (220, 450, 650, 1200), "stream" in profile.operations),
        ("stream_completion_ms", [r["completion_ms"] for r in accepted if r["operation"] == "stream"],
         (1400, 2800, 3600, 5000), "stream" in profile.operations),
        ("http_5xx_percent", [100 * c[5] / sum(c.values()) for c in http_per_second.values()],
         (0.0, 0.01, 0.05, 0.10), True),
    ):
        measured = percentiles(values)
        target = dict(zip(("p50", "p90", "p95", "p99"), targets))
        status = "NOT MEASURED" if measured is None else (
            "PASS" if all(measured[p] <= limit for p, limit in target.items()) else "FAIL")
        metrics[name] = {"percentiles": measured, "targets": target, "required": required, "status": status}
    failures_5xx = sum(c[5] for c in http_per_second.values())
    http_responses = sum(sum(c.values()) for c in http_per_second.values())
    http_5xx_rate = 100 * failures_5xx / http_responses if http_responses else None
    throughput = len(accepted) / elapsed
    throughput_pass = throughput > profile.target
    return {
        "requests": len(records), "completed": len(accepted), "errors": len(records) - len(accepted),
        "status_counts": {str(status or "transport_error"): count for status, count in statuses.items()},
        "attempts_by_operation": dict(Counter(r["operation"] for r in records)),
        "completed_by_operation": dict(Counter(r["operation"] for r in accepted)),
        "operation_weights": dict(Counter(profile.operations)),
        "isolation_violations": sum(r.get("owned") is False for r in records),
        "completed_requests_per_second": round(throughput, 3), "throughput_target_exclusive": profile.target,
        "throughput_status": "PASS" if throughput_pass else "FAIL",
        "http_5xx_rate_percent": http_5xx_rate,
        "successful_http_latency_ms": percentiles([r.get("http_latency_ms", r["latency_ms"]) for r in accepted]),
        "all_attempt_latency_ms": percentiles([r["latency_ms"] for r in records]),
        "metrics": metrics,
        "status": "PASS" if (throughput_pass and http_5xx_rate is not None and http_5xx_rate <= 0.10
            and len(records) == len(accepted) + failures_5xx
            and all(m["status"] == "PASS" for m in metrics.values() if m["required"])) else "FAIL",
    }


async def sustained_request(client, pool, user, operation, started_second):
    started = perf_counter()
    result = {"operation": operation, "started_second": started_second, "status_code": None}
    sid, headers = user["session_id"], user["headers"]
    question = {"session_id": sid, "query": f"What recovery phrase belongs to the archive of {user['subject']}?"}

    def record_status(response):
        result["status_code"] = response.status_code
        result["http_latency_ms"] = (perf_counter() - started) * 1000
        retry_after = response.headers.get("retry-after", "")
        if response.status_code in (429, 503) and retry_after.isdigit() and len(retry_after) <= 3:
            result["retry_after_s"] = int(retry_after)

    try:
        async with asyncio.timeout(15):
            if operation == "stream":
                async with client.stream("POST", "/api/chat/stream",
                    headers={**headers, "Accept": "text/event-stream"}, json=question) as response:
                    record_status(response)
                    if response.status_code == 200:
                        if not response.headers.get("content-type", "").startswith("text/event-stream"):
                            raise ValueError("Expected event stream")
                        await read_chat_stream(response, {user["document_id"]}, user["answer"], result, started)
                        result["http_latency_ms"] = (perf_counter() - started) * 1000
            elif operation == "search":
                response = await client.post("/api/search/hybrid", headers=headers, json=question)
                record_status(response)
                if response.status_code == 200:
                    payload = response.json()
                    rows = payload["results"]
                    try:
                        assert_owned_sources({"sources": rows}, {user["document_id"]})
                    except ValueError:
                        result["owned"] = False
                        raise
                    result.update(owned=True, fact_correct=any(
                        contains_fact(row["content"], user["answer"]) for row in rows),
                        retrieval_ms=payload["retrieval_ms"])
            else:
                upload_session = user["upload_session"]
                response = await client.post("/api/documents/upload", headers=headers,
                    params={"session_id": upload_session},
                    files={"files": ("capacity.pdf", user["pdf"], "application/pdf")})
                record_status(response)
                if response.status_code == 200:
                    payload = response.json()
                    if payload["session_id"] != upload_session:
                        result["owned"] = False
                        raise ValueError("Unexpected owned upload session")
                    document, = payload["documents"]
                    document_id = uuid.UUID(document["document_id"])
                    async with pool.acquire() as connection:
                        async with connection.transaction():
                            vectors = await connection.fetchval(
                                "SELECT COUNT(*) FROM embeddings e JOIN chunks c ON c.id=e.chunk_id "
                                "JOIN documents d ON d.id=c.document_id WHERE d.id=$1 AND d.session_id=$2",
                                document_id, upload_session)
                            removed = await connection.fetchval(
                                "DELETE FROM documents WHERE id=$1 AND session_id=$2 RETURNING id",
                                document_id, upload_session)
                            if removed:
                                await connection.execute(
                                    "UPDATE sessions SET total_documents=GREATEST(total_documents-1, 0) WHERE session_id=$1",
                                    upload_session)
                    result.update(owned=removed == document_id,
                                  fact_correct=0 < document["chunks_created"] == vectors)
    except (httpx.HTTPError, TimeoutError, ValueError, KeyError, TypeError, AttributeError) as error:
        result["error"] = type(error).__name__
    result["latency_ms"] = round((perf_counter() - started) * 1000, 3)
    return result


async def sustained_workload(client, identities, pool, name):
    profile = SUSTAINED_PROFILES[name]
    users = []
    for index in range(profile.namespaces):
        subject = f"{name}-{index}"
        answer = f"violet heron {2000 + index}"
        pages = [f"For the archive of {subject}, the recovery phrase is {answer}."]
        upload_session = uuid.uuid4().hex
        user = {"subject": subject, "session_id": upload_session}
        if profile.operations != ("upload",):
            user = await upload(client, identities, subject, pages, subject + ".pdf")
        users.append(dict(user, headers=identities.headers(subject), answer=answer,
                          upload_session=upload_session, pdf=generate_benchmark_pdf(pages)))

    records = []
    next_operation = 0
    started = perf_counter()
    deadline = started + profile.seconds

    async def worker(user):
        nonlocal next_operation
        while perf_counter() < deadline:
            operation = profile.operations[next_operation % len(profile.operations)]
            next_operation += 1
            record = await sustained_request(client, pool, user, operation, int(perf_counter() - started))
            records.append(record)
            delay = min(record.get("retry_after_s", 0), deadline - perf_counter())
            if delay > 0:
                await asyncio.sleep(delay)

    await asyncio.gather(*(worker(users[i % len(users)]) for i in range(profile.users)))
    elapsed = perf_counter() - started
    summary = summarize_sustained(profile, records, elapsed)
    probe_headers = identities.headers("sustained-independent-owner")
    blocked = []
    session_keys = ("session_id", "upload_session") if "upload" in profile.operations else ("session_id",)
    scope = dict.fromkeys(user[key] for user in users for key in session_keys)
    for session_id in scope:
        response = await client.get("/api/documents", headers=probe_headers, params={"session_id": session_id})
        blocked.append(response.status_code)
    return {
        "status": "PASS" if summary["status"] == "PASS" and all(code == 404 for code in blocked) else "FAIL",
        "concurrent_workers": profile.users, "session_namespaces": profile.namespaces,
        "launch_duration_s": profile.seconds, "elapsed_with_drain_s": elapsed,
        "throughput_unit": "documents/second" if profile.operations == ("upload",) else "requests/second",
        "summary": summary, "cross_owner_status_codes": blocked,
        "measurement_notes": [
            "HTTP 5xx percentiles use one-second request-start buckets with HTTP responses.",
            "Upload completion checks persisted embeddings and cleans up in the disposable schema without extra HTTP requests.",
            "Preparation is outside the timed interval. Workers honor Retry-After and time out after 15 seconds per attempt.",
        ],
    }


async def load_workload(client, identities, count, rounds, budget):
    users = []
    for i in range(count):
        subject = f"load-{count}-{i:02d}"
        answer = f"violet heron {1000 + i}"
        user = await upload(client, identities, subject,
            [f"For the private archive of operator {subject}, the recovery phrase is {answer}."], subject + ".pdf")
        user["answer"] = answer
        users.append(user)
    records, wave_times = [], []
    for round_index in range(rounds):
        gate = asyncio.Event()
        tasks = [asyncio.create_task(chat_once(client, identities, user,
            f"What is the recovery phrase for the private archive of operator {user['subject']}?",
            user["answer"], gate)) for user in users]
        started = perf_counter()
        gate.set()
        wave = await asyncio.gather(*tasks)
        wave_times.append(perf_counter() - started)
        records.extend(dict(record, round=round_index + 1) for record in wave)
    cross_owner = await check_cross_owner(client, identities, users)
    summary = summarize_requests(records, sum(wave_times), budget)
    summary["cross_owner_status"] = "PASS" if all(row["denied"] for row in cross_owner) else "FAIL"
    if summary["cross_owner_status"] == "FAIL":
        summary["status"] = "FAIL"
    return {"concurrent_users": count, "rounds": rounds, "summary": summary,
            "requests": records, "cross_owner_checks": cross_owner, "timed_wave_seconds": wave_times,
            "setup": "Sequential real HTTP uploads outside the timed barrier-synchronized chat waves"}


async def needle_workload(client, identities, pool):
    pages, gold = needle_corpus()
    user = await upload(client, identities, "needle-owner", pages, "needle-500.pdf")
    async with pool.acquire() as connection:
        rows = await connection.fetch(
            "SELECT chunk_index,page_number,content_hash,content FROM chunks WHERE document_id=$1 ORDER BY chunk_index",
            uuid.UUID(user["document_id"]))
        vectors = await connection.fetchval(
            "SELECT COUNT(*) FROM embeddings e JOIN chunks c ON c.id=e.chunk_id WHERE c.document_id=$1",
            uuid.UUID(user["document_id"]))
    corpus = corpus_layout(rows, vectors)
    if corpus["layout_status"] != "PASS":
        return {"status": "FAIL", "corpus": corpus, "cases": [], "reason": "Actual corpus layout differs from the declared 500-chunk workload"}
    cases = []
    for fact in gold:
        row = rows[fact["chunk_index"]]
        if not contains_fact(row["content"], fact["answer"]) or not contains_fact(row["content"], fact["subject"]):
            return {"status": "FAIL", "corpus": corpus, "cases": cases, "reason": "Needle is not at the declared actual chunk position"}
        question = f"What is the approved current recovery phrase for the {fact['subject']} archive?"
        response = await client.post("/api/search/hybrid", headers=identities.headers(user["subject"]),
            json={"session_id": user["session_id"], "query": question, "top_k": 5})
        raw = response.json().get("results", []) if response.status_code == 200 else []
        owned = all(item.get("document_id") == user["document_id"] for item in raw)
        hit = any(item.get("page_number") == fact["page_number"] and contains_fact(item.get("content", ""), fact["answer"]) for item in raw)
        answer = await chat_once(client, identities, user, question, fact["answer"])
        passed = (response.status_code == 200 and owned and hit and answer.get("owned") is True
                  and answer.get("fact_correct") is True and answer.get("generation_backend") == "extractive_fallback")
        cases.append({**fact, "actual_depth": fact["chunk_index"] / (len(rows) - 1),
            "search_status": response.status_code, "rrf_recall_at_5": hit, "search_owned": owned,
            "chat": answer, "status": "PASS" if passed else "FAIL"})
    return {"status": "PASS" if all(c["status"] == "PASS" for c in cases) else "FAIL", "corpus": corpus, "cases": cases,
            "scope": "500 real encoded unique chunks with 15 conflicting draft facts; local extractive answers"}


async def multiturn_workload(client, identities):
    users, turns = [], []
    for archive_index, (subject, retention, phrase) in enumerate(REFERENCE_FACTS):
        invoice_price = f"{51 + archive_index} crowns"
        user = await upload(client, identities, "reference-" + subject,
            [f"The {subject} archive retains audit logs for {retention}. Its recovery phrase is {phrase}.",
             f"This invoice costs {invoice_price} for a pressure regulator."], subject + ".pdf")
        users.append(user)
        questions = [(f"How many days does the {subject} archive retain audit logs?", retention),
                     ("What is its recovery phrase?", phrase),
                     ("And how many days does it retain them?", retention),
                     ("What does this invoice cost?", invoice_price)]
        for index, (question, expected) in enumerate(questions):
            record = await chat_once(client, identities, user, question, expected)
            resolved = record.get("retrieval_query") or ""
            unrelated = index == len(questions) - 1
            contextualized = (resolved == question if unrelated or index == 0 else subject.lower() in resolved.lower())
            turns.append({"subject": subject, "turn": index + 1, "expected": expected,
                          "contextualized": contextualized, "unrelated_topic": unrelated, **record})
        response = await client.get("/api/chat/history", headers=identities.headers(user["subject"]),
                                    params={"session_id": user["session_id"]})
        history = response.json().get("messages", []) if response.status_code == 200 else []
        user["history_complete"] = (response.status_code == 200 and len(history) == len(questions)
            and {row.get("user_message") for row in history} == {question for question, _ in questions}
            and all(row.get("session_id") == user["session_id"]
                and all(source.get("document_id") == user["document_id"] for source in row.get("source_documents", []))
                for row in history))
    checks = await check_cross_owner(client, identities, users)
    passed = (len(turns) == 8 and all(r.get("fact_correct") is True and r.get("owned") is True
        and r["contextualized"] and r.get("generation_backend") == "extractive_fallback" for r in turns)
        and all(user["history_complete"] for user in users) and all(row["denied"] for row in checks))
    return {"status": "PASS" if passed else "FAIL", "turns": turns,
            "history_complete": all(user["history_complete"] for user in users), "cross_owner_checks": checks,
            "answer_quality": "Synthetic reference resolution + extractive fact checks; semantic synthesis/faithfulness NOT MEASURED"}


async def run(args):
    validate_target(args.database_url)
    identities = TestIdentities()
    settings = build_settings(args.database_url, identities.public_key, args.capacity,
                              comparison_requests_per_minute=args.comparison_requests_per_minute)
    output = Path(args.output_dir).resolve() if args.output_dir else Path(tempfile.mkdtemp(prefix="forge-workloads-"))
    output.mkdir(parents=True, exist_ok=True)
    selected = {"all": WORKLOADS, "sustained-all": tuple(SUSTAINED_PROFILES)}.get(args.workload, (args.workload,))
    report = {"timestamp": datetime.now(timezone.utc).isoformat(), "selected_workloads": list(selected),
        "configuration": {"max_inflight_requests": settings.max_inflight_requests,
            "capacity_profile": "default" if args.capacity == 8 else "nondefault comparison",
            "requests_per_minute_per_owner": settings.requests_per_minute,
            "rate_budget_profile": "default" if args.comparison_requests_per_minute is None else "synthetic comparison",
            "postgres_pool_max": 10, "embedding_workers": 2, "http_transport": "real loopback TCP/Uvicorn h11",
            "auth": "Synthetic RS256 issuer + per-user signed subjects", "environment": "test",
            "embedding_model": settings.embedding_model, "reranker_model": settings.reranker_model,
            "provider_keys": "disabled", "models": "Real local encoder + FlashRank warmup required"},
        "limitations": ["No external LLM calls; extractive fallback is not semantic answer-quality evidence",
            "Admission saturation and per-owner rate rejection return HTTP 429 and count as failed requests.",
            "The 50-worker read profile uses ten signed owners, so the default owner budget constrains its throughput.",
            "Loopback synthetic test identities and operator-created disposable schema do not establish production IdP/runtime-role readiness"],
        "workloads": {}, "backend_release_gate": "INCOMPLETE"}
    try:
        async with isolated_http(settings, args.startup_timeout) as (client, pool, inference):
            for name in selected:
                before = inference.copy()
                print(f"Running {name} with capacity={args.capacity}, pool=10, workers=2", flush=True)
                if name.startswith("load"):
                    result = await load_workload(client, identities, int(name[4:]), args.rounds, args.p95_budget_ms)
                elif name in SUSTAINED_PROFILES:
                    result = await sustained_workload(client, identities, pool, name)
                elif name == "needle":
                    result = await needle_workload(client, identities, pool)
                else:
                    result = await multiturn_workload(client, identities)
                observed = dict(inference - before)
                result["observed_reranker_calls"] = observed
                requires_rerank = name not in {"sustained-read", "sustained-upload"}
                if (requires_rerank and not observed.get("flashrank")) or any(key != "flashrank" for key in observed):
                    result["status"] = "FAIL"
                    result["model_status"] = "Neural inference missing or a heuristic fallback was observed"
                report["workloads"][name] = result
                print(f"{name}: {result.get('status', result.get('summary', {}).get('status'))}", flush=True)
    except Exception as error:
        report["execution_error"] = type(error).__name__
        print(f"Workload stopped: {type(error).__name__}; see the saved partial report", flush=True)
    report["status"] = "PASS" if (not report.get("execution_error") and len(report["workloads"]) == len(selected)
        and all(r.get("status", r.get("summary", {}).get("status")) == "PASS" for r in report["workloads"].values())) else "FAIL"
    path = output / "workload_results.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Results: {path}", flush=True)
    return 0 if report["status"] == "PASS" else 2


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--database-url", default=os.environ.get("FORGE_HUB_TEST_DATABASE_URL", ""),
                        help="Explicit loopback forge_hub_test[_suffix] database only; never reads application DSNs")
    result.add_argument("--workload", choices=("all",) + WORKLOADS + ("sustained-all",) + tuple(SUSTAINED_PROFILES),
                        default="all", help="All wave/needle/multiturn modes, or opt-in sustained profiles")
    result.add_argument("--capacity", type=int, choices=(8, 64), default=8,
                        help="8: application default; 64: explicitly labeled nondefault admission comparison")
    result.add_argument("--comparison-requests-per-minute", type=int,
                        help="Nondefault per-owner budget for disposable synthetic SLO comparisons only")
    result.add_argument("--rounds", type=int, default=3, help="1-10 synchronized load waves")
    result.add_argument("--startup-timeout", type=float, default=180)
    result.add_argument("--p95-budget-ms", type=float, help="Optional strict successful-request p95 budget; otherwise NOT MEASURED")
    result.add_argument("--output-dir", help="New report directory; defaults to a unique temporary directory")
    return result


if __name__ == "__main__":
    logging.getLogger("httpx").setLevel(logging.WARNING)
    cli = parser()
    arguments = cli.parse_args()
    try:
        validate_target(arguments.database_url)
        if not 1 <= arguments.rounds <= 10 or not 1 <= arguments.startup_timeout <= 600:
            raise ValueError("Use 1-10 rounds and a 1-600 second startup deadline")
        if arguments.p95_budget_ms is not None and (not math.isfinite(arguments.p95_budget_ms) or arguments.p95_budget_ms <= 0):
            raise ValueError("p95 budget must be positive and finite")
        if arguments.comparison_requests_per_minute is not None and not 1 <= arguments.comparison_requests_per_minute <= 10000:
            raise ValueError("Comparison requests per minute must be between 1 and 10000")
    except ValueError as error:
        cli.error(str(error))
    raise SystemExit(asyncio.run(run(arguments)))
