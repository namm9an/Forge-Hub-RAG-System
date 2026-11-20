#!/usr/bin/env python3
"""
Forge Hub RAG System — Automated Evaluation & Benchmark Harness
Complies with:
- Rule E-1: Standalone runnable CLI via 'python scripts/benchmark_rag.py'
- Rule E-2: Empirical Latency Profiling (p50, p90, p95, p99)
- Rule E-3: Retrieval Recall@5, Context Precision, and Hallucination Rejection Rate
- Rule E-4: Reproducible Artifacts (benchmark_results.json, benchmark_report.md)
"""

import sys
import os
import io
import re
import json
import asyncio
import argparse
import tempfile
import uuid
from contextlib import ExitStack
from datetime import datetime, timezone
from urllib.parse import urlsplit, unquote
from unittest.mock import patch
from pathlib import Path
from time import perf_counter
from typing import List, Dict, Any, Tuple
import numpy as np

# Ensure backend package is in python search path
REPO_ROOT = Path(__file__).resolve().parent.parent
candidates = [
    REPO_ROOT / "apps" / "backend",
    REPO_ROOT / "advanced-rag-chatbot" / "apps" / "backend",
]
BACKEND_DIR = next((d for d in candidates if d.is_dir()), REPO_ROOT / "apps" / "backend")
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import Settings
from app.auth import Principal
from app.chat import chat
from app.schemas import ChatRequest
from app.gating import get_reranker
from app.gating import REFUSAL_MESSAGE
from app.schemas import SearchResultItem

# Multi-Page Technical PDF Dataset
TECHNICAL_PAGES = [
    # Page 1: Storage Engines & Write-Ahead Logging
    """
    Storage Engines and Write-Ahead Logging (WAL) Architecture:
    Write-ahead logging guarantees durability and crash recovery by sequentially recording all state modifications to persistent storage before writing dirty pages to table data files. In ACID transactions, write-ahead logging ensures atomicity and durability by allowing recovery managers to replay committed transactions through REDO log records and rollback uncommitted transactions through UNDO records during startup.
    Log-Structured Merge (LSM) trees optimize write performance by transforming random disk writes into sequential append-only disk operations. The LSM write path buffers incoming writes into an in-memory sorted data structure known as a memtable, typically implemented as a skiplist or concurrent red-black tree.
    When a memtable reaches its capacity threshold, it is flushed to disk as an immutable Sorted String Table (SSTable). Bloom filters are probabilistic bit arrays stored in SSTable headers that allow point lookups to determine with zero false negatives whether a key does not exist in an SSTable, preventing unnecessary disk I/O. Background compaction jobs periodically merge overlapping SSTables, discard obsolete tombstones, and minimize read amplification.
    """,

    # Page 2: Vector Search & High-Dimensional Indexing
    """
    High-Dimensional Vector Search and Indexing Structures:
    Approximate Nearest Neighbor (ANN) search enables sub-linear similarity search across high-dimensional vector spaces. Hierarchical Navigable Small World (HNSW) graphs construct a multi-layer geometric graph structure inspired by skip lists to achieve logarithmic search complexity. Upper HNSW layers contain long-range skip connections for rapid coarse routing across vector clusters, while the bottom layer contains dense local neighbor connections for fine-grained nearest neighbor convergence.
    Inverted File with Flat Quantization (IVFFlat) indexes partition the vector space into Voronoi cells using k-means centroid clustering. During query execution, IVFFlat searches only the centroids closest to the query vector governed by the nprobe parameter. While IVFFlat maintains a smaller memory footprint and rapid build time compared to HNSW, HNSW consistently achieves higher recall and lower latency at the expense of substantial RAM consumption. Normalized vector cosine similarity is computed as the dot product between unit vectors, where one minus cosine distance represents angular proximity.
    """,

    # Page 3: Consensus Protocols & Distributed State
    """
    Distributed Consensus Protocols and State Machine Replication:
    Distributed consensus protocols ensure fault-tolerant state machine replication across a cluster of independent nodes. The Raft consensus algorithm decomposes state machine replication into leader election, log replication, and safety guarantees. Raft leader election utilizes randomized election timeouts between 150 milliseconds and 300 milliseconds to prevent split-vote scenarios where multiple candidates simultaneously claim majority votes.
    Term numbers act as logical clocks in Raft, allowing servers to detect obsolete leaders and stale RPC requests. A Raft leader commits an entry once a majority quorum of followers has replicated the log entry to non-volatile disk. Paxos achieves consensus through a two-phase protocol involving Prepare and Promise phases followed by Accept and Accepted phases across proposers and acceptors. Byzantine Fault Tolerance handles malicious or arbitrary node failures requiring a three f plus one supermajority, whereas crash fault-tolerant protocols like Raft and Paxos require two f plus one nodes.
    """,

    # Page 4: Reciprocal Rank Fusion & Hybrid Retrieval
    """
    Reciprocal Rank Fusion and Two-Stage Hybrid Retrieval:
    Modern retrieval architectures employ two-stage hybrid search combining dense vector embeddings with sparse lexical indexing. Dense semantic retrieval captures conceptual meaning and synonymy via transformer encoders such as sentence-transformers all-mpnet-base-v2. Sparse lexical search utilizes inverted index frequency ranking such as BM25 or PostgreSQL tsvector with ts_rank_cd to match exact identifiers, technical acronyms, and rare terms.
    Reciprocal Rank Fusion unifies candidate rankings without requiring calibration across incompatible score distributions. The mathematical formula for Reciprocal Rank Fusion is RRF score equals the sum across rankers of ranker weight divided by the sum of smoothing constant k and the candidate rank in that ranker. The smoothing constant k is set to 60 to prevent high-ranking outliers from dominating candidates that appear consistently across multiple retrieval methods. Typical weight configurations assign dense weight equal to 0.6 and sparse weight equal to 0.4 to balance semantic generalization with exact keyword precision.
    """,

    # Page 5: Query Optimization and Cache Eviction Policies
    """
    Query Optimization and Buffer Pool Cache Policies:
    Relational database cost-based optimizers evaluate alternative execution plans by estimating I/O and CPU cost from table statistics. Equi-depth histograms and hyperloglog data structures maintain column cardinality estimates and distinct value frequencies. Buffer pool managers maintain page caches in shared memory to minimize disk read latency.
    Least Recently Used (LRU) eviction discards the oldest unaccessed pages but suffers from cache pollution during full sequential table scans. Adaptive Replacement Cache (ARC) dynamically tunes cache allocation between recency and frequency using two ghost lists to adapt to shifting query workloads. Asynchronous checkpointing flushes dirty buffer pages to persistent storage while maintaining consistency with write-ahead logs.
    """,
]

# Benchmark Query Suites (24 queries across 3 suites)
BENCHMARK_SUITES = [
    # Suite 1: Factual Retrieval (8 queries)
    {
        "suite": "Factual",
        "queries": [
            {
                "id": "FACT-01",
                "query": "How does write-ahead logging guarantee durability in database storage engines?",
                "expected_page": 1,
                "keywords": ["write-ahead logging", "durability", "redo", "persistent"],
            },
            {
                "id": "FACT-02",
                "query": "What data structures compose a Log-Structured Merge (LSM) tree?",
                "expected_page": 1,
                "keywords": ["memtable", "sstable", "skiplist", "sorted string table"],
            },
            {
                "id": "FACT-03",
                "query": "What is the function of Bloom filters in SSTable read paths?",
                "expected_page": 1,
                "keywords": ["bloom filters", "probabilistic", "zero false negatives", "disk i/o"],
            },
            {
                "id": "FACT-04",
                "query": "How does an HNSW index achieve logarithmic search complexity for vector embeddings?",
                "expected_page": 2,
                "keywords": ["hnsw", "hierarchical", "skip", "logarithmic", "layers"],
            },
            {
                "id": "FACT-05",
                "query": "What trade-offs distinguish IVFFlat indexes from HNSW graphs?",
                "expected_page": 2,
                "keywords": ["ivfflat", "memory footprint", "recall", "ram", "voronoi"],
            },
            {
                "id": "FACT-06",
                "query": "What mechanism prevents split-vote scenarios during Raft leader election?",
                "expected_page": 3,
                "keywords": ["randomized election timeouts", "raft", "split-vote", "leader election"],
            },
            {
                "id": "FACT-07",
                "query": "What is the formula used in Reciprocal Rank Fusion (RRF) for ranking candidates?",
                "expected_page": 4,
                "keywords": ["reciprocal rank fusion", "rrf", "weight", "smoothing constant"],
            },
            {
                "id": "FACT-08",
                "query": "What does the smoothing constant k=60 accomplish in Reciprocal Rank Fusion?",
                "expected_page": 4,
                "keywords": ["smoothing constant", "outliers", "dominating", "k", "60"],
            },
        ],
    },

    # Suite 2: Multi-Chunk Syntheses (8 queries)
    {
        "suite": "Multi-Chunk",
        "queries": [
            {
                "id": "MULTI-01",
                "query": "Compare the write path of LSM trees with the durability mechanism of Write-Ahead Logging.",
                "expected_page": 1,
                "keywords": ["lsm", "memtable", "write-ahead logging", "durability", "sstable"],
            },
            {
                "id": "MULTI-02",
                "query": "How do vector indexing strategies like IVFFlat differ from traditional lexical indexing with BM25?",
                "expected_page": 2,
                "keywords": ["ivfflat", "bm25", "vector", "lexical", "indexing"],
            },
            {
                "id": "MULTI-03",
                "query": "Explain how consensus in Raft relates to persistent log replication across cluster nodes.",
                "expected_page": 3,
                "keywords": ["raft", "consensus", "log replication", "quorum", "majority"],
            },
            {
                "id": "MULTI-04",
                "query": "How does hybrid search combine dense vector cosine similarity with sparse keyword rankings?",
                "expected_page": 4,
                "keywords": ["hybrid search", "dense", "sparse", "cosine", "rrf"],
            },
            {
                "id": "MULTI-05",
                "query": "What are the computational trade-offs between memory footprint in HNSW and disk I/O in SSTables?",
                "expected_page": 2,
                "keywords": ["hnsw", "sstable", "memory", "disk", "trade-offs"],
            },
            {
                "id": "MULTI-06",
                "query": "Describe how query optimization histograms interact with storage engine indexes.",
                "expected_page": 5,
                "keywords": ["query optimization", "histograms", "cost-based", "cardinality"],
            },
            {
                "id": "MULTI-07",
                "query": "How does the combination of dense embeddings and lexical search address both vocabulary mismatch and exact keyword precision?",
                "expected_page": 4,
                "keywords": ["dense embeddings", "lexical search", "vocabulary mismatch", "keyword precision"],
            },
            {
                "id": "MULTI-08",
                "query": "How does crash recovery utilize WAL logs in conjunction with storage engine checkpoints?",
                "expected_page": 1,
                "keywords": ["crash recovery", "wal", "checkpoints", "redo", "storage engine"],
            },
        ],
    },

    # Suite 3: Out-of-Domain Traps (8 queries)
    {
        "suite": "Out-of-Domain Traps",
        "queries": [
            {
                "id": "TRAP-01",
                "query": "What is the traditional recipe and fermentation time for sourdough pizza dough?",
                "is_trap": True,
            },
            {
                "id": "TRAP-02",
                "query": "Who was the top goal scorer in the 1994 FIFA World Cup soccer tournament?",
                "is_trap": True,
            },
            {
                "id": "TRAP-03",
                "query": "What are the clinical treatments for seasonal allergic rhinitis in adults?",
                "is_trap": True,
            },
            {
                "id": "TRAP-04",
                "query": "How do you torque the crankshaft bearings on a small-block V8 automotive engine?",
                "is_trap": True,
            },
            {
                "id": "TRAP-05",
                "query": "What cinematic techniques did Alfred Hitchcock use to build suspense in Rear Window?",
                "is_trap": True,
            },
            {
                "id": "TRAP-06",
                "query": "What are the optimal soil pH levels and fertilizer ratios for hybrid tea rose cultivars?",
                "is_trap": True,
            },
            {
                "id": "TRAP-07",
                "query": "How did the Apollo 11 command module heat shield survive atmospheric reentry temperatures?",
                "is_trap": True,
            },
            {
                "id": "TRAP-08",
                "query": "What are the official scoring regulations and service rules in Olympic table tennis?",
                "is_trap": True,
            },
        ],
    },
]


def _escape_pdf_text(text: str) -> str:
    """Escapes string characters for PDF stream formatting."""
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def generate_benchmark_pdf(pages_text: List[str]) -> bytes:
    """
    Generates a valid multi-page PDF 1.4 document in pure Python.
    Complies with PDF-1.4 spec, directly readable by pdfplumber and pypdf.
    """
    pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets: List[int] = []

    def add_obj(body: bytes) -> int:
        offsets.append(len(pdf))
        obj_num = len(offsets)
        pdf.extend(f"{obj_num} 0 obj\n".encode("latin1"))
        pdf.extend(body)
        pdf.extend(b"\nendobj\n")
        return obj_num

    num_pages = len(pages_text)

    # 1. Catalog Object (ID 1)
    # References Pages object (ID 2)
    add_obj(b"<< /Type /Catalog /Pages 2 0 R >>")

    # 2. Pages Object (ID 2) placeholder - we write it after assigning Page IDs
    # Page IDs: for page i, page_obj = 4 + 2*i, stream_obj = 4 + 2*i + 1
    page_obj_ids = [4 + 2 * i for i in range(num_pages)]
    kids_str = " ".join(f"{pid} 0 R" for pid in page_obj_ids)
    pages_body = f"<< /Type /Pages /Kids [{kids_str}] /Count {num_pages} >>".encode("latin1")
    add_obj(pages_body)

    # 3. Font Object (ID 3)
    add_obj(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    # 4. Write Page & Content Stream pairs
    for idx, page_content in enumerate(pages_text):
        content_id = 4 + 2 * idx + 1

        # Format stream commands
        clean_text = " ".join(page_content.split())
        words = clean_text.split()
        lines: List[str] = []
        cur_line: List[str] = []
        cur_len = 0
        for w in words:
            if cur_len + len(w) + 1 > 75 and cur_line:
                lines.append(" ".join(cur_line))
                cur_line = [w]
                cur_len = len(w)
            else:
                cur_line.append(w)
                cur_len += len(w) + 1
        if cur_line:
            lines.append(" ".join(cur_line))

        stream_cmds = ["BT", "/F1 10 Tf", "50 720 Td", "14 TL"]
        for line in lines:
            escaped = _escape_pdf_text(line)
            stream_cmds.append(f"({escaped}) Tj T*")
        stream_cmds.append("ET")
        stream_bytes = "\n".join(stream_cmds).encode("latin1")

        # Page object
        page_dict = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> "
            f"/Contents {content_id} 0 R >>"
        ).encode("latin1")
        add_obj(page_dict)

        # Content Stream object
        stream_obj_body = (
            f"<< /Length {len(stream_bytes)} >>\nstream\n".encode("latin1")
            + stream_bytes
            + b"\nendstream"
        )
        add_obj(stream_obj_body)

    # Cross-Reference (xref) table
    xref_offset = len(pdf)
    pdf.extend(b"xref\n")
    pdf.extend(f"0 {len(offsets) + 1}\n".encode("latin1"))
    pdf.extend(b"0000000000 65535 f \n")
    for off in offsets:
        pdf.extend(f"{off:010d} 00000 n \n".encode("latin1"))

    # Trailer
    pdf.extend(
        f"trailer\n<< /Size {len(offsets) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("latin1")
    )
    return bytes(pdf)


async def ingest_benchmark_dataset(session_id: str, settings) -> Tuple[str, int]:
    """Use the application's complete ingestion path, including atomic embedding writes."""
    from fastapi import UploadFile
    from starlette.datastructures import Headers
    from app.documents import upload_documents
    response = await upload_documents(
        files=[UploadFile(io.BytesIO(generate_benchmark_pdf(TECHNICAL_PAGES)), filename="distributed_database_architecture.pdf",
                          headers=Headers({"content-type": "application/pdf"}))],
        session_id_query=session_id, session_id_form=None, x_session_id=None,
        settings=settings, principal=Principal("benchmark-owner"),
    )
    document = response["documents"][0]
    return document["document_id"], document["chunks_created"]


def threshold_status(value, target, *, less_than=False):
    if value is None:
        return "NOT MEASURED"
    return "PASS" if (value < target if less_than else value >= target) else "FAIL"


async def run_benchmark(settings, session_id, output_dir):

    print("================================================================================")
    print(" Forge Hub RAG System — Application-Pipeline Diagnostic Benchmark")
    print("================================================================================")

    # 1. Ingest Dataset
    t_ingest_start = perf_counter()
    doc_id, total_chunks = await ingest_benchmark_dataset(session_id, settings)
    ingest_time_s = round(perf_counter() - t_ingest_start, 2)

    # 2. Execute Evaluation Queries across 3 Suites
    print(f"\n[2/4] Executing 24 benchmark queries across 3 evaluation suites...")
    retrieval_latencies: List[float] = []
    generation_latencies: List[float] = []
    total_latencies: List[float] = []

    recall_hits = 0
    total_factual_queries = 0
    precision_scores: List[float] = []
    traps_rejected = 0
    total_trap_queries = 0

    query_results_log: List[Dict[str, Any]] = []
    generation_backends = set()
    reranker_backends = set()

    for suite_info in BENCHMARK_SUITES:
        suite_name = suite_info["suite"]
        queries = suite_info["queries"]
        print(f"\n   Running Suite: '{suite_name}' ({len(queries)} queries)...")

        for q_entry in queries:
            qid = q_entry["id"]
            query_text = q_entry["query"]
            is_trap = q_entry.get("is_trap", False)
            expected_page = q_entry.get("expected_page")
            keywords = q_entry.get("keywords", [])

            t_q_start = perf_counter()

            result = await chat(
                ChatRequest(query=query_text, session_id=session_id, top_k=5, threshold=settings.similarity_threshold),
                settings=settings, principal=Principal("benchmark-owner"),
            )
            r_ms = result.retrieval_ms or 0
            g_ms = result.generation_ms or 0
            retrieval_latencies.append(r_ms)
            should_refuse, confidence, answer = result.refusal_triggered, result.confidence, result.answer
            retained_sources = result.sources
            candidates = [SearchResultItem(
                chunk_id=source.id, document_id=source.document_id or "", filename=source.filename or "",
                page_number=source.page_number or 1, score=source.score or 0, content=source.content or "",
                dense_score=source.dense_score, sparse_score=source.sparse_score,
                rerank_score=source.rerank_score,
            ) for source in retained_sources]
            generation_backends.add(result.generation_backend)
            reranker_backends.add(get_reranker(settings.reranker_model).backend)

            generation_latencies.append(g_ms)
            tot_ms = (perf_counter() - t_q_start) * 1000.0
            total_latencies.append(tot_ms)

            # D. Scoring Metrics (Recall, Precision, Trap Rejection)
            hit = False
            prec = 0.0

            if is_trap:
                total_trap_queries += 1
                if should_refuse and REFUSAL_MESSAGE in answer:
                    traps_rejected += 1
                    status_str = "REJECTED (PASS)"
                else:
                    status_str = "FAILED (HALLUCINATION RISK)"
            else:
                total_factual_queries += 1
                # Check Recall@5: Did candidates include expected page or keywords?
                retrieved_pages = [c.page_number for c in candidates]
                retrieved_texts = " ".join(c.content.lower() for c in candidates)

                page_match = expected_page in retrieved_pages if expected_page else False
                kw_match = any(kw.lower() in retrieved_texts for kw in keywords)

                if page_match or kw_match:
                    recall_hits += 1
                    hit = True

                # Context Precision: fraction of top-5 chunks relevant
                relevant_chunks = 0
                for c in candidates:
                    if (expected_page and c.page_number == expected_page) or any(k.lower() in c.content.lower() for k in keywords):
                        relevant_chunks += 1
                prec = relevant_chunks / max(1, len(candidates))
                precision_scores.append(prec)
                status_str = f"HIT (P={prec:.2f})" if hit else "MISS"

            print(f"      [{qid}] {query_text[:50]}... -> {status_str} (r={r_ms:.1f}ms, tot={tot_ms:.1f}ms)")

            query_results_log.append({
                "id": qid,
                "suite": suite_name,
                "query": query_text,
                "is_trap": is_trap,
                "refusal_triggered": should_refuse,
                "confidence": confidence.value if hasattr(confidence, "value") else str(confidence),
                "retrieval_ms": round(r_ms, 2),
                "generation_ms": round(g_ms, 2),
                "total_ms": round(tot_ms, 2),
                "hit": hit,
                "precision": round(prec, 3),
                "top_dense_score": candidates[0].dense_score if candidates else None,
                "top_sparse_score": candidates[0].sparse_score if candidates else None,
            })

    # 3. Aggregate Metrics & Percentile Calculation (Rule E-2 & E-3)
    print("\n[3/4] Computing empirical percentile distributions and accuracy metrics...")
    r_p50 = float(np.percentile(retrieval_latencies, 50))
    r_p90 = float(np.percentile(retrieval_latencies, 90))
    r_p95 = float(np.percentile(retrieval_latencies, 95))
    r_p99 = float(np.percentile(retrieval_latencies, 99))

    g_active = [g for g in generation_latencies if g > 0]
    g_p50 = float(np.percentile(g_active, 50)) if g_active else 0.0
    g_p90 = float(np.percentile(g_active, 90)) if g_active else 0.0
    g_p95 = float(np.percentile(g_active, 95)) if g_active else 0.0
    g_p99 = float(np.percentile(g_active, 99)) if g_active else 0.0

    tot_p50 = float(np.percentile(total_latencies, 50))
    tot_p90 = float(np.percentile(total_latencies, 90))
    tot_p95 = float(np.percentile(total_latencies, 95))
    tot_p99 = float(np.percentile(total_latencies, 99))

    context_recall_5 = (recall_hits / max(1, total_factual_queries)) * 100.0
    avg_precision = (sum(precision_scores) / max(1, len(precision_scores))) * 100.0
    rejection_rate = (traps_rejected / max(1, total_trap_queries)) * 100.0

    metrics_summary = {
        "benchmark_timestamp": datetime.now(timezone.utc).isoformat(),
        "generation_backends": sorted(generation_backends),
        "reranker_backends": sorted(reranker_backends),
        "measurement_scope": "Direct application handlers with real ingestion/retrieval/rerank/gating/history; excludes HTTP/auth/network overhead",
        "quality_limitations": "Synthetic page/keyword relevance proxies; semantic faithfulness and answer relevance are NOT MEASURED",
        "acceptance": {
            "context_hit_proxy": threshold_status(context_recall_5, 85.0),
            "context_precision_proxy": threshold_status(avg_precision, 80.0),
            "negative_refusal": threshold_status(rejection_rate, 100.0),
            "retrieval_p50": threshold_status(r_p50, 100, less_than=True),
            "retrieval_p95": threshold_status(r_p95, 250, less_than=True),
            "pipeline_p50": threshold_status(tot_p50, 1500, less_than=True),
            "faithfulness": "NOT MEASURED",
            "answer_relevance": "NOT MEASURED",
        },
        "backend_release_gate": "INCOMPLETE",
        "warmup": "Actual encoder and neural reranker exercised before timed queries",
        "system_name": "Forge Hub RAG System",
        "total_queries_evaluated": len(query_results_log),
        "suites_evaluated": ["Factual (8)", "Multi-Chunk (8)", "Out-of-Domain Traps (8)"],
        "dataset": {
            "document": "distributed_database_architecture.pdf",
            "total_pages": len(TECHNICAL_PAGES),
            "total_chunks": total_chunks,
            "ingestion_time_seconds": ingest_time_s,
        },
        "retrieval_latency_ms": {
            "p50": round(r_p50, 2),
            "p90": round(r_p90, 2),
            "p95": round(r_p95, 2),
            "p99": round(r_p99, 2),
            "min": round(min(retrieval_latencies), 2),
            "max": round(max(retrieval_latencies), 2),
            "mean": round(float(np.mean(retrieval_latencies)), 2),
        },
        "generation_latency_ms": {
            "p50": round(g_p50, 2),
            "p90": round(g_p90, 2),
            "p95": round(g_p95, 2),
            "p99": round(g_p99, 2),
            "mean": round(float(np.mean(g_active)), 2) if g_active else 0.0,
        },
        "end_to_end_latency_ms": {
            "p50": round(tot_p50, 2),
            "p90": round(tot_p90, 2),
            "p95": round(tot_p95, 2),
            "p99": round(tot_p99, 2),
        },
        "accuracy_and_grounding": {
            "context_recall_at_5_pct": round(context_recall_5, 1),
            "context_precision_pct": round(avg_precision, 1),
            "hallucination_rejection_rate_pct": round(rejection_rate, 1),
            "factual_recall_hits": f"{recall_hits}/{total_factual_queries}",
            "trap_rejections": f"{traps_rejected}/{total_trap_queries}",
        },
        "query_details": query_results_log,
    }

    # 4. Emit Artifacts (Rule E-4)
    print("\n[4/4] Generating diagnostic artifacts in the selected output directory...")
    json_path = output_dir / "benchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"      - Saved: {json_path}")

    report_path = output_dir / "benchmark_report.md"
    report_content = f"""# Forge Hub RAG System — Empirical Benchmark Report

> **Evaluation Harness:** `scripts/benchmark_rag.py`  
> **Target System:** Forge Hub RAG System (Two-Stage Hybrid Dense+BM25 with RRF & Evidence Gate)  
> **Evaluation Date:** {metrics_summary['benchmark_timestamp']}  
> **Scope:** {metrics_summary['measurement_scope']}
>
> **Generation:** {', '.join(metrics_summary['generation_backends'])}; reranker: {', '.join(metrics_summary['reranker_backends'])}
>
> **Limit:** {metrics_summary['quality_limitations']}

---

## 1. Executive Summary

| Evaluation Dimension | Metric Target | Empirical Result | Status |
| :--- | :--- | :--- | :--- |
| **Admitted-context hit proxy** | $\\ge 85.0\\%$ | **{context_recall_5:.1f}%** ({recall_hits}/{total_factual_queries}) | {threshold_status(context_recall_5, 85.0)} |
| **Context Precision** | $\\ge 80.0\\%$ | **{avg_precision:.1f}%** | {threshold_status(avg_precision, 80.0)} |
| **Hallucination Rejection Rate** | $\\ge 100.0\\%$ | **{rejection_rate:.1f}%** ({traps_rejected}/{total_trap_queries}) | {threshold_status(rejection_rate, 100.0)} |
| **Retrieval Latency ($p50$)** | $< 100$ ms | **{r_p50:.2f} ms** | {threshold_status(r_p50, 100, less_than=True)} |
| **Retrieval Latency ($p95$)** | $< 250$ ms | **{r_p95:.2f} ms** | {threshold_status(r_p95, 250, less_than=True)} |
| **End-to-End Latency ($p50$)** | $< 1500$ ms | **{tot_p50:.2f} ms** | {threshold_status(tot_p50, 1500, less_than=True)} |

---

## 2. Latency Percentile Distribution (Rule E-2)

Measured empirically across all 24 evaluation queries ($N=24$):

| Stage | Min (ms) | p50 (ms) | p90 (ms) | p95 (ms) | p99 (ms) | Max (ms) | Mean (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Retrieval (Parallel Dense+Sparse+RRF)** | {min(retrieval_latencies):.2f} | **{r_p50:.2f}** | **{r_p90:.2f}** | **{r_p95:.2f}** | **{r_p99:.2f}** | {max(retrieval_latencies):.2f} | {metrics_summary['retrieval_latency_ms']['mean']:.2f} |
| **Generation / Synthesis** | 0.00 | **{g_p50:.2f}** | **{g_p90:.2f}** | **{g_p95:.2f}** | **{g_p99:.2f}** | {max(generation_latencies):.2f} | {metrics_summary['generation_latency_ms']['mean']:.2f} |
| **Total End-to-End** | {min(total_latencies):.2f} | **{tot_p50:.2f}** | **{tot_p90:.2f}** | **{tot_p95:.2f}** | **{tot_p99:.2f}** | {max(total_latencies):.2f} | {float(np.mean(total_latencies)):.2f} |

---

## 3. Grounding & Information Retrieval Accuracy (Rule E-3)

| Benchmark Metric | Definition | Value | Benchmark Assessment |
| :--- | :--- | :--- | :--- |
| **Context Recall@5** | $\\frac{{\\text{{Relevant Hits}}}}{{\\text{{Total Queries}}}}$ in top 5 candidate chunks | **{context_recall_5:.1f}%** | Hit proxy over admitted answer context, not all retrieval candidates. |
| **Context Precision** | $\\frac{{\\text{{Relevant Chunks in Top 5}}}}{{5}}$ averaged across queries | **{avg_precision:.1f}%** | Measured page/keyword proxy on this synthetic sample only. |
| **Hallucination Rejection Rate** | Proportion of out-of-domain queries successfully refused by Gate | **{rejection_rate:.1f}%** | Observed refusal rate on the supplied negative cases; not universal hallucination assurance. |

---

## 4. Query-Level Audit Log

### 4.1. Suite 1: Factual Retrieval
| ID | Query | Recall Hit | Precision | Retrieval (ms) | Total (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: |
"""
    for q in query_results_log:
        if q["suite"] == "Factual":
            report_content += f"| `{q['id']}` | {q['query']} | {'✅' if q['hit'] else '❌'} | {q['precision']:.2f} | {q['retrieval_ms']} | {q['total_ms']} |\n"

    report_content += """
### 4.2. Suite 2: Multi-Chunk Syntheses
| ID | Query | Recall Hit | Precision | Retrieval (ms) | Total (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: |
"""
    for q in query_results_log:
        if q["suite"] == "Multi-Chunk":
            report_content += f"| `{q['id']}` | {q['query']} | {'✅' if q['hit'] else '❌'} | {q['precision']:.2f} | {q['retrieval_ms']} | {q['total_ms']} |\n"

    report_content += """
### 4.3. Suite 3: Out-of-Domain Traps (Hallucination Defense)
| ID | Query | Refusal Triggered | Confidence | Gate Outcome |
| :--- | :--- | :---: | :---: | :--- |
"""
    for q in query_results_log:
        if q["suite"] == "Out-of-Domain Traps":
            report_content += f"| `{q['id']}` | {q['query']} | {'✅' if q['refusal_triggered'] else '❌'} | `{q['confidence']}` | {'✅ Calibrated Refusal' if q['refusal_triggered'] else '❌ Hallucination Risk'} |\n"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"      - Saved: {report_path}")

    # Also save the generated benchmark PDF for reference
    pdf_bytes = generate_benchmark_pdf(TECHNICAL_PAGES)
    pdf_path = output_dir / "sample_technical_benchmark.pdf"
    with open(pdf_path, "wb") as f:
        f.write(pdf_bytes)
    print(f"      - Saved sample PDF: {pdf_path}")

    print("\n================================================================================")
    print(f" Benchmark Complete! Recall@5: {context_recall_5:.1f}% | Rejection Rate: {rejection_rate:.1f}% | p50: {r_p50:.2f}ms")
    print("================================================================================")

    return metrics_summary


def validate_test_database(dsn):
    url = urlsplit(dsn)
    if not (url.scheme in {"postgres", "postgresql"} and url.hostname in {"127.0.0.1", "localhost", "::1"}
            and re.fullmatch(r"forge_hub_test(?:_[A-Za-z0-9_]+)?", unquote(url.path.lstrip("/")))
            and not url.query and not url.fragment):
        raise ValueError("Use a separate loopback disposable forge_hub_test database; application DSNs are never used")
    return dsn


async def isolated_benchmark(args):
    import asyncpg
    import app.db as database
    import app.documents as documents
    import app.sessions as sessions
    import app.embeddings as embeddings
    dsn = validate_test_database(args.database_url)
    output = Path(args.output_dir).resolve() if args.output_dir else Path(tempfile.mkdtemp(prefix="forge-hub-evaluation-"))
    output.mkdir(parents=True, exist_ok=True)
    settings = Settings(_env_file=None, database_url=dsn, vector_db_url=None, storage_mode="postgres",
                        openrouter_api_key=None, gemini_api_key=None)
    schema = "forge_hub_eval_" + uuid.uuid4().hex
    connection, pool = None, None
    try:
        connection = await asyncpg.connect(dsn, timeout=10)
        await connection.execute('CREATE EXTENSION IF NOT EXISTS pgcrypto WITH SCHEMA public')
        await connection.execute('CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public')
        await connection.execute('CREATE SCHEMA "' + schema + '"')
        pool = await asyncpg.create_pool(dsn, min_size=1, max_size=10,
                                        server_settings={"search_path": '"' + schema + '",public'})
        with ExitStack() as stack:
            stack.enter_context(patch.object(database, "_pool", pool))
            stack.enter_context(patch.object(database, "_use_memory_fallback", False))
            for module in (database, documents, sessions, embeddings):
                stack.enter_context(patch.object(module, "get_settings", lambda: settings))
            await database.ensure_schema()
            from app.model_readiness import warmup_models
            await warmup_models(settings)
            result = await run_benchmark(settings, "benchmark-" + uuid.uuid4().hex, output)
        print("Diagnostic results:", output)
        return 2 if "FAIL" in result["acceptance"].values() else 0
    finally:
        if pool is not None:
            await pool.close()
        if connection is not None:
            # The target is generated here, never supplied by the caller.
            if re.fullmatch(r"forge_hub_eval_[0-9a-f]{32}", schema):
                await connection.execute('DROP SCHEMA IF EXISTS "' + schema + '" CASCADE')
            await connection.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Isolated application-pipeline diagnostic; does not certify RAG triad quality")
    parser.add_argument("--database-url", default=os.environ.get("FORGE_HUB_TEST_DATABASE_URL", ""), help="Explicit disposable loopback test database only")
    parser.add_argument("--output-dir", help="Results directory; defaults to a new temporary directory")
    args = parser.parse_args()
    try:
        validate_test_database(args.database_url)
    except ValueError as error:
        parser.error(str(error))
    raise SystemExit(asyncio.run(isolated_benchmark(args)))
