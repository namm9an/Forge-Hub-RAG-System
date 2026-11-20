# Forge Hub RAG System

> Production-grade hybrid retrieval-augmented generation engine with HNSW vector indexing, reciprocal rank fusion, quantized cross-encoder reranking, and evidence-gated hallucination defense.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-009688.svg)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Next.js-14-black.svg)](https://nextjs.org)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.5-blue.svg)](https://typescriptlang.org)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16%20%2B%20pgvector-336791.svg)](https://github.com/pgvector/pgvector)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Architecture Overview

Naive RAG pipelines suffer from three systemic failure modes in production:
1. **Semantic Drift & Lexical Blindness:** Dense-only bi-encoders fail on exact technical identifiers, acronyms, and error codes; sparse-only BM25 fails on semantic paraphrasing.
2. **Context Distraction & Prompt Dilution:** Stuffing raw 500+ token chunks into an LLM context introduces irrelevant sentences, confusing long-context attention heads.
3. **Unchecked Hallucination on Unanswerable Queries:** Standard RAG pipelines force generation regardless of retrieved context quality, generating plausible-sounding fabrications.

Forge Hub addresses these failures with a two-stage hybrid retrieval and evidence-gated generation pipeline:

```mermaid
flowchart TD
    subgraph Ingestion ["1. Document Ingestion & Storage"]
        PDF["Technical Document (PDF)"] --> Parser["Layout & Text Extractor (pdfplumber / pypdf)"]
        Parser --> Chunker["Token-Aware Recursive Chunker\n(Max 350 tokens, 50 overlap, sentence boundaries)"]
        Chunker --> Embedder["all-mpnet-base-v2 (768-dim normalized)"]
        Chunker --> DB_Text[("PostgreSQL 16\nGIN Index on content_tsv")]
        Embedder --> DB_Vector[("PostgreSQL 16 + pgvector\nHNSW Index (m=16, ef_c=64)")]
    end

    subgraph Stage1 ["2. Stage-1 Parallel Hybrid Retrieval"]
        Query["User Query"] --> Dense["Dense Cosine Search\n1 - (embedding <=> q_vector)"]
        Query --> Sparse["Sparse Lexical Search\nts_rank_cd(content_tsv, websearch_to_tsquery)"]
        Dense --> RRF["Reciprocal Rank Fusion (k=60)\nRRF(d) = Σ w_m / (60 + rank_m(d))"]
        Sparse --> RRF
    end

    subgraph Stage2 ["3. Stage-2 Neural Reranking & Evidence Gating"]
        RRF --> Candidates["Top-K Candidate Pool (k=20)"]
        Candidates --> Rerank["FlashRank Cross-Encoder\nQuery-Chunk Token Cross-Attention"]
        Rerank --> Gate{"Evidence Confidence Gate\n(Dense + Sparse + Cross-Encoder Scores)"}
        Gate -->|Unanswerable / Out-of-Domain| Refusal["Calibrated Refusal:\n'Context insufficient to answer'"]
        Gate -->|Confidence Met| Pruner["Sentence-Level Noise Pruner\nPer-sentence relevance thresholding"]
    end

    subgraph Generation ["4. Attributed Generation & Streaming"]
        Pruner --> Context["Cleaned Context + Exact Citations\n[Doc: file, Page: no, Chunk: id]"]
        Context --> LLM["LLM Synthesis (OpenRouter / Local Engine)"]
        LLM --> Stream["SSE Token Stream (/api/chat/stream)"]
        Refusal --> Stream
    end
```

---

## Key Engineering Decisions & Trade-Offs

### 1. PostgreSQL 16 + pgvector vs. Standalone Vector Databases
* **Decision:** Consolidate relational metadata, document content, full-text indexes, and high-dimensional vector embeddings into PostgreSQL 16 with `pgvector` and `HNSW`.
* **Trade-Off:** Dedicated vector databases (Pinecone, Qdrant, Milvus) offer specialized distributed clustering for $100\text{M}+$ vectors. However, in enterprise document environments ($< 10\text{M}$ vectors), managing separate vector and relational stores introduces distributed state synchronization risks, dual authorization models, and eventual consistency lag. PostgreSQL provides single-transaction ACID guarantees, cross-table relational joins, cascading foreign keys (`ON DELETE CASCADE`), and transactional rollback on failed ingestions.

### 2. Reciprocal Rank Fusion (RRF) vs. Linear Score Summation
* **Decision:** Merge dense cosine similarity and sparse `ts_rank_cd` scores using rank-based Reciprocal Rank Fusion ($k=60$) rather than normalizing and adding raw scores:
  $$\text{RRF}(d) = \sum_{m \in M} \frac{w_m}{60 + \text{rank}_m(d)}$$
* **Trade-Off:** Dense cosine distances are bounded $[0, 1]$, whereas lexical BM25/`ts_rank_cd` scores are unbounded and sensitive to document length and term frequency. Directly combining them via weighted summation requires domain-specific score calibration that breaks when query distribution shifts. RRF relies exclusively on positional rank, guaranteeing robust, distribution-invariant fusion without manual calibration.

### 3. Strict 350-Token Chunk Boundary vs. 512-Token Default
* **Decision:** Enforce a strict ceiling of 350 tokens (with 50-token sliding overlap) during document segmentation.
* **Trade-Off:** While modern embedding models accept 512-token sequence lengths, `sentence-transformers/all-mpnet-base-v2` has an effective position embedding capacity of 384 tokens. Standard 512-token chunking results in silent truncation of the tail 128 tokens during embedding generation. Capping chunks at 350 tokens ensures $100\%$ token coverage without data loss while preserving complete sentence boundaries.

### 4. Stage-2 Quantized Cross-Encoder Reranking
* **Decision:** Implement a two-stage retrieval pattern pairing Stage-1 hybrid search (top-20 candidates) with Stage-2 FlashRank quantized ONNX cross-encoder reranking before prompt synthesis.
* **Trade-Off:** Bi-encoders embed queries and documents independently into vector space, missing fine-grained token-level cross-interactions. Full cross-encoders provide high precision but are computationally prohibitive to run across an entire database. A two-stage pipeline restricts cross-attention evaluation to the top-20 retrieved candidates, improving MRR@5 with negligible latency overhead.

---

## Directory Structure

```text
Forge-Hub-RAG-System/
├── apps/
│   ├── backend/                     # FastAPI ASGI application
│   │   ├── app/
│   │   │   ├── chat.py              # Attributed chat & SSE streaming router
│   │   │   ├── config.py            # Environment configuration & pydantic-settings
│   │   │   ├── db.py                # asyncpg connection pool & dual-engine storage
│   │   │   ├── documents.py         # Multi-page PDF ingestion & token chunker
│   │   │   ├── embeddings.py        # SentenceTransformer embedding worker
│   │   │   ├── gating.py            # Evidence evaluator & sentence noise pruner
│   │   │   ├── main.py              # FastAPI app factory & lifecycle handlers
│   │   │   ├── retrieval.py         # Parallel dense + sparse hybrid RRF search
│   │   │   └── schemas.py           # Pydantic v2 domain models
│   │   ├── tests/                   # 27 automated unit and integration tests
│   │   ├── requirements.txt         # Pinned backend dependencies
│   │   └── pyproject.toml           # Pytest & tool configurations
│   └── frontend/                    # Next.js 14 App Router UI
│       ├── app/                     # React Server Components & client views
│       ├── components/              # Modular UI components (Chat, Upload, Sources)
│       └── package.json
├── packages/
│   └── common/                      # Shared types and validation contracts
├── scripts/
│   ├── bootstrap.sql                # PostgreSQL 16 schema, HNSW & GIN DDL
│   ├── validate_db.sql              # Database verification script
│   └── benchmark_rag.py             # Reproducible latency & retrieval benchmark harness
├── docker-compose.yml               # PostgreSQL 16 + pgvector container definition
├── turbo.json                       # Turborepo monorepo pipeline configuration
└── README.md
```

---

## Quick Start

### 1. Start Infrastructure
Launch PostgreSQL 16 with `pgvector` pre-configured with the schema and indexes:
```bash
docker compose up -d
```

### 2. Configure Environment
Copy the example environment file and configure optional LLM keys (the system includes local fallback synthesis if external API keys are omitted):
```bash
cp .env.example .env
```

### 3. Launch Backend
```bash
cd apps/backend
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

pip install -r requirements.txt
uvicorn app.main:create_app --factory --reload --port 8000
```
* API Root: `http://localhost:8000`
* Interactive Documentation: `http://localhost:8000/docs`

### 4. Launch Frontend

Run these commands from the repository root in a separate terminal.

```bash
npm ci
npm run dev --workspace @advanced-rag/frontend
```
* Frontend Interface: `http://localhost:3000`

---

## API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Health check verifying database pool connectivity |
| `POST` | `/api/documents/upload` | Ingests multi-page technical PDFs, chunks at 350 tokens, and stores vectors |
| `GET` | `/api/documents/list` | Lists ingested documents scoped to the active session |
| `POST` | `/api/chat` | Non-streaming hybrid retrieval, evidence gating, and attributed answer synthesis |
| `POST` | `/api/chat/stream` | Server-Sent Events (SSE) streaming tokens with latency profiling metadata |
| `GET` | `/api/chat/history` | Fetches historical session turns with retrieval and generation latency metrics |
| `DELETE`| `/api/sessions/{session_id}` | Cascading deletion of session, chat history, documents, and embeddings |

---

## Verification & Testing

The test suite covers chunking boundaries, HNSW schema compatibility, Reciprocal Rank Fusion mechanics, score divergence, cross-encoder reranking, and evidence gating refusals.

Run the test suite from the repository root:
```bash
python -m pytest apps/backend/tests -v
```

Execute the automated benchmark harness across factual, multi-chunk, and out-of-domain evaluation suites:
```bash
python scripts/benchmark_rag.py
```

---

## License

This project is licensed under the [MIT License](LICENSE).
