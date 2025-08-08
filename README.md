# Forge‑Hub RAG System

This repository hosts a Retrieval‑Augmented Generation (RAG) system with:

- backend/ — FastAPI service (RAG pipeline, hybrid retrieval, re‑ranking, observability)
- advanced-rag-chatbot/ — app workspace containing backend/frontend experiments
- rag-document-system/ — Next.js document UI prototype and assets

Notes
- Tests are intentionally excluded from this public repo per request.
- Workflows and migrations live under backend/.

Getting started (backend)
1. python -m venv .venv && .venv\\Scripts\\activate
2. pip install -e backend
3. uvicorn app.api.main:app --reload --app-dir backend/src

CI runs ruff, black, mypy and pytest (if tests are added) on PRs.

