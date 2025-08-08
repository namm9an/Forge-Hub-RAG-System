# Advanced RAG Chatbot Monorepo

This repository is a Turborepo + PNPM workspace managing a modern RAG (Retrieval-Augmented Generation) chatbot system with:

- apps/frontend: Next.js 14 (App Router) + TypeScript + ESLint/Prettier + Tailwind-ready
- apps/backend: Python FastAPI service (httpx, pydantic, uvicorn) with structured logging
- packages/common: Shared TypeScript utilities and schemas used by the frontend (and optionally typed API clients)

Modern Development Standards applied:
- Config via environment variables only (see .env.example files)
- No secrets committed; use dotenv locally and secret managers in CI/CD
- Structured logging and robust error handling
- Type-safe interfaces and validation (zod/pydantic)
- Ready for AI Agent/MCP integration patterns

Getting Started
- Prereqs: Node 18+, PNPM 9+, Python 3.11+, pipx or uv/poetry, and Git
- Install JS deps: pnpm install
- Set up backend venv: python -m venv .venv && source .venv/bin/activate (Windows: .venv\Scripts\activate)
- Install backend deps: pip install -r apps/backend/requirements.txt

Commands
- pnpm dev: Run all apps in dev mode via Turborepo
- pnpm build: Build all packages/apps
- pnpm lint: Lint all workspaces
- pnpm test: Run tests in all workspaces

Directory Structure
- apps/frontend: Next.js app
- apps/backend: FastAPI app
- packages/common: Shared TS utilities

Environment Variables
- Root .env.example covers shared settings
- Each app has its own .env.example for app-specific configuration

License
- MIT

