# Backend FastAPI service for Advanced RAG Chatbot

This service provides API endpoints for chat and retrieval operations.

Modern Development Standards
- Environment-based configuration (.env.example)
- Structured logging (JSON)
- httpx for async HTTP calls
- Pydantic for validation
- Health check endpoint

Local Development
- Create venv: python -m venv .venv
- Activate venv and install: pip install -r requirements.txt
- Run dev server: uvicorn app.main:app --reload --port 8000

