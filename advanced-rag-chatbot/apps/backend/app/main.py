from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from .config import get_settings
from .documents import router as documents_router
from .embeddings import router as embeddings_router
from .chat import router as chat_router
from .sessions import router as sessions_router
from .db import get_pool, close_pool, fetch
from .schemas import StatsResponse


def create_app() -> FastAPI:
    settings = get_settings()

    # Configure structured logging
    logger.remove()
    logger.add(lambda msg: print(msg, end=""))

    app = FastAPI(title="Advanced RAG Chatbot Backend", version="0.1.0")

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in settings.allowed_origins.split(",")],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    @app.get("/api/stats", response_model=StatsResponse)
    async def stats():
        rows = await fetch(
            """
            SELECT
              (SELECT COUNT(*) FROM documents) AS total_documents,
              (SELECT COUNT(*) FROM chunks) AS total_chunks,
              (SELECT COUNT(*) FROM embeddings) AS total_embeddings,
              (SELECT COUNT(*) FROM sessions) AS total_sessions,
              (SELECT COUNT(*) FROM chat_history) AS total_messages
            """
        )
        r = rows[0]
        return StatsResponse(
            total_documents=r["total_documents"],
            total_chunks=r["total_chunks"],
            total_embeddings=r["total_embeddings"],
            total_sessions=r["total_sessions"],
            total_messages=r["total_messages"],
        )

    # Routers
    app.include_router(documents_router)
    app.include_router(embeddings_router)
    app.include_router(chat_router)
    app.include_router(sessions_router)

    @app.on_event("startup")
    async def _on_startup():
        await get_pool()

    @app.on_event("shutdown")
    async def _on_shutdown():
        await close_pool()

    return app

