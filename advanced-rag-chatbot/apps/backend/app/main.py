from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from .config import get_settings


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

    return app

