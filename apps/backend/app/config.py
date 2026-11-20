from functools import lru_cache
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    """
    Forge Hub RAG System — Application Configuration
    Loads configuration from environment variables and .env files.
    Complies with Rule F-3 (pydantic-settings with sensible defaults).
    """
    model_config = SettingsConfigDict(
        env_file=(".env", "../.env", "../../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Server Settings
    backend_port: int = Field(default=8000, description="FastAPI server port")
    backend_host: str = Field(default="127.0.0.1", description="FastAPI server host")
    allowed_origins: str = Field(default="http://localhost:3000", description="CORS allowed origins (comma-separated)")
    log_level: str = Field(default="INFO", description="Loguru logging level")

    # Database Configuration (PostgreSQL 16 + pgvector)
    vector_db_url: Optional[str] = Field(default=None, description="PostgreSQL connection string (with pgvector)")
    database_url: Optional[str] = Field(default=None, description="Fallback database connection string")

    # LLM & Embedding Service
    openrouter_api_key: Optional[str] = Field(default=None, description="OpenRouter API key for LLM generation")
    embedding_api_url: Optional[str] = Field(default=None, description="Optional external embedding API URL")
    embedding_api_key: Optional[str] = Field(default=None, description="Optional external embedding API Key")
    llm_model: str = Field(default="qwen/qwen3-coder:free", description="LLM model identifier")

    # RAG Engine Parameters
    embedding_model: str = Field(default="sentence-transformers/all-mpnet-base-v2", description="Embedding model name")
    embedding_dimension: int = Field(default=768, description="Embedding vector dimensionality")
    top_k_results: int = Field(default=5, description="Default number of top candidates retrieved")
    similarity_threshold: float = Field(default=0.35, description="Confidence threshold for evidence gating")
    rrf_k: int = Field(default=60, description="RRF smoothing constant")
    dense_weight: float = Field(default=0.6, description="RRF weight for dense vector search")
    sparse_weight: float = Field(default=0.4, description="RRF weight for sparse lexical search")

    # Ingestion & Chunking Constraints (Rule B-1 & Sequence Ceiling)
    chunk_size_tokens: int = Field(default=350, description="Target chunk size in tokens (capped at 350 to avoid all-mpnet-base-v2 384-token truncation)")
    chunk_overlap_tokens: int = Field(default=50, description="Chunk overlap in tokens")
    reranker_model: str = Field(default="ms-marco-TinyBERT-L-2-v2", description="FlashRank cross-encoder model name")
    max_files_per_session: int = Field(default=3, description="Maximum files per upload session")
    max_total_bytes: int = Field(default=50 * 1024 * 1024, description="Maximum total upload size (50MB)")
    session_timeout_seconds: int = Field(default=3600, description="Session inactivity timeout")


@lru_cache
def get_settings() -> Settings:
    return Settings()
