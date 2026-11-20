from functools import lru_cache
from typing import Optional, Literal
from pydantic import Field, SecretStr, model_validator
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
    allowed_hosts: str = Field(default="localhost,127.0.0.1", description="Comma-separated production Host allowlist, without schemes")
    log_level: str = Field(default="INFO", description="Loguru logging level")
    environment: Literal["development", "test", "production"] = "development"
    storage_mode: Literal["postgres", "memory"] = "postgres"
    auth_public_key: Optional[str] = Field(default=None, description="Trusted issuer RSA public PEM; never a private signing key")
    auth_issuer: Optional[str] = None
    auth_audience: Optional[str] = None
    auth_max_token_lifetime_seconds: int = Field(default=3600, ge=60, le=86400)

    # Database Configuration (PostgreSQL 16 + pgvector)
    vector_db_url: Optional[str] = Field(default=None, repr=False, description="PostgreSQL connection string (with pgvector)")
    database_url: Optional[str] = Field(default=None, repr=False, description="Fallback database connection string")

    # LLM & Embedding Service
    openrouter_api_key: Optional[SecretStr] = Field(default=None, description="OpenRouter API key for LLM generation")
    embedding_api_url: Optional[str] = Field(default=None, description="Optional external embedding API URL")
    embedding_api_key: Optional[SecretStr] = Field(default=None, description="Optional external embedding API Key")
    llm_model: str = Field(default="qwen/qwen3-coder:free", description="LLM model identifier")

    # RAG Engine Parameters
    embedding_model: str = Field(default="sentence-transformers/all-mpnet-base-v2", description="Embedding model name")
    embedding_dimension: Literal[768] = Field(default=768, description="Matches the versioned vector schema")
    top_k_results: int = Field(default=5, description="Default number of top candidates retrieved")
    similarity_threshold: float = Field(default=0.35, description="Confidence threshold for evidence gating")
    rrf_k: int = Field(default=60, description="RRF smoothing constant")
    dense_weight: float = Field(default=0.6, description="RRF weight for dense vector search")
    sparse_weight: float = Field(default=0.4, description="RRF weight for sparse lexical search")

    # Ingestion & Chunking Constraints (Rule B-1 & Sequence Ceiling)
    chunk_size_tokens: int = Field(default=350, ge=8, le=350, description="Total model tokens including special tokens")
    chunk_overlap_tokens: int = Field(default=50, ge=0, description="Chunk overlap in tokens")
    reranker_model: str = Field(default="ms-marco-TinyBERT-L-2-v2", description="FlashRank cross-encoder model name")
    max_files_per_session: int = Field(default=3, ge=1, le=100, description="Cumulative maximum files per session")
    max_total_bytes: int = Field(default=50 * 1024 * 1024, ge=1024, le=100 * 1024 * 1024)
    session_timeout_seconds: int = Field(default=3600, ge=60, le=86400)
    max_pages_per_document: int = Field(default=500, ge=1, le=2000)
    max_extracted_chars: int = Field(default=2_000_000, ge=1000, le=10_000_000)
    max_chunks_per_document: int = Field(default=10_000, ge=1, le=50_000)
    embedding_workers: int = Field(default=2, ge=1, le=8)
    max_query_chars: int = Field(default=4000, ge=100, le=4000)
    max_context_chars: int = Field(default=24000, ge=1000, le=100000)
    max_history_turns: int = Field(default=3, ge=0, le=10)
    provider_timeout_seconds: int = Field(default=60, ge=1, le=120)
    max_answer_chars: int = Field(default=16000, ge=100, le=64000)
    requests_per_minute: int = Field(default=120, ge=1, le=10000)
    max_sessions_per_user: int = Field(default=5, ge=1, le=100)
    max_inflight_requests: int = Field(default=8, ge=1, le=100)
    run_migrations: bool = Field(default=False, description="Explicit operator opt-in for schema upgrades at startup")

    @model_validator(mode="after")
    def validate_limits(self):
        if self.chunk_overlap_tokens >= self.chunk_size_tokens - 2:
            raise ValueError("Chunk overlap must leave space for new content and model special tokens")
        if self.environment == "production" and self.storage_mode != "postgres":
            raise ValueError("Production requires PostgreSQL storage")
        if self.environment == "production":
            hosts = [host.strip() for host in self.allowed_hosts.split(",") if host.strip()]
            if not hosts or any("*" in host or "/" in host or any(char.isspace() for char in host) for host in hosts):
                raise ValueError("Production requires an explicit Host allowlist")
            if "*" in (origin.strip() for origin in self.allowed_origins.split(",")):
                raise ValueError("Production requires explicit allowed browser origins")
            if self.run_migrations:
                raise ValueError("Production migrations must use the separate operator migration command")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
