from functools import lru_cache
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
from pathlib import Path

# Robust .env loading: try backend root, then app dir, then CWD
_backend_root = Path(__file__).resolve().parent.parent  # apps/backend
_env_candidates = [
    _backend_root / ".env",
    Path(__file__).resolve().parent / ".env",
    Path.cwd() / ".env",
]
for _p in _env_candidates:
    if _p.is_file():
        load_dotenv(dotenv_path=_p, override=False)
# final fallback to default behavior
load_dotenv(override=False)


class Settings(BaseModel):
    backend_port: int = Field(default=int(os.getenv("BACKEND_PORT", 8000)))
    backend_host: str = Field(default=os.getenv("BACKEND_HOST", "127.0.0.1"))
    allowed_origins: str = Field(default=os.getenv("ALLOWED_ORIGINS", "http://localhost:3000"))
    log_level: str = Field(default=os.getenv("LOG_LEVEL", "INFO"))
    jwt_secret: str = Field(default=os.getenv("JWT_SECRET", "change_me"))

    vector_db_url: str | None = os.getenv("VECTOR_DB_URL")
    database_url: str | None = os.getenv("DATABASE_URL")
    embedding_api_url: str | None = os.getenv("EMBEDDING_API_URL")
    embedding_api_key: str | None = os.getenv("OPENROUTER_API_KEY")


@lru_cache
def get_settings() -> Settings:
    return Settings()

