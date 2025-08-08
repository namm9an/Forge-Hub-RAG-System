from functools import lru_cache
from pydantic import BaseModel, Field
import os


class Settings(BaseModel):
    backend_port: int = Field(default=int(os.getenv("BACKEND_PORT", 8000)))
    backend_host: str = Field(default=os.getenv("BACKEND_HOST", "127.0.0.1"))
    allowed_origins: str = Field(default=os.getenv("ALLOWED_ORIGINS", "http://localhost:3000"))
    log_level: str = Field(default=os.getenv("LOG_LEVEL", "INFO"))
    jwt_secret: str = Field(default=os.getenv("JWT_SECRET", "change_me"))

    vector_db_url: str | None = os.getenv("VECTOR_DB_URL")
    vector_db_api_key: str | None = os.getenv("VECTOR_DB_API_KEY")
    embedding_api_url: str | None = os.getenv("EMBEDDING_API_URL")
    embedding_api_key: str | None = os.getenv("EMBEDDING_API_KEY")


@lru_cache
def get_settings() -> Settings:
    return Settings()

