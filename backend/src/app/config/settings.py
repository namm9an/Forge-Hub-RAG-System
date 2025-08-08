from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Application settings loaded from environment variables or .env file.

    Uses Pydantic Settings for type-safe configuration and supports environment
    variable overrides. Designed for Clean Architecture separation and testability.
    """

    # App
    APP_NAME: str = "ForgeHub Backend"
    ENV: str = "development"  # development|staging|production|test
    DEBUG: bool = True

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Database
    DATABASE_URL: Optional[str] = None  # e.g., postgres://user:pass@host:5432/db
    DB_POOL_MIN_SIZE: int = 1
    DB_POOL_MAX_SIZE: int = 10

    # Supabase
    SUPABASE_URL: Optional[str] = None
    SUPABASE_ANON_KEY: Optional[str] = None

    # OpenAI or other model providers can be injected via env
    OPENAI_API_KEY: Optional[str] = None

    model_config = SettingsConfigDict(env_file=(".env", ".env.local"), env_prefix="APP_", case_sensitive=False)

    @property
    def is_production(self) -> bool:
        return self.ENV.lower() == "production"

    @property
    def is_development(self) -> bool:
        return self.ENV.lower() == "development"

    @property
    def is_test(self) -> bool:
        return self.ENV.lower() == "test"


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Load and cache application settings.

    Returns a singleton-like settings instance for the process lifetime.
    """
    return AppSettings()
