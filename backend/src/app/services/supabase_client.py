from __future__ import annotations

from typing import Optional

import structlog

from app.config.settings import get_settings

try:
    from supabase import create_client, Client  # type: ignore
except Exception:  # pragma: no cover - optional import for environments without supabase
    create_client = None  # type: ignore
    Client = object  # type: ignore

logger = structlog.get_logger(__name__)

_supabase: Optional["Client"] = None


def get_supabase() -> Optional["Client"]:
    global _supabase
    if _supabase is not None:
        return _supabase
    settings = get_settings()
    url = settings.SUPABASE_URL
    key = settings.SUPABASE_ANON_KEY
    if not url or not key or create_client is None:
        logger.warning("supabase_not_configured")
        return None
    try:
        _supabase = create_client(url, key)
        return _supabase
    except Exception as e:
        logger.error("supabase_init_failed", error=str(e))
        return None
