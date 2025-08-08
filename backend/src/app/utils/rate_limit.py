from __future__ import annotations

import time
from typing import Callable, Dict, Tuple

from fastapi import HTTPException, Request

# Simple in-memory rate limiter (sliding window approximation)
# key: ip -> (reset_ts, count)
_state: Dict[str, Tuple[float, int]] = {}


def rate_limit_tracking(limit: int = 60, window_seconds: int = 60) -> Callable[[Request], None]:
    """FastAPI dependency to enforce a per-IP rate limit.

    Uses a simple fixed window counter per IP. In production, replace with Redis.
    """

    async def _dependency(request: Request) -> None:
        now = time.time()
        ip = request.client.host if request.client else "unknown"
        reset, count = _state.get(ip, (now + window_seconds, 0))
        if now > reset:
            reset = now + window_seconds
            count = 0
        count += 1
        _state[ip] = (reset, count)
        if count > limit:
            raise HTTPException(status_code=429, detail="Too Many Requests")

    return _dependency
