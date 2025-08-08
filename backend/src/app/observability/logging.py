from __future__ import annotations

import logging
import os
from typing import Any, Dict

import structlog


def _add_env_and_service(_, __, event_dict: Dict[str, Any]) -> Dict[str, Any]:
    event_dict.setdefault("service", os.getenv("APP_APP_NAME", "ForgeHub Backend"))
    event_dict.setdefault("env", os.getenv("APP_ENV", "development"))
    return event_dict


def _add_request_context(_, __, event_dict: Dict[str, Any]) -> Dict[str, Any]:
    # Pull contextvars (request_id, trace_id) if present
    request_id = structlog.contextvars.get_contextvars().get("request_id")
    if request_id:
        event_dict.setdefault("request_id", request_id)
    # Trace/span IDs may be injected by telemetry processor
    return event_dict


def configure_logging(level: int | str = logging.INFO) -> None:
    """Configure structlog for JSON output with contextvars.

    Safe to call multiple times; structlog will reconfigure.
    """
    timestamper = structlog.processors.TimeStamper(fmt="iso")

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            _add_request_context,
            _add_env_and_service,
            structlog.processors.add_log_level,
            timestamper,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(level) if isinstance(level, str) else level
        ),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Bridge standard logging to structlog
    logging.basicConfig(level=level)
    for noisy in ("uvicorn", "uvicorn.access", "asyncio", "httpx"):
        logging.getLogger(noisy).setLevel(os.getenv("APP_LOG_NOISY_LEVEL", "WARNING"))
