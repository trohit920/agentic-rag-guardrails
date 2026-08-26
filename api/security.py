"""API-key authentication and in-memory rate limiting.

Provides a FastAPI dependency that validates the API key header and enforces a
per-key request rate limit. The expected key is read from the environment; a
development default is used with a warning when the variable is unset. State is
in-process (sufficient for a single-instance POC).
"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict, deque

from fastapi import Header, HTTPException, status

from src.config import API_KEY_ENV_VAR, API_KEY_HEADER, RATE_LIMIT_PER_MINUTE

DEFAULT_DEV_KEY: str = "dev-secret-key"
_RATE_WINDOW_SECONDS: float = 60.0
_request_times: dict[str, deque[float]] = defaultdict(deque)
_logger = logging.getLogger("rag.api")


def expected_api_key() -> str:
    """Return the configured API key, falling back to a dev default with a warning."""
    key = os.environ.get(API_KEY_ENV_VAR)
    if key is None:
        _logger.warning("%s not set; using insecure development key", API_KEY_ENV_VAR)
        return DEFAULT_DEV_KEY
    return key


def _enforce_rate_limit(api_key: str) -> None:
    """Raise HTTP 429 when a key exceeds the per-minute request budget."""
    now = time.time()
    window = _request_times[api_key]
    while window and now - window[0] > _RATE_WINDOW_SECONDS:
        window.popleft()
    if len(window) >= RATE_LIMIT_PER_MINUTE:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="rate limit exceeded"
        )
    window.append(now)


async def require_api_key(x_api_key: str = Header(default="", alias=API_KEY_HEADER)) -> str:
    """Validate the API key header and apply rate limiting; returns the key."""
    if x_api_key != expected_api_key():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid or missing API key"
        )
    _enforce_rate_limit(x_api_key)
    return x_api_key
