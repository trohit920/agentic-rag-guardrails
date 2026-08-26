"""Unit tests for API-key resolution and rate limiting (no model loading)."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from api.security import DEFAULT_DEV_KEY, _enforce_rate_limit, expected_api_key
from src.config import API_KEY_ENV_VAR, RATE_LIMIT_PER_MINUTE


def test_expected_api_key_prefers_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """The configured environment key overrides the development default."""
    monkeypatch.setenv(API_KEY_ENV_VAR, "secret-123")
    assert expected_api_key() == "secret-123"


def test_expected_api_key_falls_back_to_dev_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the environment variable is unset, the dev default is returned."""
    monkeypatch.delenv(API_KEY_ENV_VAR, raising=False)
    assert expected_api_key() == DEFAULT_DEV_KEY


def test_rate_limit_allows_budget_then_blocks() -> None:
    """A key may make RATE_LIMIT_PER_MINUTE requests, after which it is blocked."""
    key = "unit-test-key"
    for _ in range(RATE_LIMIT_PER_MINUTE):
        _enforce_rate_limit(key)
    with pytest.raises(HTTPException) as exc:
        _enforce_rate_limit(key)
    assert exc.value.status_code == 429
