"""Pydantic request/response models for the API layer.

Defines the HTTP contract for the /chat and /ingest endpoints, including input
validation (non-empty query, valid clearance level). Kept separate from the app
so the contract is easy to review.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    """A single prior conversation turn."""

    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    """A chat query with the caller's clearance and optional history."""

    query: str = Field(min_length=1, max_length=2000)
    clearance: Literal["public", "internal", "confidential"] = "public"
    history: list[Message] = Field(default_factory=list)


class ChatResponse(BaseModel):
    """The agent's grounded answer with citations and guardrail outcome."""

    answer: str
    citations: list[str]
    abstained: bool
    latency_seconds: float


class IngestResponse(BaseModel):
    """Result of a corpus (re)indexing request."""

    chunks_indexed: int
