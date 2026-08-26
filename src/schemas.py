"""Shared data structures for the RAG pipeline.

Defines the typed records passed between ingestion, retrieval, generation, and
guardrail stages. This is the lowest-level module (imports only config and the
standard library) so every other module can depend on it without cycles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import Sensitivity


@dataclass(frozen=True)
class DocumentMeta:
    """Metadata attached to every ingested document (and its chunks)."""

    doc_id: str
    title: str
    department: str
    sensitivity: Sensitivity
    owner: str
    effective_date: str  # ISO date string


@dataclass(frozen=True)
class Chunk:
    """A single retrievable text chunk with provenance and access metadata."""

    chunk_id: str
    doc_id: str
    text: str
    meta: DocumentMeta
    ordinal: int  # position of the chunk within its source document


@dataclass(frozen=True)
class RetrievalResult:
    """A chunk returned by retrieval, with its fused relevance score."""

    chunk: Chunk
    score: float


@dataclass(frozen=True)
class GuardVerdict:
    """Outcome of a guardrail check with a human-readable reason."""

    passed: bool
    reason: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GroundednessReport:
    """Per-answer groundedness result from NLI faithfulness scoring."""

    grounded_ratio: float
    supported_sentences: int
    total_sentences: int
    ungrounded: list[str] = field(default_factory=list)


@dataclass
class AgentAnswer:
    """Final agent output: the answer text, its citations, and guard results."""

    query: str
    answer: str
    citations: list[str] = field(default_factory=list)  # chunk_ids
    abstained: bool = False
    verdicts: list[GuardVerdict] = field(default_factory=list)
    latency_seconds: float = 0.0
    completion_tokens: int = 0
