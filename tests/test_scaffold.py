"""Scaffold tests: validate the pure foundation (config + schemas).

These deliberately avoid importing model-dependent modules so they run before
heavy dependencies (torch, transformers) are installed. Phase-specific modules
get their own tests as they are implemented.
"""

from __future__ import annotations

from src import config
from src.schemas import AgentAnswer, Chunk, DocumentMeta


def test_seed_is_fixed() -> None:
    """Reproducibility seed is the project-wide constant 42."""
    assert config.SEED == 42


def test_sensitivity_ordering_is_ascending() -> None:
    """Clearance ordering runs public -> internal -> confidential."""
    order = config.SENSITIVITY_ORDER
    assert order[0] is config.Sensitivity.PUBLIC
    assert order[-1] is config.Sensitivity.CONFIDENTIAL
    assert len(order) == len(set(order))


def test_thresholds_are_in_valid_ranges() -> None:
    """Guardrail ratio thresholds are probabilities in [0, 1]."""
    assert 0.0 <= config.GROUNDEDNESS_ENTAILMENT_MIN <= 1.0
    assert 0.0 <= config.GROUNDEDNESS_ACCEPT_RATIO <= 1.0
    assert 0.0 <= config.HYBRID_DENSE_WEIGHT <= 1.0


def test_chunk_schema_construction() -> None:
    """A Chunk carries provenance and access metadata."""
    meta = DocumentMeta(
        doc_id="d1",
        title="Expense Policy",
        department="Finance",
        sensitivity=config.Sensitivity.INTERNAL,
        owner="cfo",
        effective_date="2025-01-01",
    )
    chunk = Chunk(chunk_id="d1-0", doc_id="d1", text="hello", meta=meta, ordinal=0)
    assert chunk.meta.sensitivity is config.Sensitivity.INTERNAL
    assert chunk.doc_id == "d1"


def test_agent_answer_defaults() -> None:
    """A fresh AgentAnswer defaults to not-abstained with empty citations."""
    ans = AgentAnswer(query="q", answer="a")
    assert ans.abstained is False
    assert ans.citations == []
