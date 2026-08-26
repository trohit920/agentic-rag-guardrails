"""Unit tests for token-budgeted context assembly and history compaction."""

from __future__ import annotations

from src.context import NO_CONTEXT_PLACEHOLDER, assemble_context, summarize_history
from src.schemas import Chunk, DocumentMeta, RetrievalResult, Sensitivity


def _count(text: str) -> int:
    """Simple whitespace token counter for deterministic tests."""
    return len(text.split())


def _result(chunk_id: str, text: str) -> RetrievalResult:
    meta = DocumentMeta("d", "T", "Finance", Sensitivity.INTERNAL, "cfo", "2025-01-01")
    return RetrievalResult(chunk=Chunk(chunk_id, "d", text, meta, 0), score=1.0)


def test_summarize_history_keeps_recent_and_notes_omissions() -> None:
    """Older turns are dropped to fit the budget and an omission note is added."""
    history = [{"role": "user", "content": "a b c"} for _ in range(5)]
    compact = summarize_history(history, _count, max_history_tokens=6)
    assert compact[0]["role"] == "system"
    assert "omitted" in compact[0]["content"]
    assert len(compact) == 3  # note + 2 kept turns


def test_summarize_history_no_note_when_within_budget() -> None:
    """When everything fits, history passes through unchanged."""
    history = [{"role": "user", "content": "a b"}]
    assert summarize_history(history, _count, max_history_tokens=100) == history


def test_assemble_context_places_system_first_and_query_last() -> None:
    """Assembled messages start with the system prompt and end with the query."""
    messages = assemble_context("my question", [_result("d-0", "fact one")], [], _count)
    assert messages[0]["role"] == "system"
    assert "[1] fact one" in messages[0]["content"]
    assert messages[-1] == {"role": "user", "content": "my question"}


def test_assemble_context_truncates_to_budget_but_keeps_one_chunk() -> None:
    """Context respects the token budget yet always includes at least one chunk."""
    results = [_result(f"d-{i}", "word " * 20) for i in range(5)]
    messages = assemble_context("q", results, [], _count, max_context_tokens=5)
    assert "[1]" in messages[0]["content"]
    assert "[5]" not in messages[0]["content"]


def test_assemble_context_placeholder_when_no_results() -> None:
    """With no retrieval, the context block shows the placeholder."""
    messages = assemble_context("q", [], [], _count)
    assert NO_CONTEXT_PLACEHOLDER in messages[0]["content"]
