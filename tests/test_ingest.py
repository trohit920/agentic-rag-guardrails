"""Unit tests for the pure ingestion/chunking logic and ACL helper.

No model or storage I/O: these validate chunking correctness across strategies,
corpus loading, and clearance-based sensitivity filtering deterministically.
"""

from __future__ import annotations

from src.config import ChunkStrategy, Sensitivity, allowed_sensitivities
from src.ingest import chunk_document, load_corpus
from src.schemas import DocumentMeta

_META = DocumentMeta(
    doc_id="doc",
    title="Test Doc",
    department="Finance",
    sensitivity=Sensitivity.INTERNAL,
    owner="cfo",
    effective_date="2025-01-01",
)

_PARAGRAPH = (
    "First paragraph sentence one. First paragraph sentence two.\n\n"
    "Second paragraph is here with several words to fill space. It continues on.\n\n"
    "Third paragraph closes the document with a final statement of some length."
)


def test_fixed_chunks_respect_size_and_overlap() -> None:
    """Fixed-window chunks never exceed the size and consecutive chunks overlap."""
    chunks = chunk_document(_META, _PARAGRAPH, ChunkStrategy.FIXED, chunk_size=60, overlap=15)
    assert len(chunks) >= 2
    assert all(len(c.text) <= 60 for c in chunks)
    # Overlap: the tail of chunk i shares characters with the head of chunk i+1.
    tail = chunks[0].text[-10:]
    assert tail in chunks[1].text or chunks[1].text.startswith(chunks[0].text[45:])


def test_chunk_ids_are_unique_and_ordinals_sequential() -> None:
    """Chunk ids are unique and ordinals run 0..n-1 in order."""
    chunks = chunk_document(_META, _PARAGRAPH, ChunkStrategy.RECURSIVE, chunk_size=80, overlap=10)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    assert all(c.chunk_id == f"doc-{c.ordinal}" for c in chunks)


def test_recursive_prefers_paragraph_boundaries() -> None:
    """With a generous size, recursive chunking keeps whole paragraphs together."""
    chunks = chunk_document(_META, _PARAGRAPH, ChunkStrategy.RECURSIVE, chunk_size=200, overlap=20)
    assert any("First paragraph sentence one." in c.text for c in chunks)
    assert all(c.text.strip() for c in chunks)


def test_sentence_strategy_groups_sentences() -> None:
    """Sentence chunking produces non-empty, size-bounded groups of sentences."""
    chunks = chunk_document(_META, _PARAGRAPH, ChunkStrategy.SENTENCE, chunk_size=90, overlap=0)
    assert len(chunks) >= 2
    assert all(c.text.strip() for c in chunks)


def test_load_corpus_returns_documents_with_valid_metadata() -> None:
    """The shipped corpus loads into typed documents with valid sensitivity."""
    documents = load_corpus()
    assert len(documents) == 10
    for meta, text in documents:
        assert isinstance(meta.sensitivity, Sensitivity)
        assert text.strip()


def test_allowed_sensitivities_is_inclusive_downward() -> None:
    """A clearance grants access to its own level and all lower levels."""
    assert allowed_sensitivities(Sensitivity.PUBLIC) == (Sensitivity.PUBLIC,)
    assert Sensitivity.CONFIDENTIAL in allowed_sensitivities(Sensitivity.CONFIDENTIAL)
    assert Sensitivity.CONFIDENTIAL not in allowed_sensitivities(Sensitivity.INTERNAL)
    assert len(allowed_sensitivities(Sensitivity.INTERNAL)) == 2
