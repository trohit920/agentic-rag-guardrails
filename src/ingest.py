"""Document loading and chunking.

Loads the enterprise/financial corpus with its metadata from ``corpus.json`` and
splits documents into overlapping chunks using one of the configured strategies
(fixed-window, recursive paragraph, or sentence grouping). Chunking is pure text
logic with no model I/O, so it is unit-testable and is where the
document-processing function is optimized.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .config import (
    CHUNK_OVERLAP_CHARS,
    CHUNK_SIZE_CHARS,
    DATA_DIR,
    DEFAULT_CHUNK_STRATEGY,
    ChunkStrategy,
    Sensitivity,
)
from .schemas import Chunk, DocumentMeta

CORPUS_FILENAME: str = "corpus.json"
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")


def load_corpus(data_dir: Path = DATA_DIR) -> list[tuple[DocumentMeta, str]]:
    """Load all documents and their metadata from ``corpus.json``."""
    corpus_path = data_dir / CORPUS_FILENAME
    with corpus_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    documents: list[tuple[DocumentMeta, str]] = []
    for entry in payload["documents"]:
        meta = DocumentMeta(
            doc_id=entry["doc_id"],
            title=entry["title"],
            department=entry["department"],
            sensitivity=Sensitivity(entry["sensitivity"]),
            owner=entry["owner"],
            effective_date=entry["effective_date"],
        )
        documents.append((meta, entry["text"]))
    return documents


def _fixed_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Slide a fixed character window over text with the given overlap."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    step = chunk_size - overlap
    pieces: list[str] = []
    start = 0
    while start < len(text):
        piece = text[start : start + chunk_size].strip()
        if piece:
            pieces.append(piece)
        start += step
    return pieces


def _group_units(units: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Greedily pack text units (paragraphs/sentences) into size-bounded chunks."""
    pieces: list[str] = []
    buffer = ""
    for unit in units:
        if len(unit) > chunk_size:
            if buffer:
                pieces.append(buffer)
                buffer = ""
            pieces.extend(_fixed_split(unit, chunk_size, overlap))
        elif not buffer:
            buffer = unit
        elif len(buffer) + len(unit) + 1 <= chunk_size:
            buffer = f"{buffer}\n{unit}"
        else:
            pieces.append(buffer)
            buffer = unit
    if buffer:
        pieces.append(buffer)
    return pieces


def _split_text(text: str, strategy: ChunkStrategy, chunk_size: int, overlap: int) -> list[str]:
    """Dispatch to the requested chunking strategy, returning raw text pieces."""
    if strategy is ChunkStrategy.FIXED:
        return _fixed_split(text, chunk_size, overlap)
    if strategy is ChunkStrategy.RECURSIVE:
        paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT.split(text) if p.strip()]
        return _group_units(paragraphs, chunk_size, overlap)
    if strategy is ChunkStrategy.SENTENCE:
        flattened = " ".join(text.split())
        sentences = [s.strip() for s in _SENTENCE_SPLIT.split(flattened) if s.strip()]
        return _group_units(sentences, chunk_size, overlap)
    raise ValueError(f"unknown chunk strategy: {strategy}")


def chunk_document(
    meta: DocumentMeta,
    text: str,
    strategy: ChunkStrategy = DEFAULT_CHUNK_STRATEGY,
    chunk_size: int = CHUNK_SIZE_CHARS,
    overlap: int = CHUNK_OVERLAP_CHARS,
) -> list[Chunk]:
    """Split one document's text into overlapping chunks under the given strategy."""
    pieces = _split_text(text, strategy, chunk_size, overlap)
    return [
        Chunk(
            chunk_id=f"{meta.doc_id}-{ordinal}",
            doc_id=meta.doc_id,
            text=piece,
            meta=meta,
            ordinal=ordinal,
        )
        for ordinal, piece in enumerate(pieces)
    ]


def chunk_corpus(
    documents: list[tuple[DocumentMeta, str]],
    strategy: ChunkStrategy = DEFAULT_CHUNK_STRATEGY,
    chunk_size: int = CHUNK_SIZE_CHARS,
    overlap: int = CHUNK_OVERLAP_CHARS,
) -> list[Chunk]:
    """Chunk every document in a loaded corpus into a flat list of chunks."""
    chunks: list[Chunk] = []
    for meta, text in documents:
        chunks.extend(chunk_document(meta, text, strategy, chunk_size, overlap))
    return chunks
