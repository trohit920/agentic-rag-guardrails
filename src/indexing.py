"""Corpus indexing orchestration.

Higher-level glue that loads the corpus, chunks it under a chosen strategy,
embeds the chunks, and upserts them into the vector store. Kept separate from the
pure ``ingest`` module (which has no model or storage dependencies) to preserve
import direction.
"""

from __future__ import annotations

from pathlib import Path

from .config import (
    CHUNK_OVERLAP_CHARS,
    CHUNK_SIZE_CHARS,
    DATA_DIR,
    DEFAULT_CHUNK_STRATEGY,
    ChunkStrategy,
)
from .embeddings import Embedder
from .ingest import chunk_corpus, load_corpus
from .vector_store import VectorStore


def build_vector_store(
    store: VectorStore,
    embedder: Embedder,
    strategy: ChunkStrategy = DEFAULT_CHUNK_STRATEGY,
    chunk_size: int = CHUNK_SIZE_CHARS,
    overlap: int = CHUNK_OVERLAP_CHARS,
    data_dir: Path = DATA_DIR,
) -> int:
    """Load, chunk, embed, and upsert the corpus; return the number of chunks."""
    documents = load_corpus(data_dir)
    chunks = chunk_corpus(documents, strategy, chunk_size, overlap)
    embeddings = embedder.encode_documents([chunk.text for chunk in chunks])
    store.upsert(chunks, embeddings)
    return len(chunks)
