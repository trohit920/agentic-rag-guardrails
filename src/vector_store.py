"""Persistent local vector store built on Chroma.

Wraps a Chroma collection: upsert of chunks with their embeddings and metadata,
and similarity query with metadata / access-control pre-filtering. Access control
is enforced here in the ``where`` clause so a user never sees a chunk above their
clearance. Keeps all Chroma-specific calls behind a narrow interface so retrieval
strategies stay storage-agnostic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
import numpy as np

from .config import CHROMA_DIR, allowed_sensitivities
from .schemas import Chunk, DocumentMeta, RetrievalResult, Sensitivity


def _meta_to_record(chunk: Chunk) -> dict[str, Any]:
    """Flatten a chunk's document metadata into a Chroma-storable record."""
    return {
        "doc_id": chunk.doc_id,
        "title": chunk.meta.title,
        "department": chunk.meta.department,
        "sensitivity": chunk.meta.sensitivity.value,
        "owner": chunk.meta.owner,
        "effective_date": chunk.meta.effective_date,
        "ordinal": chunk.ordinal,
    }


def _record_to_chunk(chunk_id: str, text: str, record: dict[str, Any]) -> Chunk:
    """Rebuild a Chunk from a stored Chroma record."""
    meta = DocumentMeta(
        doc_id=record["doc_id"],
        title=record["title"],
        department=record["department"],
        sensitivity=Sensitivity(record["sensitivity"]),
        owner=record["owner"],
        effective_date=record["effective_date"],
    )
    return Chunk(
        chunk_id=chunk_id,
        doc_id=record["doc_id"],
        text=text,
        meta=meta,
        ordinal=int(record["ordinal"]),
    )


def _acl_where(max_sensitivity: Sensitivity, department: str | None) -> dict[str, Any]:
    """Build a Chroma ``where`` filter enforcing clearance and optional department."""
    allowed = [s.value for s in allowed_sensitivities(max_sensitivity)]
    conditions: list[dict[str, Any]] = [{"sensitivity": {"$in": allowed}}]
    if department is not None:
        conditions.append({"department": {"$eq": department}})
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


class VectorStore:
    """Chroma-backed store of embedded chunks with metadata filtering."""

    def __init__(self, persist_dir: Path = CHROMA_DIR, collection: str = "docs") -> None:
        """Open (or create) a persistent Chroma collection using cosine space."""
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )

    def upsert(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """Insert or update chunks and their vectors, storing metadata + ACL."""
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("chunks and embeddings must have equal length")
        self._collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=embeddings.tolist(),
            documents=[chunk.text for chunk in chunks],
            metadatas=[_meta_to_record(chunk) for chunk in chunks],
        )

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        max_sensitivity: Sensitivity,
        department: str | None = None,
    ) -> list[RetrievalResult]:
        """Return the top_k chunks at or below max_sensitivity (ACL enforced)."""
        result = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=_acl_where(max_sensitivity, department),
            include=["documents", "metadatas", "distances"],
        )
        ids = result["ids"][0]
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]
        results: list[RetrievalResult] = []
        for chunk_id, text, record, distance in zip(ids, documents, metadatas, distances):
            score = max(0.0, 1.0 - float(distance))
            results.append(RetrievalResult(chunk=_record_to_chunk(chunk_id, text, record), score=score))
        return results

    def all_chunks(self) -> list[Chunk]:
        """Return every stored chunk (used to build the lexical BM25 index)."""
        result = self._collection.get(include=["documents", "metadatas"])
        return [
            _record_to_chunk(chunk_id, text, record)
            for chunk_id, text, record in zip(
                result["ids"], result["documents"], result["metadatas"]
            )
        ]

    def count(self) -> int:
        """Number of chunks currently stored."""
        return self._collection.count()
