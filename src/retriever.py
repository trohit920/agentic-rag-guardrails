"""Retrieval strategies over the vector store.

Implements dense (embedding-only), hybrid (dense + BM25 fused with reciprocal-rank
fusion), and metadata-filtered retrieval. Access control is enforced at the store
layer and, for the lexical path, mirrored here so BM25 never surfaces a chunk above
the user's clearance. This is the surface the chunking/embedding/search ablation
study varies.
"""

from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi

from .config import (
    DEFAULT_SEARCH_STRATEGY,
    HYBRID_DENSE_WEIGHT,
    TOP_K,
    SearchStrategy,
    allowed_sensitivities,
)
from .embeddings import Embedder
from .schemas import Chunk, RetrievalResult, Sensitivity
from .vector_store import VectorStore

_RRF_K: int = 60  # reciprocal-rank-fusion smoothing constant
_HYBRID_POOL_MULTIPLIER: int = 4  # candidate pool size relative to top_k
_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    """Lowercase word-token split used for BM25 scoring."""
    return _TOKEN_RE.findall(text.lower())


class Retriever:
    """Runs a configurable search strategy against the vector store."""

    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        strategy: SearchStrategy = DEFAULT_SEARCH_STRATEGY,
    ) -> None:
        """Bind a store + embedder and select a search strategy."""
        self.store = store
        self.embedder = embedder
        self.strategy = strategy
        self._chunks: list[Chunk] | None = None

    def _ensure_chunks(self) -> list[Chunk]:
        """Lazily load and cache all chunks (needed for the lexical BM25 index)."""
        if self._chunks is None:
            self._chunks = self.store.all_chunks()
        return self._chunks

    def retrieve(
        self,
        query: str,
        max_sensitivity: Sensitivity,
        top_k: int = TOP_K,
        department: str | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve top_k chunks for a query under the configured strategy + ACL."""
        if self.strategy is SearchStrategy.DENSE:
            return self._dense(query, max_sensitivity, top_k, department)
        if self.strategy is SearchStrategy.METADATA:
            return self._dense(query, max_sensitivity, top_k, department)
        if self.strategy is SearchStrategy.HYBRID:
            return self._hybrid(query, max_sensitivity, top_k, department)
        raise ValueError(f"unknown search strategy: {self.strategy}")

    def _dense(
        self, query: str, max_sensitivity: Sensitivity, top_k: int, department: str | None
    ) -> list[RetrievalResult]:
        """Embedding-similarity retrieval delegated to the store (ACL enforced)."""
        embedding = self.embedder.encode_query(query)
        return self.store.query(embedding, top_k, max_sensitivity, department)

    def max_relevance(self, query: str, results: list[RetrievalResult]) -> float:
        """Max cosine similarity between the query and the retrieved chunk texts.

        Used as a retrieval-confidence signal: a low value means the accessible
        context is off-topic, so the agent should abstain rather than confabulate.
        """
        if not results:
            return 0.0
        query_vector = self.embedder.encode_query(query)
        chunk_vectors = self.embedder.encode_documents([r.chunk.text for r in results])
        return float(np.max(chunk_vectors @ query_vector))

    def _accessible_chunks(
        self, max_sensitivity: Sensitivity, department: str | None
    ) -> list[Chunk]:
        """Filter all chunks to those the user may read (clearance + department)."""
        allowed = {s.value for s in allowed_sensitivities(max_sensitivity)}
        chunks = [c for c in self._ensure_chunks() if c.meta.sensitivity.value in allowed]
        if department is not None:
            chunks = [c for c in chunks if c.meta.department == department]
        return chunks

    def _hybrid(
        self, query: str, max_sensitivity: Sensitivity, top_k: int, department: str | None
    ) -> list[RetrievalResult]:
        """Fuse dense and BM25 rankings with weighted reciprocal-rank fusion."""
        pool = top_k * _HYBRID_POOL_MULTIPLIER
        dense_results = self._dense(query, max_sensitivity, pool, department)
        dense_ranking = [r.chunk.chunk_id for r in dense_results]

        accessible = self._accessible_chunks(max_sensitivity, department)
        by_id = {c.chunk_id: c for c in accessible}
        bm25 = BM25Okapi([_tokenize(c.text) for c in accessible])
        scores = bm25.get_scores(_tokenize(query))
        lexical_ranking = [
            accessible[i].chunk_id
            for i in sorted(range(len(accessible)), key=lambda j: scores[j], reverse=True)[:pool]
        ]

        fused = self._reciprocal_rank_fusion(dense_ranking, lexical_ranking)
        dense_by_id = {r.chunk.chunk_id: r.chunk for r in dense_results}
        ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [
            RetrievalResult(chunk=dense_by_id.get(cid) or by_id[cid], score=score)
            for cid, score in ordered
        ]

    @staticmethod
    def _reciprocal_rank_fusion(
        dense_ranking: list[str], lexical_ranking: list[str]
    ) -> dict[str, float]:
        """Combine two rankings into fused scores, weighting the dense signal."""
        fused: dict[str, float] = {}
        for rank, chunk_id in enumerate(dense_ranking):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + HYBRID_DENSE_WEIGHT / (_RRF_K + rank + 1)
        for rank, chunk_id in enumerate(lexical_ranking):
            weight = 1.0 - HYBRID_DENSE_WEIGHT
            fused[chunk_id] = fused.get(chunk_id, 0.0) + weight / (_RRF_K + rank + 1)
        return fused
