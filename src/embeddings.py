"""Local sentence-embedding encoder for retrieval.

Thin adapter over a local sentence-transformers model (bge-small). Applies the
bge query prefix for asymmetric search and returns L2-normalized vectors so that
inner product equals cosine similarity. Runs on GPU when available, otherwise CPU.
"""

from __future__ import annotations

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from .config import BGE_QUERY_PREFIX, EMBEDDING_MODEL


class Embedder:
    """Encodes documents and queries into normalized dense vectors."""

    def __init__(self, model_id: str = EMBEDDING_MODEL, device: str | None = None) -> None:
        """Load the embedding model onto the requested device (auto if None)."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_id, device=self.device)

    @property
    def dimension(self) -> int:
        """Embedding vector dimensionality."""
        return int(self.model.get_sentence_embedding_dimension())

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        """Encode raw document chunks; returns (n, dim) normalized float32 array."""
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a query with the bge prefix; returns (dim,) normalized vector."""
        vector = self.model.encode(
            f"{BGE_QUERY_PREFIX}{query}",
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vector.astype(np.float32)
