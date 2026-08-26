"""Retrieval-quality metrics.

Pure functions computing standard information-retrieval metrics (hit@k, MRR,
nDCG@k) from a ranked list of ids against a set of gold-relevant ids. No model
I/O, fully deterministic, and unit-tested. Implements the "evaluation metrics
for retrieval quality" without any external judge. Relevance is binary and ids are
document ids, so the metrics are independent of the chunking strategy.
"""

from __future__ import annotations

import math

from .config import EVAL_K_VALUES


def hit_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """1.0 if any relevant id appears in the top-k, else 0.0."""
    return 1.0 if any(rid in relevant_ids for rid in ranked_ids[:k]) else 0.0


def reciprocal_rank(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    """Reciprocal of the rank of the first relevant id (0.0 if none present)."""
    for index, rid in enumerate(ranked_ids):
        if rid in relevant_ids:
            return 1.0 / (index + 1)
    return 0.0


def ndcg_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Normalized discounted cumulative gain over the top-k (binary relevance)."""
    dcg = sum(
        1.0 / math.log2(index + 2)
        for index, rid in enumerate(ranked_ids[:k])
        if rid in relevant_ids
    )
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(index + 2) for index in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def aggregate_retrieval(
    rankings: list[tuple[list[str], set[str]]], k_values: tuple[int, ...] = EVAL_K_VALUES
) -> dict[str, float]:
    """Aggregate hit@k / MRR / nDCG@k across a set of (ranked, gold) pairs."""
    if not rankings:
        return {}
    metrics: dict[str, float] = {}
    for k in k_values:
        metrics[f"hit@{k}"] = sum(hit_at_k(r, g, k) for r, g in rankings) / len(rankings)
        metrics[f"ndcg@{k}"] = sum(ndcg_at_k(r, g, k) for r, g in rankings) / len(rankings)
    metrics["mrr"] = sum(reciprocal_rank(r, g) for r, g in rankings) / len(rankings)
    return metrics
