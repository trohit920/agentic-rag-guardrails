"""Unit tests for the pure retrieval and answer evaluation metrics."""

from __future__ import annotations

import math

from src.eval_answer import abstention_correct, token_f1
from src.eval_retrieval import aggregate_retrieval, hit_at_k, ndcg_at_k, reciprocal_rank


def test_hit_at_k_respects_cutoff() -> None:
    """A relevant id beyond k does not count as a hit."""
    assert hit_at_k(["a", "b", "c"], {"c"}, k=3) == 1.0
    assert hit_at_k(["a", "b", "c"], {"c"}, k=2) == 0.0


def test_reciprocal_rank_uses_first_relevant() -> None:
    """MRR reflects the rank of the first relevant id."""
    assert reciprocal_rank(["a", "b", "c"], {"b"}) == 0.5
    assert reciprocal_rank(["a", "b"], {"z"}) == 0.0


def test_ndcg_is_one_when_relevant_first() -> None:
    """A relevant id at rank 1 yields perfect nDCG; later ranks score lower."""
    assert math.isclose(ndcg_at_k(["a", "b"], {"a"}, k=2), 1.0)
    assert ndcg_at_k(["b", "a"], {"a"}, k=2) < 1.0


def test_aggregate_retrieval_averages_across_queries() -> None:
    """Aggregation averages each metric over all (ranked, gold) pairs."""
    rankings = [(["a", "b"], {"a"}), (["x", "y"], {"y"})]
    metrics = aggregate_retrieval(rankings, k_values=(1,))
    assert metrics["hit@1"] == 0.5  # first query hits at 1, second does not
    assert math.isclose(metrics["mrr"], (1.0 + 0.5) / 2)


def test_token_f1_scores_overlap() -> None:
    """Token F1 is 1.0 for an exact match and 0.0 for no overlap."""
    assert token_f1("USD 60 per day", "USD 60 per day") == 1.0
    assert token_f1("USD 60", "nothing here") == 0.0
    assert 0.0 < token_f1("the allowance is USD 60", "USD 60") < 1.0


def test_abstention_correct_matches_answerability() -> None:
    """Abstaining is correct exactly when the question is unanswerable."""
    assert abstention_correct(abstained=True, is_answerable=False) is True
    assert abstention_correct(abstained=False, is_answerable=True) is True
    assert abstention_correct(abstained=True, is_answerable=True) is False
