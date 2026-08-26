"""Answer-quality metrics (local, no LLM judge).

Scores generated answers against gold answers using token-level F1 (SQuAD-style
overlap) and optional embedding semantic similarity, and scores abstention
correctness on unanswerable questions. NLI-based faithfulness is provided by the
GroundednessGuard and reused by the eval harness. Implements the requirement for
"generated-answer quality" metrics locally, with no external judge.
"""

from __future__ import annotations

import re
import string

import numpy as np

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _normalize(text: str) -> list[str]:
    """Lowercase, strip punctuation and articles, and split into tokens (SQuAD)."""
    lowered = text.lower().translate(_PUNCT_TABLE)
    return _ARTICLES_RE.sub(" ", lowered).split()


def token_f1(prediction: str, gold: str) -> float:
    """Token-overlap F1 between prediction and gold answer (SQuAD-style)."""
    pred_tokens = _normalize(prediction)
    gold_tokens = _normalize(gold)
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common: dict[str, int] = {}
    for token in pred_tokens:
        if token in gold_tokens:
            common[token] = min(pred_tokens.count(token), gold_tokens.count(token))
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def semantic_similarity(prediction: str, gold: str, embedder: object) -> float:
    """Cosine similarity between prediction and gold answer embeddings."""
    vectors = embedder.encode_documents([prediction, gold])  # type: ignore[attr-defined]
    return float(np.dot(vectors[0], vectors[1]))


def abstention_correct(abstained: bool, is_answerable: bool) -> bool:
    """True when the model abstained iff the question was unanswerable."""
    return abstained == (not is_answerable)
