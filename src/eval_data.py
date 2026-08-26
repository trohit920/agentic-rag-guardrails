"""Gold evaluation set loader.

Parses ``eval/gold.json`` into typed GoldQuestion records used by the ablation
study, the answer-quality evaluation, and the two-model comparison. Pure I/O
adapter with no model dependencies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .config import EVAL_DIR, Sensitivity

GOLD_FILENAME: str = "gold.json"


@dataclass(frozen=True)
class GoldQuestion:
    """A single evaluation question with its gold answer and metadata."""

    id: str
    question: str
    answer: str
    relevant_doc_ids: list[str]
    answerable: bool
    required_clearance: Sensitivity
    category: str


def load_gold(eval_dir: Path = EVAL_DIR) -> list[GoldQuestion]:
    """Load the gold question set from ``gold.json``."""
    gold_path = eval_dir / GOLD_FILENAME
    with gold_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return [
        GoldQuestion(
            id=entry["id"],
            question=entry["question"],
            answer=entry["answer"],
            relevant_doc_ids=entry["relevant_doc_ids"],
            answerable=entry["answerable"],
            required_clearance=Sensitivity(entry["required_clearance"]),
            category=entry["category"],
        )
        for entry in payload["questions"]
    ]
