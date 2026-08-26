"""Retrieval ablation study.

Sweeps chunking strategy and search strategy, evaluating each configuration on the
answerable gold questions with hit@k / MRR / nDCG@k. Writes outputs/ablation.json
and a bar-chart figure, answering the question "why are relevant documents not
retrieved / improve chunking, embeddings, metadata, search" item with numbers.

Run from the project root:
    python scripts/run_ablation.py
"""

from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt

from src.config import (
    FIGURE_DIR,
    OUTPUT_DIR,
    ChunkStrategy,
    SearchStrategy,
)
from src.embeddings import Embedder
from src.eval_data import load_gold
from src.eval_retrieval import aggregate_retrieval
from src.indexing import build_vector_store
from src.retriever import Retriever
from src.vector_store import VectorStore

ABLATION_PERSIST = OUTPUT_DIR / "chroma_ablation"
CHUNK_STRATEGIES = (ChunkStrategy.FIXED, ChunkStrategy.RECURSIVE, ChunkStrategy.SENTENCE)
SEARCH_STRATEGIES = (SearchStrategy.DENSE, SearchStrategy.HYBRID)
PLOT_METRICS = ("hit@1", "hit@5", "mrr", "ndcg@5")


def _ordered_doc_ids(results: list) -> list[str]:
    """Deduplicate a chunk ranking down to an ordered list of document ids."""
    seen: list[str] = []
    for result in results:
        if result.chunk.doc_id not in seen:
            seen.append(result.chunk.doc_id)
    return seen


def _evaluate(retriever: Retriever, gold: list) -> dict[str, float]:
    """Compute aggregate retrieval metrics for one retriever over the gold set."""
    rankings = []
    for question in gold:
        results = retriever.retrieve(question.question, question.required_clearance)
        rankings.append((_ordered_doc_ids(results), set(question.relevant_doc_ids)))
    return aggregate_retrieval(rankings)


def _plot(results: dict[str, dict[str, float]]) -> pathlib.Path:
    """Render a grouped bar chart of the plotted metrics per configuration."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    configs = list(results)
    x = range(len(configs))
    width = 0.2
    fig, ax = plt.subplots(figsize=(11, 5))
    for offset, metric in enumerate(PLOT_METRICS):
        ax.bar([i + offset * width for i in x], [results[c][metric] for c in configs], width, label=metric)
    ax.set_xticks([i + width * 1.5 for i in x])
    ax.set_xticklabels(configs, rotation=20, ha="right")
    ax.set_ylabel("score")
    ax.set_title("Retrieval ablation: chunking x search strategy")
    ax.legend()
    fig.tight_layout()
    path = FIGURE_DIR / "ablation.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main() -> None:
    """Run the retrieval ablation sweep and persist results + figure."""
    gold = [q for q in load_gold() if q.answerable and q.relevant_doc_ids]
    embedder = Embedder()
    results: dict[str, dict[str, float]] = {}
    for chunk_strategy in CHUNK_STRATEGIES:
        store = VectorStore(persist_dir=ABLATION_PERSIST, collection=chunk_strategy.value)
        build_vector_store(store, embedder, strategy=chunk_strategy)
        for search_strategy in SEARCH_STRATEGIES:
            retriever = Retriever(store, embedder, strategy=search_strategy)
            name = f"{chunk_strategy.value}+{search_strategy.value}"
            results[name] = _evaluate(retriever, gold)
            print(f"{name}: {results[name]}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "ablation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    figure_path = _plot(results)
    print(f"wrote {OUTPUT_DIR / 'ablation.json'} and {figure_path}")


if __name__ == "__main__":
    main()
