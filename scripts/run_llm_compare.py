"""Two-model comparison: Qwen2.5-1.5B (fp16) vs Phi-3-mini (4-bit).

Holds retrieval and guardrails fixed and swaps only the generator, evaluating
answer token-F1, NLI faithfulness, abstention accuracy, median latency, throughput
(tokens/sec), and peak VRAM. Models are loaded one at a time to fit 6 GB. Writes
outputs/compare.json with a recommendation. Answers the "compare two LLMs on
accuracy, latency and cost" item; cost is compute (tokens/sec + VRAM), since both
models run locally at no monetary cost.

Run from the project root:
    python scripts/run_llm_compare.py
"""

from __future__ import annotations

import gc
import json
import pathlib
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from src.agent import RagAgent
from src.config import (
    GENERATION_MODEL_A,
    GENERATION_MODEL_B,
    OUTPUT_DIR,
    SearchStrategy,
)
from src.embeddings import Embedder
from src.eval_answer import abstention_correct, token_f1
from src.eval_data import GoldQuestion, load_gold
from src.guardrails import GroundednessGuard, InjectionGuard
from src.indexing import build_vector_store
from src.llm import LocalLLM
from src.retriever import Retriever
from src.vector_store import VectorStore

COMPARE_PERSIST = OUTPUT_DIR / "chroma_eval"
MODELS = (
    ("qwen2.5-1.5b-fp16", GENERATION_MODEL_A, False),
    ("phi-3-mini-4bit", GENERATION_MODEL_B, True),
)


def _evaluate_model(agent: RagAgent, retriever: Retriever, grounded: GroundednessGuard,
                    gold: list[GoldQuestion]) -> dict:
    """Run the gold set through one agent and aggregate quality + speed metrics."""
    f1_scores, faith_scores, latencies, throughputs = [], [], [], []
    abstention_hits = 0
    for question in gold:
        result = agent.answer(question.question, question.required_clearance)
        abstention_hits += int(abstention_correct(result.abstained, question.answerable))
        if question.answerable and not result.abstained:
            retrieved = retriever.retrieve(question.question, question.required_clearance)
            f1_scores.append(token_f1(result.answer, question.answer))
            faith_scores.append(grounded.score(result.answer, retrieved).grounded_ratio)
            latencies.append(result.latency_seconds)
            if result.latency_seconds > 0:
                throughputs.append(result.completion_tokens / result.latency_seconds)
    return {
        "n_answered": len(f1_scores),
        "mean_token_f1": round(statistics.fmean(f1_scores), 4) if f1_scores else 0.0,
        "mean_faithfulness": round(statistics.fmean(faith_scores), 4) if faith_scores else 0.0,
        "abstention_accuracy": round(abstention_hits / len(gold), 4),
        "median_latency_s": round(statistics.median(latencies), 3) if latencies else 0.0,
        "mean_tokens_per_s": round(statistics.fmean(throughputs), 1) if throughputs else 0.0,
    }


def _recommend(results: dict[str, dict]) -> str:
    """Pick a model: higher answer quality, noting the speed/VRAM trade-off."""
    names = list(results)

    def quality(name: str) -> float:
        metrics = results[name]
        return metrics["mean_token_f1"] + metrics["mean_faithfulness"] + metrics["abstention_accuracy"]

    best = max(names, key=quality)
    fastest = min(names, key=lambda n: results[n]["median_latency_s"] or float("inf"))
    lowest_vram = min(names, key=lambda n: results[n]["peak_vram_gb"] or float("inf"))
    if best == fastest:
        recommendation = (
            f"{best} is recommended: it leads on answer quality and has the lowest median latency "
            f"({results[best]['median_latency_s']}s)."
        )
        if lowest_vram != best:
            recommendation += (
                f" {lowest_vram} uses less VRAM ({results[lowest_vram]['peak_vram_gb']} GB) and a "
                f"higher abstention rate, so it may suit tighter memory or stricter-refusal deployments."
            )
        return recommendation
    return (
        f"{best} leads on answer quality while {fastest} has the lowest median latency "
        f"({results[fastest]['median_latency_s']}s). Recommend {best} for this grounded-QA workload "
        f"unless latency or VRAM is the binding constraint."
    )


def main() -> None:
    """Run the two-model comparison and persist outputs/compare.json."""
    gold = load_gold()
    embedder = Embedder()
    store = VectorStore(persist_dir=COMPARE_PERSIST, collection="eval")
    if store.count() == 0:
        build_vector_store(store, embedder)
    retriever = Retriever(store, embedder, strategy=SearchStrategy.HYBRID)
    injection = InjectionGuard(embedder)
    grounded = GroundednessGuard(embedder=embedder)

    results: dict[str, dict] = {}
    for name, model_id, load_in_4bit in MODELS:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        llm = LocalLLM(model_id, load_in_4bit=load_in_4bit)
        agent = RagAgent(retriever, llm, injection, grounded)
        metrics = _evaluate_model(agent, retriever, grounded, gold)
        metrics["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        results[name] = metrics
        print(f"{name}: {metrics}")
        del agent, llm
        gc.collect()
        torch.cuda.empty_cache()

    report = {"models": results, "recommendation": _recommend(results)}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "compare.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {OUTPUT_DIR / 'compare.json'}")


if __name__ == "__main__":
    main()
