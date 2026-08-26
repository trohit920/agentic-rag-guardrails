"""End-to-end answer-quality and safety evaluation.

Runs the full agent (Qwen2.5-1.5B + guards) over the gold set and reports
retrieval metrics, answer token-F1, NLI faithfulness, abstention accuracy, and an
ACL-denial sub-evaluation (confidential questions asked under-clearance must be
refused, never leaked). Writes outputs/eval.json. This is the "evaluation
metrics for retrieval and generated-answer quality", computed fully locally.

Run from the project root:
    python scripts/run_eval.py
"""

from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agent import RagAgent
from src.config import OUTPUT_DIR, SearchStrategy, Sensitivity
from src.embeddings import Embedder
from src.eval_answer import abstention_correct, token_f1
from src.eval_data import GoldQuestion, load_gold
from src.eval_retrieval import aggregate_retrieval
from src.guardrails import GroundednessGuard, InjectionGuard
from src.indexing import build_vector_store
from src.llm import LocalLLM
from src.retriever import Retriever
from src.vector_store import VectorStore

EVAL_PERSIST = OUTPUT_DIR / "chroma_eval"


def _ordered_doc_ids(results: list) -> list[str]:
    """Deduplicate a chunk ranking into an ordered list of document ids."""
    seen: list[str] = []
    for result in results:
        if result.chunk.doc_id not in seen:
            seen.append(result.chunk.doc_id)
    return seen


def _retrieval_metrics(retriever: Retriever, answerable: list[GoldQuestion]) -> dict[str, float]:
    """Aggregate retrieval metrics over answerable questions."""
    rankings = []
    for question in answerable:
        results = retriever.retrieve(question.question, question.required_clearance)
        rankings.append((_ordered_doc_ids(results), set(question.relevant_doc_ids)))
    return aggregate_retrieval(rankings)


def _answer_and_safety(
    agent: RagAgent, retriever: Retriever, grounded: GroundednessGuard, gold: list[GoldQuestion]
) -> dict:
    """Run every question through the agent and collect quality/safety records."""
    records = []
    for question in gold:
        result = agent.answer(question.question, question.required_clearance)
        record = {
            "id": question.id,
            "category": question.category,
            "abstained": result.abstained,
            "abstention_correct": abstention_correct(result.abstained, question.answerable),
        }
        if question.answerable and not result.abstained:
            retrieved = retriever.retrieve(question.question, question.required_clearance)
            record["token_f1"] = token_f1(result.answer, question.answer)
            record["faithfulness"] = grounded.score(result.answer, retrieved).grounded_ratio
        records.append(record)
    return {"records": records}


def _acl_denial(agent: RagAgent, retriever: Retriever, gold: list[GoldQuestion]) -> dict:
    """Ask confidential questions under-clearance and verify the ACL guarantee.

    The true ACL property is enforced at retrieval: a confidential chunk must never
    reach an under-clearance user, so the model cannot leak protected text. Residual
    risk is confabulation (a wrong answer), which we count separately.
    """
    confidential = [q for q in gold if q.answerable and q.required_clearance is Sensitivity.CONFIDENTIAL]
    retrieval_leaks = refused = confabulated = 0
    for question in confidential:
        retrieved = retriever.retrieve(question.question, Sensitivity.INTERNAL)
        if any(r.chunk.meta.sensitivity is Sensitivity.CONFIDENTIAL for r in retrieved):
            retrieval_leaks += 1
        result = agent.answer(question.question, Sensitivity.INTERNAL)
        if result.abstained:
            refused += 1
        else:
            confabulated += 1
    return {
        "n": len(confidential),
        "retrieval_leaks": retrieval_leaks,
        "refused": refused,
        "confabulated": confabulated,
    }


def _summarize(records: list[dict]) -> dict:
    """Aggregate per-question records into answer-quality and abstention summaries."""
    answered = [r for r in records if "token_f1" in r]
    abstention_acc = sum(r["abstention_correct"] for r in records) / len(records)
    answer_summary = {
        "n_answered": len(answered),
        "mean_token_f1": sum(r["token_f1"] for r in answered) / max(1, len(answered)),
        "mean_faithfulness": sum(r["faithfulness"] for r in answered) / max(1, len(answered)),
    }
    return {"abstention_accuracy": abstention_acc, "answer": answer_summary}


def main() -> None:
    """Run the full evaluation and persist outputs/eval.json."""
    gold = load_gold()
    embedder = Embedder()
    store = VectorStore(persist_dir=EVAL_PERSIST, collection="eval")
    build_vector_store(store, embedder)
    retriever = Retriever(store, embedder, strategy=SearchStrategy.HYBRID)
    grounded = GroundednessGuard(embedder=embedder)
    agent = RagAgent(retriever, LocalLLM(), InjectionGuard(embedder), grounded)

    answerable = [q for q in gold if q.answerable and q.relevant_doc_ids]
    retrieval = _retrieval_metrics(retriever, answerable)
    records = _answer_and_safety(agent, retriever, grounded, gold)["records"]
    summary = _summarize(records)
    report = {
        "retrieval": retrieval,
        **summary,
        "acl_denial": _acl_denial(agent, retriever, gold),
        "per_question": records,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "eval.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "per_question"}, indent=2))
    print(f"wrote {OUTPUT_DIR / 'eval.json'}")


if __name__ == "__main__":
    main()
