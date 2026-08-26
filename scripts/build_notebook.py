"""Build and execute analysis.ipynb from the generated output artifacts.

Assembles a narrated notebook that loads outputs/{eval,ablation,compare}.json and
the ablation figure, renders them as tables/plots, and discusses the results. The
notebook only reads artifacts (no model loading), so it executes quickly. Run the
eval/ablation/compare scripts first.

Run from the project root:
    python scripts/build_notebook.py
"""

from __future__ import annotations

import pathlib

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = pathlib.Path(__file__).resolve().parent.parent

CELLS = [
    new_markdown_cell(
        "# Agentic RAG with Guardrails - Analysis\n\n"
        "Fully local, open-source agentic RAG over an enterprise/financial corpus, "
        "with access control, NLI-based hallucination guardrails, prompt-injection "
        "defense, and a judge-free evaluation harness. This notebook reads the "
        "artifacts produced by the `scripts/run_*.py` entry points."
    ),
    new_code_cell(
        "import json\n"
        "from pathlib import Path\n"
        "import pandas as pd\n"
        "from IPython.display import Image\n\n"
        "OUT = Path('outputs')\n"
        "eval_r = json.loads((OUT / 'eval.json').read_text())\n"
        "ablation = json.loads((OUT / 'ablation.json').read_text())\n"
        "compare = json.loads((OUT / 'compare.json').read_text())\n"
        "print('loaded eval, ablation, compare artifacts')"
    ),
    new_markdown_cell(
        "## 1. Retrieval ablation\n\n"
        "Chunking strategy x search strategy, scored on the answerable gold "
        "questions. On this small corpus retrieval saturates; `recursive+hybrid` is "
        "the chosen default."
    ),
    new_code_cell("pd.DataFrame(ablation).T[['hit@1', 'hit@5', 'mrr', 'ndcg@5']]"),
    new_code_cell("Image(str(OUT / 'figures' / 'ablation.png'))"),
    new_markdown_cell(
        "## 2. Answer quality and safety\n\n"
        "Faithfulness is the NLI+embedding groundedness of produced answers; "
        "abstention accuracy covers unanswerable and injection cases; ACL retrieval "
        "leaks count confidential chunks reaching an under-clearance user (target 0)."
    ),
    new_code_cell(
        "summary = {\n"
        "    'retrieval hit@1': eval_r['retrieval']['hit@1'],\n"
        "    'retrieval nDCG@5': eval_r['retrieval']['ndcg@5'],\n"
        "    'abstention accuracy': eval_r['abstention_accuracy'],\n"
        "    'mean faithfulness': eval_r['answer']['mean_faithfulness'],\n"
        "    'mean token-F1': round(eval_r['answer']['mean_token_f1'], 3),\n"
        "    'ACL retrieval leaks': eval_r['acl_denial']['retrieval_leaks'],\n"
        "}\n"
        "pd.Series(summary, name='value').to_frame()"
    ),
    new_markdown_cell(
        "## 3. Two-model comparison\n\n"
        "Retrieval and guardrails fixed; only the generator swapped. Cost is "
        "compute (tokens/sec + VRAM), since both models run locally."
    ),
    new_code_cell("pd.DataFrame(compare['models']).T"),
    new_code_cell("print(compare['recommendation'])"),
    new_markdown_cell(
        "## 4. Limitations\n\n"
        "- **Grounded-but-irrelevant confabulation** under-clearance: no data leaks "
        "(protected text never reaches the model), but the answer can be wrong. A "
        "cosine relevance gate was rejected on the data (on-topic vs off-topic "
        "similarity overlap).\n"
        "- **Small corpus** saturates retrieval metrics.\n"
        "- **token-F1** undersells correct-but-verbose answers; faithfulness is the "
        "truer signal.\n\n"
        "See `report.md` for the full discussion."
    ),
]


def main() -> None:
    """Assemble, execute, and save the analysis notebook."""
    notebook = new_notebook(cells=CELLS)
    client = NotebookClient(notebook, timeout=120, resources={"metadata": {"path": str(ROOT)}})
    client.execute()
    output_path = ROOT / "analysis.ipynb"
    nbformat.write(notebook, output_path)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
