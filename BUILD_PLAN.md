# Agentic RAG with Guardrails (Build Plan / Design Doc)

The design document written before the code. It fixes the requirements, the module
boundaries, and the phase plan, and is preserved here so the reasoning behind the
build is inspectable alongside the result.

Constraint: fully local, open-source models. No paid APIs, no network at
inference once models are cached. Developed on a local NVIDIA GPU (~6 GB VRAM).
Style: `src/` library + `config.py` + `scripts/` + `tests/` + `outputs/` +
README/report/notebook, SEED=42.

---

## 1. Goal

Build a small but production-shaped Agentic RAG assistant over a financial /
enterprise document corpus that satisfies the full set of capability requirements
below, end to end, running entirely on local open models. The output is a GitHub
repo plus a written report where every claim is backed by a number the reader can
reproduce.

The differentiator: hallucination guardrails and answer-quality evaluation are
built on the NLI-groundedness and semantic-entropy machinery proven in
[llm-uncertainty-quantification](https://github.com/trohit920/llm-uncertainty-quantification),
rather than on an LLM judge.

---

## 2. Requirement -> feature traceability

The capability requirements for a production enterprise RAG system, each mapped to
the module that delivers it. Writing this table before any code is what kept the
scope from drifting into a demo.

| # | Requirement | Feature delivering it | Module |
|---|---------|-----------------------|--------|
| 1 | Agentic RAG + explain end-to-end flow | LangGraph agent; report + notebook walk the flow | `agent.py`, `report.md` |
| 2 | Why docs not retrieved; improve chunking/embeddings/metadata/search | Ablation study over chunk size, embedding model, metadata filter, dense vs hybrid | `ingest.py`, `retriever.py`, `scripts/run_ablation.py` |
| 3 | Agent chooses between tools/APIs | Router node picks: retrieve / calculator / metadata-filtered-retrieve / refuse | `agent.py`, `tools.py` |
| 4 | Handle incorrect/hallucinated response + guardrails | NLI-groundedness check + semantic-entropy abstention + citation enforcement | `guardrails.py` |
| 5 | Write/optimize a Python function (doc processing / validation) | Documented, vectorized chunker + a before/after optimization note | `ingest.py`, report |
| 6 | Compare two LLMs (accuracy/latency/cost) | Qwen2.5-1.5B vs Llama-3.2-1B; cost = tokens/sec + VRAM (no dollars) | `scripts/run_llm_compare.py` |
| 7 | Production arch: LLM+vectorDB+API+auth+monitoring | FastAPI + API-key auth + Chroma + structured JSON logging/metrics | `api/main.py` |
| 8 | Conversation history + token limits | Token-budgeted context assembly + rolling summarization | `context.py` |
| 9 | Error handling: API fail/timeout/rate-limit/missing source | Retry/timeout wrappers + graceful degradation + typed exceptions | `llm.py`, `api/main.py` |
| 10 | Retrieval + answer eval metrics | Retrieval: hit@k, MRR, nDCG. Answer: NLI-faithfulness, token-F1, semantic sim | `eval_retrieval.py`, `eval_answer.py` |
| 11 | Sensitive data / access control / prompt injection | Per-user doc ACL on retrieval + injection detector (rules + classifier) | `guardrails.py`, `retriever.py` |
| 12 | Review a solution, find bottlenecks, propose improvements | "Limitations & next steps" section grounded in the ablation numbers | `report.md` |
| 13 | Explain exact contribution + measurable outcomes | Report frames decisions + metrics deltas | `report.md` |
| 14 | Present to technical + business stakeholders | Two-audience README: exec summary + technical deep-dive | `README.md` |
| 15 | Hands-on Agentic frameworks (LangGraph) | LangGraph is the orchestration layer, not hand-rolled | `agent.py` |

Every requirement is covered locally. RAGAS is deliberately excluded (it defaults
to an OpenAI judge); the equivalent metrics are implemented locally with the same
NLI approach.

---

## 3. Architecture / data flow

```
                         +-------------------+
  user query ----------> |  FastAPI  /chat   |  API-key auth, request logging
                         +---------+---------+
                                   |
                                   v
                      +------------------------+
                      |  LangGraph agent       |
                      |  1. injection guard    |
                      |  2. router (tool pick) |
                      |  3. retrieve (ACL)     |
                      |  4. context assembly   |
                      |  5. generate (local LLM)|
                      |  6. groundedness guard |
                      +-----+-----------+------+
                            |           |
             (dense/hybrid) |           | (NLI + semantic entropy)
                            v           v
                   +--------------+  +-----------------+
                   |  Chroma      |  |  answer + cites |
                   |  vector DB   |  |  or  "abstain"  |
                   +------+-------+  +-----------------+
                          ^
              ingest.py   |  chunk -> embed (bge-small) -> upsert w/ metadata+ACL
        financial docs ---+
```

Offline (scripts): `run_ablation.py` and `run_llm_compare.py` write metrics +
figures to `outputs/`, feeding `report.md` and `analysis.ipynb`.

---

## 4. Tech stack (all local) + VRAM budget

| Concern | Choice | Why | Approx VRAM |
|---------|--------|-----|-------------|
| Generation LLM (A) | `Qwen/Qwen2.5-1.5B-Instruct` | Strong for its size | ~3.0 GB (fp16) |
| Generation LLM (B) | `microsoft/Phi-3-mini-4k-instruct` (4-bit) | Stronger-reasoning contrast; MIT-licensed, ungated | ~2.3 GB (4-bit NF4) |
| Embeddings | `BAAI/bge-small-en-v1.5` | Small, strong retrieval, local | ~0.2 GB |
| NLI (guardrails/eval) | `cross-encoder/nli-deberta-v3-small` | Reused from llm-uncertainty-quantification | CPU-friendly |
| Injection classifier | rules + `bge` similarity to known-attack set | No extra model needed | - |
| Vector DB | Chroma (persistent, local dir) | Zero external service, metadata filters | - |
| Agent framework | LangGraph | Standard agent orchestration | - |
| API | FastAPI + uvicorn | Standard production shape | - |
| Monitoring | structured JSON logs + metrics json in outputs/ | Local, no SaaS | - |

Models are loaded one at a time for the comparison to stay within 6 GB. Phi-3-mini
is loaded in 4-bit NF4 via bitsandbytes (fp16 would be ~7.6 GB). Chunking/
embedding runs on GPU; NLI on CPU to leave VRAM for generation.

---

## 5. Corpus + evaluation set (financial / enterprise)

Since real proprietary financial docs cannot be shipped, generate a small,
self-contained synthetic corpus (10-20 documents) that looks enterprise-real:
- Sample: expense/travel policy, information-security policy, KYC/AML procedure,
  data-retention standard, vendor-onboarding SOP, a quarterly-report extract,
  an access-control matrix.
- Each doc gets metadata: `doc_id`, `title`, `department`, `sensitivity`
  (public/internal/confidential), `owner`, `effective_date`.
- ACL demo: some docs marked `confidential`; a user without clearance must not
  retrieve them (proves access control).

Eval question set (~30-40 Q/A pairs), hand-authored with gold answers + the
doc/chunk that supports them, so retrieval and answer metrics are computable:
- factual lookups, multi-doc synthesis, a calculator-needing question, and
  several unanswerable / adversarial (prompt-injection) questions to test
  abstention and guardrails.

Everything committed under `data/` and `eval/` so the repo is reproducible with
no downloads beyond the models.

---

## 6. Module design (one responsibility each, <500 lines, <~50-line functions)

- `config.py` - model ids, chunk sizes, top-k, thresholds, paths, `SEED=42`.
- `llm.py` - local HF chat wrapper (adapted from `UncertaintyLM`): `generate()`,
  timeout/retry, token counting; exposes latency + token stats for the compare.
- `embeddings.py` - batch encode with bge; L2-normalized vectors.
- `vector_store.py` - Chroma wrapper: upsert, query, metadata + ACL filtering.
- `ingest.py` - loaders + chunking strategies (fixed, recursive, sentence-aware);
  the optimized document-processing function lives here.
- `retriever.py` - dense / hybrid (dense + BM25) / metadata-filtered search;
  ACL enforcement applied here before results leave the store.
- `tools.py` - agent tools: `retrieve`, `metadata_retrieve`, `calculator`.
- `agent.py` - LangGraph graph wiring guard -> route -> retrieve -> assemble ->
  generate -> groundedness-guard, with typed state.
- `guardrails.py` - injection detection; NLI-groundedness scoring of answer vs
  retrieved context; semantic-entropy abstention; citation check.
- `context.py` - token-budgeted history assembly + rolling summarization.
- `eval_retrieval.py` - hit@k, MRR, nDCG (pure functions, unit-tested).
- `eval_answer.py` - NLI-faithfulness, token-F1, semantic similarity.
- `api/main.py` - FastAPI: `/ingest`, `/chat`; API-key auth dependency;
  timeout/rate-limit handling; per-request structured logs.

---

## 7. LangGraph agent graph

State: `{query, user, history, route, retrieved, answer, citations, verdict}`.

Nodes:
1. `guard_input` - reject/flag prompt injection; short-circuit to safe refusal.
2. `route` - classify query -> {needs_retrieval, needs_calculator, chit-chat};
   demonstrates tool selection .
3. `retrieve` - ACL-filtered search; sets `retrieved` (may be empty).
4. `assemble_context` - token-budgeted packing of history + chunks .
5. `generate` - local LLM answer with citation instructions.
6. `guard_output` - NLI-groundedness + semantic entropy; if ungrounded or
   high-uncertainty -> abstain ("I don't have enough information") with reason.

Conditional edges: empty retrieval -> abstain; calculator route -> tool node
-> generate. This is the "explain the end-to-end flow" artifact .

---

## 8. Guardrails

- Prompt injection : rule set (instruction-override phrases, role hijack,
  exfiltration asks) + embedding similarity to a small known-attack corpus.
- Groundedness / hallucination : score each answer sentence for
  entailment against retrieved chunks with `nli-deberta-v3-small`; unsupported
  sentences are flagged or the answer is downgraded to abstain.
- Uncertainty abstention: semantic-entropy / self-consistency
  to catch confident-but-wrong answers; abstain above a tuned threshold.
- Citation enforcement: every claim must map to a chunk id or it is dropped.
- Access control : retrieval filters by `sensitivity` vs user clearance.

---

## 9. Evaluation plan

Retrieval (`eval_retrieval.py`): hit@k, MRR, nDCG against gold supporting chunks.
Answer (`eval_answer.py`): NLI-faithfulness, token-F1 vs gold, semantic
similarity; abstention correctness on unanswerable set (did it correctly refuse?).
All metrics are pure functions with unit tests and fixed seeds. Results ->
`outputs/eval.json` + figures, discussed in `report.md`.

---

## 10. LLM comparison 

Same eval set, same retrieval, swap only the generator: Qwen2.5-1.5B (fp16) vs
Phi-3-mini (4-bit). Report a table of: answer-F1, faithfulness, abstention accuracy,
median latency (ms/query), throughput (tokens/sec), peak VRAM. "Cost" is
expressed as compute (tokens/sec + VRAM), not dollars, since both are local.
Recommendation with justification, in your CLAUDE.md trade-off style.

---

## 11. Phased milestones (each ends with a runnable checkpoint)

- **Phase 0 - Scaffold**: repo structure, `config.py`, `requirements.txt`,
  README skeleton, empty modules with docstrings + type stubs. Checkpoint: imports
  + `pytest` collect pass.
- **Phase 1 - Ingest + retrieve**: corpus, chunking, embeddings, Chroma, dense
  retrieval. Checkpoint: `scripts/demo.py` retrieves relevant chunks for a query.
- **Phase 2 - Agent + generation**: LangGraph graph, local LLM answer with
  citations. Checkpoint: end-to-end Q -> grounded answer via CLI.
- **Phase 3 - Guardrails**: injection, NLI-groundedness, abstention, ACL.
  Checkpoint: adversarial + confidential queries handled correctly.
- **Phase 4 - Eval + ablation**: retrieval/answer metrics, chunking/embedding/
  search ablation. Checkpoint: `outputs/eval.json` + ablation figures.
- **Phase 5 - LLM comparison**: two-model table. Checkpoint: `outputs/compare.json`.
- **Phase 6 - API**: FastAPI + auth + error handling + logging. Checkpoint:
  `/chat` works with an API key; unauthorized rejected.
- **Phase 7 - Write-up**: `report.md`, `analysis.ipynb`, two-audience README,
  architecture diagram. Checkpoint: repo reads cleanly top to bottom.

Estimated ~5-7 focused days given the reusable NLI/uncertainty assets.

---

## 12. Testing

- Pure logic (chunking, metrics, token budgeting, citation parsing, injection
  rules) has deterministic, seeded unit tests - no model I/O in unit tests.
- Model-dependent paths validated via small scripted smoke runs (like the
  `spikes/smoke_test.py`).
- Follow the self-improving loop: write -> `pytest -q` + `ruff check` -> fix ->
  repeat; never weaken a test to make it pass.

---

## 13. Risks & mitigations

- **VRAM (6 GB)**: load one generator at a time; NLI on CPU; small embed batches.
- **Llama-3.2 gated download**: needs HF license acceptance; fallback
  `Qwen2.5-0.5B-Instruct` or `Phi-3-mini` if access is blocked.
- **Small-model answer quality**: guardrails/abstention turn low quality into
  honest refusals - which is itself a strength, not a failure.
- **Scope creep**: API/monitoring kept minimal-but-real; no k8s/cloud theater.

---

## 14. Resume / GitHub framing

- Repo name: `agentic-rag-guardrails` (or similar).
- One-line: "Local, fully open-source Agentic RAG over enterprise documents with
  NLI-based hallucination guardrails, access control, and a retrieval+answer
  evaluation harness - LangGraph, Chroma, FastAPI."
- Resume bullet: quantified deltas from the ablation + the LLM comparison table.
- Keeps the driver-monitoring CV project (`cabinsense`) as the separate CV flagship;
  this repo is self-contained and does not depend on it.

---

## Resolved decisions (locked)

1. Folder/repo name: **`agentic-rag-guardrails`**.
2. Corpus: try a small, cleanly-licensed **public financial dataset first**;
   fall back to a **hand-authored synthetic** corpus if none drops in easily.
3. Models: **non-gated only**. Generator A = `Qwen/Qwen2.5-1.5B-Instruct` (fp16),
   Generator B = `microsoft/Phi-3-mini-4k-instruct` (4-bit NF4 via bitsandbytes).
4. **RAGAS excluded.** RAGAS's core metrics (faithfulness, answer relevancy,
   context precision/recall) use an LLM judge that defaults to OpenAI and requires
   a paid `OPENAI_API_KEY`. It can be pointed at a local judge, but tiny local
   models judge unreliably and it adds a fragile LangChain layer. We implement the
   same metric *ideas* locally: NLI-based faithfulness (`nli-deberta-v3-small`,
   reused) + standard IR math (hit@k, MRR, nDCG). Optional
   RAGAS-with-local-judge variant can be added later if desired for resume
   name-recognition.
