# Agentic RAG with Guardrails - Analysis Report

A fully local, open-source agentic retrieval-augmented-generation (RAG) system over
an enterprise/financial document corpus. It routes between tools, retrieves with
access control, generates grounded answers with citations, refuses when it cannot
ground an answer or when input looks adversarial, and is measured with a local
retrieval + answer-quality evaluation harness. No paid APIs; everything runs on a
single 6 GB GPU.

## 1. Problem and approach

Enterprise RAG has to do more than fetch-and-generate. It must respect who is
allowed to see what, refuse rather than hallucinate, resist prompt injection, and
be measurable. This project implements those requirements end to end and, crucially,
quantifies them.

The design reuses a proven asset from a prior project ([llm-uncertainty-quantification](https://github.com/trohit920/llm-uncertainty-quantification)): the DeBERTa NLI model that powered semantic-entropy clustering is
repurposed here as the faithfulness/groundedness engine. That gives a
purpose-built, deterministic hallucination check instead of asking a small
generative model to grade itself.

## 2. Architecture

```
user query
  -> FastAPI /chat        (API-key auth, per-key rate limit, timeout, JSON logging)
  -> LangGraph agent
       1. guard_input     injection detection (rules + embedding similarity)
       2. route           tool selection: retrieve vs calculator
       3. retrieve        ACL-filtered hybrid search (dense + BM25, RRF fusion)
       4. assemble        token-budgeted context + history compaction
       5. generate        local instruct LLM, numbered citations
       6. guard_output    NLI + embedding groundedness -> answer or abstain
  -> answer + citations, or a safe refusal
```

All models are local and open-source: `Qwen/Qwen2.5-1.5B-Instruct` (generation),
`BAAI/bge-small-en-v1.5` (embeddings), `cross-encoder/nli-deberta-v3-small`
(groundedness), Chroma (vector store), LangGraph (orchestration).

## 3. Corpus and evaluation set

A hand-authored corpus of 10 enterprise-financial documents (expense policy,
information-security policy, KYC/AML procedure, data-retention standard, vendor
onboarding, a Q3 financial summary, an access-control matrix, code of conduct,
remote-work policy, incident-response plan). Each document carries real metadata:
`department`, `sensitivity` (public/internal/confidential), `owner`,
`effective_date`. Synthetic content was chosen deliberately: public financial
datasets carry no clearance metadata and no chunk-level gold labels, both of which
are required to *prove* access control and to *measure* retrieval. The gold set has
16 questions spanning factual lookups, a multi-document synthesis, a calculator
case, unanswerable questions, and prompt-injection attempts.

## 4. Results

### 4.1 Retrieval ablation (`outputs/ablation.json`, `figures/ablation.png`)

Chunking strategy x search strategy, scored on the answerable gold questions.
On this small, well-separated corpus retrieval saturates: every configuration
reaches hit@1 = 1.0. The best configurations - `recursive+hybrid` and
`sentence+hybrid` - reach nDCG@5 = 1.0, so recursive chunking with hybrid
(dense + BM25) search is the chosen default. On a larger, noisier corpus the gaps
would widen; the harness is built to surface them.

### 4.2 Answer quality and safety (`outputs/eval.json`)

| Metric | Value |
|---|---|
| Retrieval hit@1 / MRR / nDCG@5 | 1.00 / 1.00 / 1.00 |
| Abstention accuracy (all 16) | 0.94 |
| Answers produced (of 12 answerable) | 11 |
| Mean faithfulness (NLI+embedding) | 1.00 |
| Mean token-F1 vs gold | 0.42 |
| ACL retrieval leaks (under-clearance) | 0 |

Notes:
- **Faithfulness 1.00** - every produced answer is grounded in retrieved context.
- **token-F1 0.42** is a surface-overlap metric penalizing correct-but-verbose
  answers against short gold strings; faithfulness is the truer answer-quality
  signal here.
- The single false abstention is the calculator-over-context question ("total meal
  reimbursement for a 4-day trip"): the computed value "240" is not an extractable
  sentence, so the groundedness guard abstains. Correct arithmetic is better served
  by the explicit calculator route.
- **Zero ACL retrieval leaks**: an under-clearance user never receives a
  confidential chunk (enforced in the vector-store `where` filter).

### 4.3 Two-model comparison (`outputs/compare.json`)

Retrieval and guardrails fixed; only the generator swapped. Cost is expressed as
compute (tokens/sec + VRAM), since both run locally at no monetary cost.

| Model | token-F1 | faithfulness | abstention acc | median latency | tokens/sec | peak VRAM |
|---|---|---|---|---|---|---|
| Qwen2.5-1.5B (fp16) | 0.42 | 1.00 | 0.94 | 1.91 s | 9.3 | 3.35 GB |
| Phi-3-mini (4-bit)  | 0.34 | 1.00 | 1.00 | 3.20 s | 6.4 | 2.94 GB |

Recommendation: **Qwen2.5-1.5B** leads on answer quality and latency for this
grounded-QA workload. **Phi-3-mini (4-bit)** uses less VRAM and abstains more
readily, so it suits tighter-memory or stricter-refusal deployments. The 4-bit
dequantization overhead makes Phi-3 slower here despite its smaller footprint.

## 5. Key design decisions

- **Numbered citations, not slug ids.** Small models reliably emit `[1]`/`[2]` but
  not `[expense-travel-policy-1]`; the agent maps numbers back to chunk ids. This
  raised citation recall from ~0 to reliable.
- **Hybrid faithfulness (NLI OR embedding).** The NLI model returns "neutral" for
  many valid paraphrases (e.g. answers adding a year not in the source sentence),
  which over-abstained on correct answers. Adding an embedding-cosine rescue lifted
  abstention accuracy from 0.69 to 0.94 while keeping faithfulness at 1.00. NLI
  still catches contradictions; cosine rescues restatements.
- **Sentence-level groundedness (SummaC-style).** NLI on a whole multi-sentence
  chunk returns "neutral" even for verbatim facts; splitting premises into
  sentences and taking the max fixed this (measured: 0.0003 -> 0.988 entailment on
  a known-grounded pair).
- **RAGAS deliberately excluded.** Its core metrics use an LLM judge defaulting to
  a paid OpenAI key; the same metric *ideas* are implemented locally with NLI +
  standard IR math, which is cheaper, deterministic, and offline.

## 6. Limitations (measured, not hand-waved)

- **Grounded-but-irrelevant confabulation.** When a user asks about a fact that
  exists only in a document above their clearance, the model can produce a wrong
  answer grounded in an off-topic accessible chunk (e.g. answering an EDD-threshold
  question with a lodging-cap figure). No confidential data leaks - the protected
  text never reaches the model - but the answer is wrong. A cosine
  retrieval-confidence gate was evaluated and **rejected on the data**: on-topic
  query-chunk similarity (min 0.68) overlaps with off-topic denied similarity
  (max 0.79), so no threshold separates them without dropping legitimate answers.
  This is a genuine open problem (answer relevance vs mere groundedness), left as
  documented future work rather than papered over with a fragile constant.
- **Small corpus saturates retrieval metrics.** Differences between chunking/search
  strategies are compressed; a larger corpus is needed to rank them meaningfully.
- **token-F1 undersells verbose answers.** A judge-free semantic answer-correctness
  metric would complement it.

## 7. Reproducing

```bash
python scripts/run_ablation.py      # -> outputs/ablation.json + figures/ablation.png
python scripts/run_eval.py          # -> outputs/eval.json
python scripts/run_llm_compare.py   # -> outputs/compare.json
python scripts/demo.py              # end-to-end CLI walkthrough
uvicorn api.main:app                # REST API (auth via RAG_API_KEY)
pytest -q                           # 48 unit tests
```

All stochastic operations are seeded (`SEED = 42`).
