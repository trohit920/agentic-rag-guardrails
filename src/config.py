"""Central configuration and named constants for the agentic-rag-guardrails POC.

Keeps every tunable (model ids, chunking parameters, retrieval depth, guardrail
thresholds, paths) in one place so runs are reproducible and free of magic
numbers. Import these constants rather than hard-coding values elsewhere. All
models are open-source and run locally; no external or paid APIs are used.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED: int = 42

# --------------------------------------------------------------------------- #
# Models (all open-source, run locally, non-gated)
# --------------------------------------------------------------------------- #
# Primary generator (fp16, ~3 GB VRAM). Proven small instruct model.
GENERATION_MODEL_A: str = "Qwen/Qwen2.5-1.5B-Instruct"
# Comparison generator (loaded in 4-bit NF4, ~2.3 GB VRAM). MIT-licensed.
GENERATION_MODEL_B: str = "microsoft/Phi-3-mini-4k-instruct"
# Sentence embedding model for retrieval (local, ~0.2 GB).
EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
# NLI model for groundedness / faithfulness scoring (CPU-friendly).
NLI_MODEL: str = "cross-encoder/nli-deberta-v3-small"

# bge models expect a query prefix for asymmetric retrieval; documents are raw.
BGE_QUERY_PREFIX: str = "Represent this sentence for searching relevant passages: "

# --------------------------------------------------------------------------- #
# Generation / decoding
# --------------------------------------------------------------------------- #
MAX_NEW_TOKENS: int = 384
GEN_TEMPERATURE: float = 0.0  # deterministic answers by default
SAMPLE_TEMPERATURE: float = 1.0  # for semantic-entropy abstention sampling
SAMPLE_TOP_P: float = 0.95
NUM_ABSTENTION_SAMPLES: int = 5  # stochastic samples for semantic entropy

# Rolling-summary + context assembly token budget (leaves room for the answer).
MAX_CONTEXT_TOKENS: int = 3072
MAX_HISTORY_TOKENS: int = 1024

# --------------------------------------------------------------------------- #
# Chunking (the ablation knobs; defaults are the recommended setting)
# --------------------------------------------------------------------------- #
class ChunkStrategy(str, Enum):
    """Supported document chunking strategies."""

    FIXED = "fixed"            # fixed character window
    RECURSIVE = "recursive"    # split on paragraph/sentence boundaries first
    SENTENCE = "sentence"      # one chunk per sentence group


DEFAULT_CHUNK_STRATEGY: ChunkStrategy = ChunkStrategy.RECURSIVE
CHUNK_SIZE_CHARS: int = 800
CHUNK_OVERLAP_CHARS: int = 120

# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #
class SearchStrategy(str, Enum):
    """Supported retrieval strategies."""

    DENSE = "dense"            # embedding similarity only
    HYBRID = "hybrid"          # dense + BM25 lexical fusion
    METADATA = "metadata"      # dense + metadata pre-filter


DEFAULT_SEARCH_STRATEGY: SearchStrategy = SearchStrategy.HYBRID
TOP_K: int = 5
HYBRID_DENSE_WEIGHT: float = 0.6  # weight on dense score in hybrid fusion (0..1)

# --------------------------------------------------------------------------- #
# Guardrail thresholds
# --------------------------------------------------------------------------- #
# Minimum NLI entailment probability for an answer sentence to count as grounded.
GROUNDEDNESS_ENTAILMENT_MIN: float = 0.50
# Fraction of answer sentences that must be grounded to accept the answer.
GROUNDEDNESS_ACCEPT_RATIO: float = 0.60
# Embedding cosine above this also counts a sentence as grounded (paraphrase
# rescue for the NLI model, which returns neutral for many valid restatements).
GROUNDEDNESS_SIM_MIN: float = 0.75
# Semantic-entropy above this (nats) triggers abstention (confident-but-wrong).
ABSTENTION_ENTROPY_MAX: float = 1.10
# Prompt-injection similarity threshold vs known-attack corpus.
INJECTION_SIMILARITY_MAX: float = 0.75

ABSTAIN_MESSAGE: str = (
    "I don't have enough grounded information in the provided documents to answer "
    "that reliably."
)

# --------------------------------------------------------------------------- #
# Access control
# --------------------------------------------------------------------------- #
class Sensitivity(str, Enum):
    """Document sensitivity levels, ordered from least to most restricted."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"


# Clearance ordering for ACL checks; a user may read at or below their level.
SENSITIVITY_ORDER: tuple[Sensitivity, ...] = (
    Sensitivity.PUBLIC,
    Sensitivity.INTERNAL,
    Sensitivity.CONFIDENTIAL,
)


def allowed_sensitivities(clearance: Sensitivity) -> tuple[Sensitivity, ...]:
    """Return the sensitivity levels a user with the given clearance may read."""
    return SENSITIVITY_ORDER[: SENSITIVITY_ORDER.index(clearance) + 1]

# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
API_KEY_ENV_VAR: str = "RAG_API_KEY"
API_KEY_HEADER: str = "x-api-key"
REQUEST_TIMEOUT_SECONDS: float = 60.0
RATE_LIMIT_PER_MINUTE: int = 30

# --------------------------------------------------------------------------- #
# Numerical
# --------------------------------------------------------------------------- #
EPS: float = 1e-12

# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
EVAL_K_VALUES: tuple[int, ...] = (1, 3, 5)  # for hit@k / nDCG@k

# --------------------------------------------------------------------------- #
# Filesystem layout
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
EVAL_DIR: Path = PROJECT_ROOT / "eval"
OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"
FIGURE_DIR: Path = OUTPUT_DIR / "figures"
CHROMA_DIR: Path = PROJECT_ROOT / "outputs" / "chroma"
