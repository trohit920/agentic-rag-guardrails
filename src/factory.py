"""Agent assembly factory.

Single place that wires the full pipeline (embedder, vector store, retriever,
guardrails, local LLM, LangGraph agent) with project defaults. Reused by the API
layer and the CLI demo so construction logic is not duplicated. Builds the index
on first use if the store is empty.
"""

from __future__ import annotations

from pathlib import Path

from .agent import RagAgent
from .config import CHROMA_DIR, DEFAULT_SEARCH_STRATEGY
from .embeddings import Embedder
from .guardrails import GroundednessGuard, InjectionGuard
from .indexing import build_vector_store
from .llm import LocalLLM
from .retriever import Retriever
from .vector_store import VectorStore


def build_agent(
    persist_dir: Path = CHROMA_DIR, load_in_4bit: bool = False
) -> tuple[RagAgent, VectorStore, Embedder]:
    """Construct the default agent; returns it with the store and embedder."""
    embedder = Embedder()
    store = VectorStore(persist_dir=persist_dir)
    if store.count() == 0:
        build_vector_store(store, embedder)
    retriever = Retriever(store, embedder, strategy=DEFAULT_SEARCH_STRATEGY)
    injection_guard = InjectionGuard(embedder)
    groundedness_guard = GroundednessGuard(embedder=embedder)
    llm = LocalLLM(load_in_4bit=load_in_4bit)
    agent = RagAgent(retriever, llm, injection_guard, groundedness_guard)
    return agent, store, embedder
