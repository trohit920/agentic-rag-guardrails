"""FastAPI application exposing /ingest and /chat over the RAG agent.

Provides API-key authentication, per-key rate limiting, a request timeout, typed
error responses (auth, validation, timeout), and structured JSON request logging.
The heavy models are loaded once at startup via the lifespan handler; business
logic is delegated to the src/ library. Run with:
    uvicorn api.main:app
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator

from fastapi import Depends, FastAPI, HTTPException, status

from src.config import REQUEST_TIMEOUT_SECONDS, Sensitivity
from src.factory import build_agent
from src.indexing import build_vector_store

from .schemas import ChatRequest, ChatResponse, IngestResponse
from .security import require_api_key

logging.basicConfig(level=logging.INFO, format="%(message)s")
_logger = logging.getLogger("rag.api")


def _log(event: str, **fields: object) -> None:
    """Emit a structured JSON log line for monitoring."""
    _logger.info(json.dumps({"event": event, **fields}))


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load models and build the index once at startup."""
    _log("startup_begin")
    agent, store, embedder = build_agent()
    app.state.agent = agent
    app.state.store = store
    app.state.embedder = embedder
    _log("startup_ready", chunks=store.count())
    yield
    _log("shutdown")


app = FastAPI(title="agentic-rag-guardrails", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe (no auth)."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, _: str = Depends(require_api_key)) -> ChatResponse:
    """Answer a query through the guarded agent, with a hard request timeout."""
    history = [{"role": message.role, "content": message.content} for message in request.history]
    clearance = Sensitivity(request.clearance)
    try:
        answer = await asyncio.wait_for(
            asyncio.to_thread(app.state.agent.answer, request.query, clearance, history),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        _log("chat_timeout", clearance=request.clearance, query_len=len(request.query))
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail="generation timed out"
        ) from None
    _log(
        "chat",
        clearance=request.clearance,
        query_len=len(request.query),
        abstained=answer.abstained,
        citations=len(answer.citations),
        latency_s=round(answer.latency_seconds, 3),
    )
    return ChatResponse(
        answer=answer.answer,
        citations=answer.citations,
        abstained=answer.abstained,
        latency_seconds=round(answer.latency_seconds, 3),
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest(_: str = Depends(require_api_key)) -> IngestResponse:
    """Rebuild the vector index from the corpus on disk."""
    count = build_vector_store(app.state.store, app.state.embedder)
    _log("ingest", chunks_indexed=count)
    return IngestResponse(chunks_indexed=count)
