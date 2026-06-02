"""Lazy vLLM client accessors bound to :class:`~backend.state.AppState`.

Each accessor returns the cached client if present, else constructs it on first
use and writes it back onto the same ``AppState`` (so subsequent requests reuse
it), mapping any failure to a structured 503. The ``raudio.vllm.*`` imports
are deferred *inside* each function: it keeps FTS-only startup free of the
multimodal deps, and it's the seam tests monkeypatch (the symbol must resolve at
call time, not import time).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import HTTPException

if TYPE_CHECKING:
    from backend.state import AppState
    from raudio.vllm.embedding import VLLMEmbeddingClient
    from raudio.vllm.reranker import VLLMReranker

logger = logging.getLogger(__name__)


def ensure_embedder(state: AppState) -> VLLMEmbeddingClient:
    """Return the app's embedding client, connecting on first use (503 on failure)."""
    if state.embedder is not None:
        return state.embedder
    try:
        from raudio.vllm.embedding import VLLMEmbeddingClient

        state.embedder = VLLMEmbeddingClient()
        return state.embedder
    except Exception as e:
        logger.exception("failed to initialize embedding client")
        raise HTTPException(
            status_code=503,
            detail=f"embedding service unavailable: {e}",
        ) from e


def ensure_reranker(state: AppState) -> VLLMReranker:
    """Return the app's reranker client, connecting on first use (503 on failure)."""
    if state.reranker is not None:
        return state.reranker
    try:
        from raudio.vllm.reranker import VLLMReranker

        state.reranker = VLLMReranker()
        return state.reranker
    except Exception as e:
        logger.exception("failed to initialize reranker client")
        raise HTTPException(
            status_code=503,
            detail=f"rerank service unavailable: {e}",
        ) from e
