"""Lazy vLLM client accessors bound to :class:`~backend.state.AppState`.

Each accessor returns the cached client if present, else constructs it on first
use and writes it back onto the same ``AppState`` (so subsequent requests reuse
it), mapping any failure to a ``ServiceUnavailableError`` (HTTP 503). The vLLM
URLs come from ``state.settings`` (env ``RAUDIO_EMBED_URL``/``RAUDIO_RERANK_URL``,
defaulting to the local servers). The ``raudio.vllm.*`` imports are deferred
*inside* each function: it keeps FTS-only startup free of the multimodal deps,
and it's the seam tests monkeypatch (the symbol must resolve at call time, not
import time).
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

from backend.core.exceptions import ServiceUnavailableError

if TYPE_CHECKING:
    from backend.state import AppState
    from raudio.vllm.embedding import VLLMEmbeddingClient
    from raudio.vllm.reranker import VLLMReranker

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


def _construct(factory: Callable[..., _T], url_kwarg: str, url: str) -> _T:
    """Build a vLLM client, injecting ``url_kwarg=url`` only if the factory takes it.

    The real ``raudio.vllm`` clients accept ``embed_url=``/``rerank_url=``, so the
    configured URL (from ``Settings``) is injected. Test fakes monkeypatched over
    the client class have no such parameter; we detect that via the signature and
    fall back to a no-arg construction so the injection never breaks them.
    """
    try:
        takes_url = url_kwarg in inspect.signature(factory).parameters
    except (TypeError, ValueError):
        takes_url = False
    return factory(**{url_kwarg: url}) if takes_url else factory()


def ensure_embedder(state: AppState) -> VLLMEmbeddingClient:
    """Return the app's embedding client, connecting on first use (503 on failure)."""
    if state.embedder is not None:
        return state.embedder
    try:
        from raudio.vllm.embedding import VLLMEmbeddingClient

        state.embedder = _construct(VLLMEmbeddingClient, "embed_url", state.settings.embed_url)
        return state.embedder
    except Exception as e:
        logger.exception("failed to initialize embedding client")
        raise ServiceUnavailableError("embedding service unavailable") from e


def ensure_reranker(state: AppState) -> VLLMReranker:
    """Return the app's reranker client, connecting on first use (503 on failure)."""
    if state.reranker is not None:
        return state.reranker
    try:
        from raudio.vllm.reranker import VLLMReranker

        state.reranker = _construct(VLLMReranker, "rerank_url", state.settings.rerank_url)
        return state.reranker
    except Exception as e:
        logger.exception("failed to initialize reranker client")
        raise ServiceUnavailableError("rerank service unavailable") from e
