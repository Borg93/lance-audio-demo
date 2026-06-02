"""FastAPI dependency wrappers — the seam between the app and the routers.

Routers depend on these instead of capturing closures or touching ``app.state``.
``StateDep`` hands a router the per-app :class:`AppState`; the embedder/reranker
deps hand back *zero-arg getters* (bound to that state) so the search service can
keep its ``Callable[[], Client]`` signature and connect lazily on first use.
Tests can override any of these via ``app.dependency_overrides``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Annotated

from fastapi import Depends, Request

from backend.clients import ensure_embedder, ensure_reranker
from backend.state import AppState

if TYPE_CHECKING:
    from raudio.vllm.embedding import VLLMEmbeddingClient
    from raudio.vllm.reranker import VLLMReranker


def get_state(request: Request) -> AppState:
    """The resources opened in the app factory, stashed on ``app.state``."""
    return request.app.state.resources


StateDep = Annotated[AppState, Depends(get_state)]


def get_embedder(state: StateDep) -> Callable[[], VLLMEmbeddingClient]:
    """A zero-arg getter that lazily connects (then caches) the embedding client."""
    return lambda: ensure_embedder(state)


def get_reranker(state: StateDep) -> Callable[[], VLLMReranker]:
    """A zero-arg getter that lazily connects (then caches) the reranker client."""
    return lambda: ensure_reranker(state)


EmbedderFactoryDep = Annotated[Callable[[], "VLLMEmbeddingClient"], Depends(get_embedder)]
RerankerFactoryDep = Annotated[Callable[[], "VLLMReranker"], Depends(get_reranker)]
