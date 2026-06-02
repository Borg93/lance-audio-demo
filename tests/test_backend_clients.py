"""Lazy vLLM client accessors + the DI seam that binds them to a request.

No GPU/network: the deferred ``raudio.vllm.*`` imports inside the accessors
are the documented monkeypatch seam, so we swap the client classes for fakes (or
a raiser) and assert the cache-then-construct behaviour and the 503 mapping.
"""

from __future__ import annotations

from pathlib import Path
from typing import Never
from unittest.mock import Mock

import pytest
from backend import clients, deps
from backend.state import AppState
from fastapi import HTTPException

import raudio.vllm.embedding as embeddings_mod
import raudio.vllm.reranker as reranker_mod


def _state() -> AppState:
    return AppState(db_path=Path("x"), names=["chunks"], chunks=object())


def _boom() -> Never:
    raise RuntimeError("server down")


class TestEnsureEmbedder:
    def test_returns_cached_without_constructing(self, monkeypatch) -> None:
        state = _state()
        sentinel = object()
        state.embedder = sentinel
        # If it tried to construct, _boom would fire — proving the cache short-circuit.
        monkeypatch.setattr(embeddings_mod, "VLLMEmbeddingClient", _boom)
        assert clients.ensure_embedder(state) is sentinel

    def test_constructs_and_caches_on_miss(self, monkeypatch) -> None:
        state = _state()
        made = object()
        monkeypatch.setattr(embeddings_mod, "VLLMEmbeddingClient", lambda: made)
        assert clients.ensure_embedder(state) is made
        assert state.embedder is made  # written back for reuse

    def test_construction_failure_maps_to_503(self, monkeypatch) -> None:
        state = _state()
        monkeypatch.setattr(embeddings_mod, "VLLMEmbeddingClient", _boom)
        with pytest.raises(HTTPException) as exc:
            clients.ensure_embedder(state)
        assert exc.value.status_code == 503


class TestEnsureReranker:
    def test_returns_cached_without_constructing(self, monkeypatch) -> None:
        state = _state()
        sentinel = object()
        state.reranker = sentinel
        monkeypatch.setattr(reranker_mod, "VLLMReranker", _boom)
        assert clients.ensure_reranker(state) is sentinel

    def test_constructs_and_caches_on_miss(self, monkeypatch) -> None:
        state = _state()
        made = object()
        monkeypatch.setattr(reranker_mod, "VLLMReranker", lambda: made)
        assert clients.ensure_reranker(state) is made
        assert state.reranker is made

    def test_construction_failure_maps_to_503(self, monkeypatch) -> None:
        state = _state()
        monkeypatch.setattr(reranker_mod, "VLLMReranker", _boom)
        with pytest.raises(HTTPException) as exc:
            clients.ensure_reranker(state)
        assert exc.value.status_code == 503


class TestDeps:
    def test_get_embedder_returns_getter_bound_to_state(self, monkeypatch) -> None:
        state = _state()
        seen: list[AppState] = []
        monkeypatch.setattr(deps, "ensure_embedder", lambda s: seen.append(s) or "embedder")
        getter = deps.get_embedder(state)
        assert getter() == "embedder"
        assert seen == [state]

    def test_get_reranker_returns_getter_bound_to_state(self, monkeypatch) -> None:
        state = _state()
        seen: list[AppState] = []
        monkeypatch.setattr(deps, "ensure_reranker", lambda s: seen.append(s) or "reranker")
        getter = deps.get_reranker(state)
        assert getter() == "reranker"
        assert seen == [state]

    def test_get_state_reads_app_state_resources(self) -> None:
        state = _state()
        request = Mock()
        request.app.state.resources = state
        assert deps.get_state(request) is state
