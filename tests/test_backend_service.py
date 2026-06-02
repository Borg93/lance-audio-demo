"""run_search routing — the error and graceful-degradation branches that the
happy-path tests in ``test_backend_search.py`` don't reach.

Uses a real chunks table with NO embedding columns (``embed-chunks`` not run), so
the "embeddings not built" 400s and the "no frames → empty" degradations execute
against genuine Lance handles. The reranker getter raises if touched — none of
these paths should construct it.
"""

from __future__ import annotations

from typing import Never, cast

import lancedb
import pytest
from backend.search.service import (
    _frame_search,
    _postprocess_hits,
    _vector_search,
    run_search,
)
from backend.search.spec import SearchMode, SearchSpec
from fastapi import HTTPException

from fakes import TOPICS, FakeEmbedClient, make_doc
from raudio.ingest.ingest import ingest_many
from raudio.vllm.embedding import VLLMEmbeddingClient


@pytest.fixture
def chunks(tmp_path):
    """A real chunks table with no text/frame embedding columns."""
    db = tmp_path / "transcripts.lance"
    ingest_many(db, [make_doc(f"input/{name}.mp4", [text]) for name, text in TOPICS.items()])
    return lancedb.connect(str(db)).open_table("chunks")


def _embedder() -> VLLMEmbeddingClient:
    # The offline fake satisfies the embed_text/embed_image surface run_search uses.
    return cast(VLLMEmbeddingClient, FakeEmbedClient())


def _no_reranker() -> Never:
    raise AssertionError("reranker must not be constructed on these paths")


class TestRunSearchErrors:
    def test_semantic_without_embedding_column_is_400(self, chunks) -> None:
        spec = SearchSpec(q="carbon", mode=SearchMode.SEMANTIC)
        with pytest.raises(HTTPException) as exc:
            run_search(chunks, None, _embedder, _no_reranker, spec, image_bytes=None)
        assert exc.value.status_code == 400
        assert "embed-chunks" in exc.value.detail

    def test_hybrid_without_text_query_is_400(self, chunks) -> None:
        # Image-only hybrid: no text vector, so FTS half can't run.
        spec = SearchSpec(q="", mode=SearchMode.HYBRID)
        with pytest.raises(HTTPException) as exc:
            run_search(chunks, None, _embedder, _no_reranker, spec, image_bytes=b"jpeg")
        assert exc.value.status_code == 400
        assert "text query" in exc.value.detail

    def test_hybrid_without_embedding_column_is_400(self, chunks) -> None:
        spec = SearchSpec(q="carbon", mode=SearchMode.HYBRID)
        with pytest.raises(HTTPException) as exc:
            run_search(chunks, None, _embedder, _no_reranker, spec, image_bytes=None)
        assert exc.value.status_code == 400


class TestRunSearchDegradation:
    def test_fts_returns_parsed_alignments_key(self, chunks) -> None:
        spec = SearchSpec(q="carbon", mode=SearchMode.FTS)
        hits = run_search(chunks, None, _embedder, _no_reranker, spec, image_bytes=None)
        assert hits
        assert "alignments" in hits[0]
        assert "alignments_json" not in hits[0]

    def test_visual_without_frames_returns_empty(self, chunks) -> None:
        # No chunk_frames table → frame search degrades to [] instead of erroring.
        spec = SearchSpec(q="anything", mode=SearchMode.VISUAL)
        hits = run_search(chunks, None, _embedder, _no_reranker, spec, image_bytes=None)
        assert hits == []

    def test_all_without_embeddings_falls_back_to_fts(self, chunks) -> None:
        # 'all' fuses FTS + (absent) vectors; the FTS ranking still comes through.
        spec = SearchSpec(q="carbon emissions", mode=SearchMode.ALL)
        hits = run_search(chunks, None, _embedder, _no_reranker, spec, image_bytes=None)
        assert hits


class TestSearchHelpers:
    def test_vector_search_missing_column_returns_empty(self, chunks) -> None:
        vec = FakeEmbedClient().embed_text(["x"])[0]
        assert _vector_search(chunks, vec, "text_embedding", 5, None) == []

    def test_vector_search_none_vec_returns_empty(self, chunks) -> None:
        assert _vector_search(chunks, None, "text_embedding", 5, None) == []

    def test_frame_search_none_table_returns_empty(self, chunks) -> None:
        vec = FakeEmbedClient().embed_text(["x"])[0]
        assert _frame_search(None, chunks, vec, 5, None) == []

    def test_postprocess_parses_and_pops_raw(self) -> None:
        out = _postprocess_hits([{"text": "hi", "alignments_json": '[{"words": []}]'}])
        assert out[0]["alignments"] == [{"words": []}]
        assert "alignments_json" not in out[0]

    def test_postprocess_missing_json_becomes_empty_list(self) -> None:
        out = _postprocess_hits([{"text": "hi"}])
        assert out[0]["alignments"] == []
