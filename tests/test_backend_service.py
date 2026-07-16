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
from backend.search.postprocess import _rrf_fuse
from backend.search.service import (
    _frame_search,
    _postprocess_hits,
    _vector_search,
    run_search,
)
from backend.search.spec import SearchMode, SearchSpec
from fastapi import HTTPException

from fakes import (
    TOPICS,
    FakeCaptionClient,
    FakeEmbedClient,
    make_doc,
    write_frames_aligned_to_chunks,
)
from rmedia.clients.embedding import VLLMEmbeddingClient
from rmedia.core.engine import ensure_fts_index
from rmedia.features.columns import caption_column, embed_caption_column
from rmedia.ingest.ingest import ingest_many


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


def _no_embedder() -> Never:
    raise AssertionError("embedder must not be constructed on keyword-only paths")


class TestRunSearchErrors:
    def test_semantic_without_embedding_column_is_400(self, chunks) -> None:
        spec = SearchSpec(q="carbon", mode=SearchMode.SEMANTIC)
        with pytest.raises(HTTPException) as exc:
            run_search(chunks, None, get_embedder=_embedder, get_reranker=_no_reranker, spec=spec, image_bytes=None)
        assert exc.value.status_code == 400
        assert "embed-chunks" in exc.value.detail

    def test_hybrid_without_text_query_is_400(self, chunks) -> None:
        # Image-only hybrid: no text vector, so FTS half can't run.
        spec = SearchSpec(q="", mode=SearchMode.HYBRID)
        with pytest.raises(HTTPException) as exc:
            run_search(chunks, None, get_embedder=_embedder, get_reranker=_no_reranker, spec=spec, image_bytes=b"jpeg")
        assert exc.value.status_code == 400
        assert "text query" in exc.value.detail

    def test_hybrid_without_embedding_column_is_400(self, chunks) -> None:
        spec = SearchSpec(q="carbon", mode=SearchMode.HYBRID)
        with pytest.raises(HTTPException) as exc:
            run_search(chunks, None, get_embedder=_embedder, get_reranker=_no_reranker, spec=spec, image_bytes=None)
        assert exc.value.status_code == 400


class TestRunSearchDegradation:
    def test_fts_returns_parsed_alignments_key(self, chunks) -> None:
        spec = SearchSpec(q="carbon", mode=SearchMode.FTS)
        hits = run_search(chunks, None, get_embedder=_embedder, get_reranker=_no_reranker, spec=spec, image_bytes=None)
        assert hits
        assert "alignments" in hits[0]
        assert "alignments_json" not in hits[0]

    def test_visual_without_frames_returns_empty(self, chunks) -> None:
        # No chunk_frames table → frame search degrades to [] instead of erroring.
        spec = SearchSpec(q="anything", mode=SearchMode.VISUAL)
        hits = run_search(chunks, None, get_embedder=_embedder, get_reranker=_no_reranker, spec=spec, image_bytes=None)
        assert hits == []

    def test_all_without_embeddings_falls_back_to_fts(self, chunks) -> None:
        # 'all' fuses FTS + (absent) vectors; the FTS ranking still comes through.
        spec = SearchSpec(q="carbon emissions", mode=SearchMode.ALL)
        hits = run_search(chunks, None, get_embedder=_embedder, get_reranker=_no_reranker, spec=spec, image_bytes=None)
        assert hits


class TestSceneSearch:
    @pytest.fixture
    def captioned(self, tmp_path):
        """chunks + frames with caption + caption_embedding built (offline fakes)."""
        db = tmp_path / "scene.lance"
        ingest_many(db, [make_doc(f"input/{n}.mp4", [t]) for n, t in TOPICS.items()])
        write_frames_aligned_to_chunks(db)
        frames_path = db / "chunk_frames.lance"
        caption_column(frames_path, client=FakeCaptionClient())
        embed_caption_column(frames_path, client=FakeEmbedClient())
        conn = lancedb.connect(str(db))
        frames = conn.open_table("chunk_frames")
        ensure_fts_index(frames, "caption", language="English")  # captions are keyword-searchable
        return conn.open_table("chunks"), frames

    def test_scene_returns_chunk_hits(self, captioned) -> None:
        chunks, frames = captioned
        spec = SearchSpec(q="anything", mode=SearchMode.SCENE)
        hits = run_search(chunks, frames, get_embedder=_embedder, get_reranker=_no_reranker, spec=spec, image_bytes=None)
        # caption_embedding is populated → scene search ranks frames, joins to chunks.
        assert hits
        assert {"doc_id", "text", "caption"} <= hits[0].keys()

    def test_scene_hit_carries_the_frame_distance(self, captioned) -> None:
        # Regression: scene hits used to arrive scoreless — the frame→chunk join
        # dropped the ranking signal, so the results table had nothing to sort on.
        # The best frame's cosine `_distance` must ride onto the joined chunk row.
        chunks, frames = captioned
        spec = SearchSpec(q="anything", mode=SearchMode.SCENE)
        hits = run_search(chunks, frames, get_embedder=_embedder, get_reranker=_no_reranker, spec=spec, image_bytes=None)
        assert hits and "_distance" in hits[0]

    def test_scene_fts_hit_carries_the_bm25_score(self, captioned) -> None:
        # Same regression on the keyword side: the caption BM25 `_score` must
        # survive the join back to chunks.
        chunks, frames = captioned
        spec = SearchSpec(q="caption", mode=SearchMode.SCENE_FTS)
        hits = run_search(chunks, frames, get_embedder=_no_embedder, get_reranker=_no_reranker, spec=spec, image_bytes=None)
        assert hits and "_score" in hits[0]

    def test_scene_without_frames_is_empty(self, chunks) -> None:
        # No chunk_frames table → scene degrades to [] like visual (not a 500).
        spec = SearchSpec(q="carbon", mode=SearchMode.SCENE)
        assert run_search(chunks, None, get_embedder=_embedder, get_reranker=_no_reranker, spec=spec, image_bytes=None) == []

    def test_caption_rides_along_on_fts_hits(self, captioned) -> None:
        # Captions surface on EVERY mode (for the list/table views), not just scene.
        chunks, frames = captioned
        spec = SearchSpec(q="carbon", mode=SearchMode.FTS)
        hits = run_search(chunks, frames, get_embedder=_embedder, get_reranker=_no_reranker, spec=spec, image_bytes=None)
        assert hits and hits[0]["caption"]

    def test_scene_fts_keyword_search(self, captioned) -> None:
        # BM25 over the caption text. The fake captions all contain "caption", so a
        # keyword query returns chunk hits — and needs NO embedder (note _no_reranker
        # only; the embed getter is never called on this path).
        chunks, frames = captioned
        spec = SearchSpec(q="caption", mode=SearchMode.SCENE_FTS)
        hits = run_search(chunks, frames, get_embedder=_no_embedder, get_reranker=_no_reranker, spec=spec, image_bytes=None)
        assert hits
        assert {"doc_id", "text", "caption"} <= hits[0].keys()

    def test_scene_fts_without_captions_is_empty(self, chunks) -> None:
        spec = SearchSpec(q="anything", mode=SearchMode.SCENE_FTS)
        assert run_search(chunks, None, get_embedder=_no_embedder, get_reranker=_no_reranker, spec=spec, image_bytes=None) == []


class TestSearchHelpers:
    def test_vector_search_missing_column_returns_empty(self, chunks) -> None:
        vec = FakeEmbedClient().embed_text(["x"])[0]
        assert _vector_search(chunks, vec, "text_embedding", n=5, where=None) == []

    def test_vector_search_none_vec_returns_empty(self, chunks) -> None:
        assert _vector_search(chunks, None, "text_embedding", n=5, where=None) == []

    def test_frame_search_none_table_returns_empty(self, chunks) -> None:
        vec = FakeEmbedClient().embed_text(["x"])[0]
        assert _frame_search(None, chunks, vec, n=5, where=None) == []

    def test_postprocess_parses_and_pops_raw(self) -> None:
        out = _postprocess_hits([{"text": "hi", "alignments_json": '[{"words": []}]'}])
        assert out[0]["alignments"] == [{"words": []}]
        assert "alignments_json" not in out[0]

    def test_postprocess_missing_json_becomes_empty_list(self) -> None:
        out = _postprocess_hits([{"text": "hi"}])
        assert out[0]["alignments"] == []


class TestRrfFuse:
    """Reciprocal-rank fusion for the multi-column (text+frame) hybrid path."""

    def test_doc_in_both_rankings_outranks_doc_in_one(self) -> None:
        a = {"doc_id": "A", "chunk_id": 0}
        b = {"doc_id": "B", "chunk_id": 0}
        c = {"doc_id": "C", "chunk_id": 0}
        # A is top of both lists → its reciprocal ranks sum → it must win.
        fused = _rrf_fuse([[a, b], [a, c]])
        assert fused[0]["doc_id"] == "A"

    def test_dedups_by_key_keeping_first_occurrence(self) -> None:
        first = {"doc_id": "A", "chunk_id": 0, "tag": "first"}
        second = {"doc_id": "A", "chunk_id": 0, "tag": "second"}
        fused = _rrf_fuse([[first], [second]])
        assert len(fused) == 1
        assert fused[0]["tag"] == "first"  # the highest-ranked occurrence is canonical

    def test_empty_rankings_is_empty(self) -> None:
        assert _rrf_fuse([]) == []
