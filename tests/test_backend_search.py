"""No-GPU backend search tests — exercise the real ``LanceTable.search()`` chain.

The dataset-gated smoke tests in ``test_backend_smoke.py`` can't cover the vector
modes: with no vLLM they 503 *before* the query builder runs. These tests inject a
deterministic offline embedder, so semantic / visual / hybrid / all actually
execute the query builder — the path where the sync-API ``.search()`` (vs the
async-only ``.query()``) matters.

Run:  uv run pytest tests/test_backend_search.py -v
"""

from __future__ import annotations

import pytest

from fakes import TOPICS, FakeEmbedClient, FakeReranker, make_doc, write_frames_aligned_to_chunks
from raudio.features.columns import embed_frame_column, embed_text_column
from raudio.ingest.ingest import ingest_many


@pytest.fixture
def client(tmp_path, monkeypatch):
    db = tmp_path / "transcripts.lance"
    ingest_many(db, [make_doc(f"input/{name}.mp4", [text]) for name, text in TOPICS.items()])
    embed_text_column(db / "chunks.lance", client=FakeEmbedClient())
    write_frames_aligned_to_chunks(db)
    embed_frame_column(db / "chunk_frames.lance", client=FakeEmbedClient())

    # The backend lazily constructs VLLMEmbeddingClient() with no args — swap in
    # the offline fake so the search query builder actually runs.
    import raudio.vllm.embedding as embeddings

    monkeypatch.setattr(embeddings, "VLLMEmbeddingClient", FakeEmbedClient)

    from backend import create_app
    from fastapi.testclient import TestClient

    return TestClient(create_app(db))


def _hits(client, **params) -> list[dict]:
    r = client.get("/api/search", params=params)
    assert r.status_code == 200, (params, r.status_code, r.text)
    body = r.json()
    assert isinstance(body, list)
    return body


def test_fts_runs(client):
    hits = _hits(client, q="carbon", mode="fts", n=5)
    assert hits and "carbon emissions" in hits[0]["text"]


def test_semantic_ranks_planted_nearest(client):
    # An exact-text query embeds to the same vector as its chunk → distance 0.
    hits = _hits(client, q=TOPICS["climate"], mode="semantic", n=3)
    assert hits and hits[0]["text"] == TOPICS["climate"]
    assert {"doc_id", "speech_id", "chunk_id", "text", "alignments"} <= hits[0].keys()


def test_hybrid_runs(client):
    hits = _hits(client, q=TOPICS["sports"], mode="hybrid", n=3)
    assert hits  # FTS + vector fused via RRF (no rerank server needed)


def test_all_runs(client):
    hits = _hits(client, q=TOPICS["economy"], mode="all", n=3)
    assert hits


def test_visual_runs(client):
    # Text-query → frame-vector search → join back to chunks. Ranking is arbitrary
    # under the fake embedder; the point is the query builder + join execute.
    hits = _hits(client, q="anything", mode="visual", n=3)
    assert hits
    assert "text" in hits[0]


def test_post_hybrid_runs(client):
    # POST form path (the multipart/image branch), default mode is hybrid.
    r = client.post("/api/search", data={"q": TOPICS["sports"], "mode": "hybrid", "n": 3})
    assert r.status_code == 200, r.text
    assert isinstance(r.json(), list)


def test_post_empty_query_returns_empty(client):
    r = client.post("/api/search", data={"q": "", "mode": "hybrid"})
    assert r.status_code == 200
    assert r.json() == []


def test_all_mode_with_rerank_runs(client, monkeypatch):
    # rerank=true on 'all' fuses, then cross-encoder re-orders the top-K.
    import raudio.vllm.reranker as reranker

    monkeypatch.setattr(reranker, "VLLMReranker", FakeReranker)
    hits = _hits(client, q=TOPICS["economy"], mode="all", rerank="true", n=3)
    assert hits
