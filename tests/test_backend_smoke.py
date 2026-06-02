"""End-to-end smoke tests for the FastAPI backend against the local Lance DB.

These run only when a real ``transcripts.lance`` with a ``chunks`` table is
present (it's gitignored / local-only), so CI without the dataset skips them
cleanly. They exercise the no-GPU surface of the API — FTS search, health,
documents, thumbnail, media Range streaming — and assert that the GPU-only
modes degrade to a structured 503 rather than a 500 when vLLM is down.

Run:  uv run --with pytest --with httpx pytest tests/test_backend_smoke.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

DB_PATH = Path(__file__).resolve().parent.parent / "transcripts.lance"

pytestmark = pytest.mark.skipif(
    not (DB_PATH / "chunks.lance").exists(),
    reason="local transcripts.lance dataset not present",
)


@pytest.fixture(scope="module")
def client():
    from backend import create_app
    from fastapi.testclient import TestClient

    return TestClient(create_app(DB_PATH))


def test_health_reports_db_facts(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert "chunks" in body["db"]["tables"]
    assert body["db"]["chunks"] > 0
    # The embed-server ping resolves to a bool either way, so the endpoint is
    # robust whether or not vLLM happens to be running on this host.
    assert isinstance(body["embed"]["ok"], bool)


def test_fts_search_returns_hits(client):
    r = client.get("/api/search", params={"q": "jag", "mode": "fts", "n": 5})
    assert r.status_code == 200
    hits = r.json()
    assert isinstance(hits, list)
    assert len(hits) <= 5
    if hits:
        h = hits[0]
        assert {"doc_id", "speech_id", "chunk_id", "text", "alignments"} <= h.keys()
        assert isinstance(h["alignments"], list)  # alignments_json parsed


def test_empty_query_returns_empty(client):
    r = client.get("/api/search", params={"q": "", "mode": "fts"})
    assert r.status_code == 200
    assert r.json() == []


def test_unknown_mode_is_rejected(client):
    # mode is a StrEnum query param → FastAPI validates it at the boundary (422).
    r = client.get("/api/search", params={"q": "jag", "mode": "bogus"})
    assert r.status_code == 422


def test_semantic_without_embeddings_or_vllm_is_4xx_not_500(client):
    # Two clean failure modes, never a 500:
    #   * text_embedding column not built yet → 400 (run embed-chunks)
    #   * column present but vLLM down        → 503 (embed server unreachable)
    r = client.get("/api/search", params={"q": "klimat", "mode": "semantic"})
    assert r.status_code in (400, 503)


def test_documents_pagination(client):
    r = client.get("/api/documents", params={"page": 1, "per_page": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["total"] > 0
    assert len(body["docs"]) <= 3


def test_thumbnail_and_media_roundtrip(client):
    docs = client.get("/api/documents", params={"page": 1, "per_page": 1}).json()["docs"]
    if not docs:
        pytest.skip("no documents in dataset")
    doc_id = docs[0]["doc_id"]

    # Thumbnail is optional (may be NULL) → accept 200 or 404, never 500.
    assert client.get(f"/api/thumbnail/{doc_id}").status_code in (200, 404)

    # A Range request must yield 206 with the right headers, or 404 if no media.
    r = client.get(f"/api/media/{doc_id}", headers={"Range": "bytes=0-1023"})
    assert r.status_code in (206, 404)
    if r.status_code == 206:
        assert r.headers["Accept-Ranges"] == "bytes"
        assert r.headers["Content-Range"].startswith("bytes 0-1023/")


def test_invalid_doc_id_is_400(client):
    assert client.get("/api/media/not-a-valid-id").status_code == 400
