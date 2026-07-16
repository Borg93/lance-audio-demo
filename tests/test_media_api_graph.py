"""Knowledge-graph endpoints (media_api port) over a synthetic ``kg_*`` graph.

Port of ``tests/test_backend_graph.py`` to the descriptor-driven
``backend.media_api.graph`` router. Two layers:

- ``test_enforce_limit_*`` unit-test the pure ``_enforce_limit`` helper — the
  regression guard for the LIMIT-cap bypass.
- The endpoint tests cover the contract the ``/graph`` page depends on:
  ``built: false`` when the capability is undeclared or the tables are absent,
  the cap enforced end-to-end through ``lance_graph``, single-statement-only,
  the ``entity_id`` whitelist that guards Cypher interpolation, the
  reverse-edge dedupe — and the descriptor-fed ``graph_presets`` clip title
  column (with its doc-id fallback when unset).
"""

from __future__ import annotations

import json
from pathlib import Path

import lance
import pyarrow as pa
import pytest
from backend.core.config import Settings
from backend.core.handlers import register_handlers
from backend.media_api import graph
from backend.media_api.graph import _enforce_limit
from backend.state import AppState
from fastapi import FastAPI
from fastapi.testclient import TestClient

# A 4-entity star: Sverige is mentioned in every chunk, so it co-occurs with
# everyone — enough structure to exercise MATCH / MENTIONS / RELATIONSHIP.
_ENTITIES = [
    ("1111111111111111", "Sverige", "GEO", 3),
    ("2222222222222222", "Regeringen", "ORG", 2),
    ("3333333333333333", "Anna Lindberg", "PERSON", 1),
    ("4444444444444444", "Stockholm", "GEO", 1),
]
_CHUNKS = [
    ("doc0:0:0", "doc0", "Video A", 0.0, 30.0, "Sverige och regeringen ..."),
    ("doc0:0:1", "doc0", "Video A", 30.0, 60.0, "Anna Lindberg i Stockholm ..."),
    ("doc1:0:0", "doc1", "Video B", 0.0, 30.0, "Sverige igen ..."),
]
_MENTIONS = [
    ("1111111111111111", "doc0:0:0"),
    ("1111111111111111", "doc0:0:1"),
    ("1111111111111111", "doc1:0:0"),
    ("2222222222222222", "doc0:0:0"),
    ("3333333333333333", "doc0:0:1"),
    ("4444444444444444", "doc0:0:1"),
]
_RELATIONSHIPS = [
    ("1111111111111111", "2222222222222222", "Regeringen styr Sverige.", "doc0:0:0"),
    ("3333333333333333", "4444444444444444", "Anna Lindberg bor i Stockholm.", "doc0:0:1"),
    # same undirected pair, reversed + a second chunk — the subgraph view must
    # collapse these into ONE edge (no double-render in the force layout).
    ("2222222222222222", "1111111111111111", "Sverige har en regering.", "doc1:0:0"),
]

_DECLARED = {
    "identity": {"key_fields": ["doc_id"]},
    "graph_presets": {"clip_title_column": "namn"},
    "capabilities": {"graph": "kg_entities"},
}


def _write_graph(db: Path) -> None:
    lance.write_dataset(
        pa.table(
            {
                "entity_id": [e[0] for e in _ENTITIES],
                "name": [e[1] for e in _ENTITIES],
                "entity_type": [e[2] for e in _ENTITIES],
                "name_lower": [e[1].lower() for e in _ENTITIES],
                "mention_count": [e[3] for e in _ENTITIES],
            }
        ),
        str(db / "kg_entities.lance"),
        mode="overwrite",
    )
    lance.write_dataset(
        pa.table(
            {
                "chunk_id": [c[0] for c in _CHUNKS],
                "doc_id": [c[1] for c in _CHUNKS],
                "namn": [c[2] for c in _CHUNKS],
                "start_s": [c[3] for c in _CHUNKS],
                "end_s": [c[4] for c in _CHUNKS],
                "text": [c[5] for c in _CHUNKS],
            }
        ),
        str(db / "kg_chunks.lance"),
        mode="overwrite",
    )
    lance.write_dataset(
        pa.table(
            {
                "source_entity_id": [m[0] for m in _MENTIONS],
                "target_chunk_id": [m[1] for m in _MENTIONS],
            }
        ),
        str(db / "kg_mentions.lance"),
        mode="overwrite",
    )
    lance.write_dataset(
        pa.table(
            {
                "source_entity_id": [r[0] for r in _RELATIONSHIPS],
                "target_entity_id": [r[1] for r in _RELATIONSHIPS],
                "relationship_type": ["RELATIONSHIP"] * len(_RELATIONSHIPS),
                "description": [r[2] for r in _RELATIONSHIPS],
                "chunk_id": [r[3] for r in _RELATIONSHIPS],
                "doc_id": [r[3].split(":")[0] for r in _RELATIONSHIPS],
            }
        ),
        str(db / "kg_relationships.lance"),
        mode="overwrite",
    )


def _make_app(root: Path, declared: dict) -> FastAPI:
    desc_dir = root / "descriptors"
    desc_dir.mkdir(exist_ok=True)
    (desc_dir / "corpus.json").write_text(json.dumps(declared))
    db = root / "corpus.lance"
    db.mkdir(exist_ok=True)
    settings = Settings(RAUDIO_DB=db, MEDIA_DB_ROOT=root, MEDIA_DESCRIPTOR_DIR=desc_dir)
    app = FastAPI()
    register_handlers(app)
    app.include_router(graph.router)
    app.state.resources = AppState(db_path=db, names=[], chunks=None, settings=settings)
    return app


def _built_client(tmp_path: Path, declared: dict = _DECLARED) -> TestClient:
    db = tmp_path / "corpus.lance"
    db.mkdir()
    _write_graph(db)
    return TestClient(_make_app(tmp_path, declared))


@pytest.mark.parametrize(
    ("query", "cap", "expected"),
    [
        ("MATCH (a:Entity) RETURN a.name", 200, "MATCH (a:Entity) RETURN a.name LIMIT 200"),
        ("MATCH (a:Entity) RETURN a.name LIMIT 5", 200, "MATCH (a:Entity) RETURN a.name LIMIT 5"),
        # the bug: a user LIMIT above the cap must be clamped, not honored.
        (
            "MATCH (a:Entity) RETURN a.name LIMIT 999999",
            200,
            "MATCH (a:Entity) RETURN a.name LIMIT 200",
        ),
        # 'limit' as a string literal must NOT suppress the appended cap.
        (
            "MATCH (a:Entity) WHERE a.name = 'limit' RETURN a.name",
            50,
            "MATCH (a:Entity) WHERE a.name = 'limit' RETURN a.name LIMIT 50",
        ),
        # trailing whitespace / mixed case still matches.
        (
            "MATCH (a:Entity) RETURN a.name limit 9000  ",
            100,
            "MATCH (a:Entity) RETURN a.name LIMIT 100",
        ),
    ],
)
def test_enforce_limit(query: str, cap: int, expected: str) -> None:
    assert _enforce_limit(query, cap) == expected


def test_status_absent(tmp_path: Path) -> None:
    """Capability declared but no kg_* tables → built:false, never a 500."""
    with TestClient(_make_app(tmp_path, _DECLARED)) as client:
        body = client.get("/api/graph/status").json()
    assert body == {"built": False, "entities": 0, "relations": 0, "mentions": 0, "videos": 0}


def test_status_undeclared_capability(tmp_path: Path) -> None:
    """kg_* tables on disk but no declared graph capability → built:false."""
    db = tmp_path / "corpus.lance"
    db.mkdir()
    _write_graph(db)
    declared = {"identity": {"key_fields": ["doc_id"]}, "capabilities": {}}
    with TestClient(_make_app(tmp_path, declared)) as client:
        assert client.get("/api/graph/status").json()["built"] is False


def test_status_built(tmp_path: Path) -> None:
    with _built_client(tmp_path) as client:
        body = client.get("/api/graph/status").json()
    assert body["built"] is True
    assert body["entities"] == 4
    assert body["videos"] == 2


def test_unknown_dataset_is_404(tmp_path: Path) -> None:
    with _built_client(tmp_path) as client:
        assert client.get("/api/graph/status", params={"dataset": "nope"}).status_code == 404


def test_cypher_limit_enforced_end_to_end(tmp_path: Path) -> None:
    """A user LIMIT above the request cap returns at most ``limit`` rows."""
    with _built_client(tmp_path) as client:
        body = client.post(
            "/api/graph/cypher",
            json={"query": "MATCH (a:Entity) RETURN a.name LIMIT 999999", "limit": 2},
        ).json()
    assert body["error"] is None
    assert len(body["rows"]) == 2


def test_cypher_rejects_multi_statement(tmp_path: Path) -> None:
    with _built_client(tmp_path) as client:
        body = client.post(
            "/api/graph/cypher",
            json={"query": "MATCH (a:Entity) RETURN a.name; MATCH (b:Chunk) RETURN b.doc_id"},
        ).json()
    assert "single statement" in (body["error"] or "").lower()


@pytest.mark.parametrize(
    "bad_id",
    [
        "not-a-hex-id",
        "x';DROP--",  # injection payload — must never reach Cypher
        "ZZZZZZZZZZZZZZZZ",  # 16 chars but not hex
        "1111",  # too short
    ],
)
def test_entity_malformed_id_is_inert(tmp_path: Path, bad_id: str) -> None:
    """Non-hex ids are whitelisted out before Cypher: 200 with ``entity: null``."""
    with _built_client(tmp_path) as client:
        resp = client.get(f"/api/graph/entity/{bad_id}")
    assert resp.status_code == 200
    assert resp.json()["entity"] is None


def test_entity_known_id_resolves(tmp_path: Path) -> None:
    with _built_client(tmp_path) as client:
        body = client.get("/api/graph/entity/1111111111111111").json()
    assert body["entity"]["name"] == "Sverige"
    assert len(body["clips"]) == 3  # Sverige is mentioned in 3 chunks
    # The clip title column comes from the descriptor's graph_presets.
    assert {c["namn"] for c in body["clips"]} == {"Video A", "Video B"}


def test_clip_title_falls_back_to_doc_id_without_presets(tmp_path: Path) -> None:
    """No ``graph_presets`` declared → clips still serve, titled by doc id."""
    declared = {"identity": {"key_fields": ["doc_id"]}, "capabilities": {"graph": "kg_entities"}}
    with _built_client(tmp_path, declared) as client:
        body = client.get("/api/graph/entity/1111111111111111").json()
    assert len(body["clips"]) == 3
    assert {c["namn"] for c in body["clips"]} == {"doc0", "doc1"}


def test_subgraph_dedupes_reverse_edges(tmp_path: Path) -> None:
    """Sverige↔Regeringen appears as a->b, b->a and across two chunks — the
    overview subgraph must render it as a single undirected edge."""
    with _built_client(tmp_path) as client:
        body = client.get("/api/graph/subgraph?limit=50").json()
    pairs = [tuple(sorted((e["source"], e["target"]))) for e in body["edges"]]
    assert len(pairs) == len(set(pairs))  # no duplicate undirected pair
    sv, reg = "1111111111111111", "2222222222222222"
    assert sum(1 for p in pairs if p == tuple(sorted((sv, reg)))) == 1


def test_search_matches_by_substring(tmp_path: Path) -> None:
    with _built_client(tmp_path) as client:
        body = client.get("/api/graph/search", params={"q": "sver"}).json()
    assert body["built"] is True
    assert [m["name"] for m in body["matches"]] == ["Sverige"]
    assert body["matches"][0]["videos"] == 2
