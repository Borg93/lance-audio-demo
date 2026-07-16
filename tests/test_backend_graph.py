"""Knowledge-graph endpoints over a synthetic ``kg_*.lance`` graph.

Two layers:

- :func:`test_enforce_limit_*` unit-test the pure ``_enforce_limit`` helper —
  the regression guard for the LIMIT-cap bypass (a user ``LIMIT 999999`` once
  pulled the whole table because the old guard only *appended* a cap when the
  substring ``limit`` was textually absent).
- The endpoint tests cover the contract the ``/graph`` page depends on:
  ``built: false`` when the tables are absent, the cap enforced end-to-end
  through ``lance_graph``, single-statement-only, and the ``entity_id``
  whitelist (16-char hex) that guards Cypher interpolation.
"""

from __future__ import annotations

from pathlib import Path

import lance
import pyarrow as pa
import pytest
from backend import create_app
from backend.graph.router import _enforce_limit
from fastapi.testclient import TestClient

from fakes import make_doc
from rmedia.ingest.ingest import ingest_many

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


def _db_with_chunks(tmp_path: Path) -> Path:
    """A DB carrying the base ``chunks`` table the app opens at startup.

    ``create_app`` requires it regardless of the kg_* graph (mirrors
    ``_db_with_chunks`` in the diarization test).
    """
    db = tmp_path / "transcripts.lance"
    ingest_many(db, [make_doc("input/x.mp4", ["hello world"])])
    return db


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
        ("MATCH (a:Entity) RETURN a.name limit 9000  ", 100, "MATCH (a:Entity) RETURN a.name LIMIT 100"),
    ],
)
def test_enforce_limit(query: str, cap: int, expected: str) -> None:
    assert _enforce_limit(query, cap) == expected


def test_status_absent(tmp_path: Path) -> None:
    """No kg_* tables → built:false, never a 500 (mirrors diarization)."""
    with TestClient(create_app(_db_with_chunks(tmp_path))) as client:
        body = client.get("/api/graph/status").json()
    assert body == {"built": False, "entities": 0, "relations": 0, "mentions": 0, "videos": 0}


def test_status_built(tmp_path: Path) -> None:
    db = _db_with_chunks(tmp_path)
    _write_graph(db)
    with TestClient(create_app(db)) as client:
        body = client.get("/api/graph/status").json()
    assert body["built"] is True
    assert body["entities"] == 4
    assert body["videos"] == 2


def test_cypher_limit_enforced_end_to_end(tmp_path: Path) -> None:
    """A user LIMIT above the request cap returns at most ``limit`` rows."""
    db = _db_with_chunks(tmp_path)
    _write_graph(db)
    with TestClient(create_app(db)) as client:
        body = client.post(
            "/api/graph/cypher",
            json={"query": "MATCH (a:Entity) RETURN a.name LIMIT 999999", "limit": 2},
        ).json()
    assert body["error"] is None
    assert len(body["rows"]) == 2


def test_cypher_rejects_multi_statement(tmp_path: Path) -> None:
    db = _db_with_chunks(tmp_path)
    _write_graph(db)
    with TestClient(create_app(db)) as client:
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
    db = _db_with_chunks(tmp_path)
    _write_graph(db)
    with TestClient(create_app(db)) as client:
        resp = client.get(f"/api/graph/entity/{bad_id}")
    assert resp.status_code == 200
    assert resp.json()["entity"] is None


def test_entity_known_id_resolves(tmp_path: Path) -> None:
    db = _db_with_chunks(tmp_path)
    _write_graph(db)
    with TestClient(create_app(db)) as client:
        body = client.get("/api/graph/entity/1111111111111111").json()
    assert body["entity"]["name"] == "Sverige"
    assert len(body["clips"]) == 3  # Sverige is mentioned in 3 chunks


def test_subgraph_dedupes_reverse_edges(tmp_path: Path) -> None:
    """Sverige↔Regeringen appears as a->b, b->a and across two chunks — the
    overview subgraph must render it as a single undirected edge."""
    db = _db_with_chunks(tmp_path)
    _write_graph(db)
    with TestClient(create_app(db)) as client:
        body = client.get("/api/graph/subgraph?limit=50").json()
    pairs = [tuple(sorted((e["source"], e["target"]))) for e in body["edges"]]
    assert len(pairs) == len(set(pairs))  # no duplicate undirected pair
    sv, reg = "1111111111111111", "2222222222222222"
    assert sum(1 for p in pairs if p == tuple(sorted((sv, reg)))) == 1
