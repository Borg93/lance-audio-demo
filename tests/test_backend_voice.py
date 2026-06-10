"""Voice-similarity endpoints over synthetic speaker_embeddings/speakers tables.

Covers the contract the voice search UI depends on — status built/unbuilt, the
three anchor forms (turn_id | speaker | t), exactly-one-anchor validation, the
same-doc exclusion toggle, the turn→max-overlap-chunk join, the uniform Hit
shape (+ voice fields), and the result-count cap. No encoder runs anywhere:
anchors are read from Lance, so planted one-hot voiceprints make the cosine
ranking exact.
"""

from __future__ import annotations

from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import pytest
from backend import create_app
from fastapi.testclient import TestClient

from fakes import make_doc
from raudio.ingest.ingest import ingest_many

VOICE_DIM = 256

# Inline mirror of the voice-table design (the canonical pa.schema constants
# land in raudio.model.schema with the embed-speaker-turns pipeline work).
SPEAKER_EMBEDDINGS_SCHEMA = pa.schema(
    [
        pa.field("doc_id", pa.string(), nullable=False),
        pa.field("turn_id", pa.int32(), nullable=False),
        pa.field("speaker_label", pa.string(), nullable=False),
        pa.field("start", pa.float32()),
        pa.field("end", pa.float32()),
        pa.field("duration", pa.float32()),
        pa.field("embedding", pa.list_(pa.float32(), VOICE_DIM)),
    ]
)
SPEAKERS_SCHEMA = pa.schema(
    [
        pa.field("doc_id", pa.string(), nullable=False),
        pa.field("speaker_label", pa.string(), nullable=False),
        pa.field("n_turns", pa.int32()),
        pa.field("total_duration", pa.float32()),
        pa.field("embedding", pa.list_(pa.float32(), VOICE_DIM)),
        pa.field("speaker_cluster", pa.int32()),
        pa.field("speaker_name", pa.string()),
    ]
)


def _unit(axis: int, leak_axis: int | None = None, leak: float = 0.0) -> np.ndarray:
    """L2-normalized one-hot(ish) voiceprint — exact cosine geometry for tests."""
    v = np.zeros(VOICE_DIM, dtype=np.float32)
    v[axis] = 1.0
    if leak_axis is not None:
        v[leak_axis] = leak
    return v / np.linalg.norm(v)


# Voice X speaks in docs A and B; voice Y in docs A and C. The leaked axes keep
# same-voice pairs at cosine ≈ 0.9988 (distance ≈ 0.0012) and cross-voice at 0.
VX = _unit(0)
VX_A = _unit(0, leak_axis=2, leak=0.05)
VY_A = _unit(1, leak_axis=3, leak=0.05)
VY_C = _unit(1)


def _embedding_array(vecs: list[np.ndarray]) -> pa.Array:
    return pa.array([v.tolist() for v in vecs], type=pa.list_(pa.float32(), VOICE_DIM))


def _write_speaker_embeddings(
    db: Path, rows: list[tuple[str, int, str, float, float, np.ndarray]]
) -> None:
    """rows: (doc_id, turn_id, speaker_label, start, end, embedding)."""
    table = pa.table(
        {
            "doc_id": pa.array([r[0] for r in rows], pa.string()),
            "turn_id": pa.array([r[1] for r in rows], pa.int32()),
            "speaker_label": pa.array([r[2] for r in rows], pa.string()),
            "start": pa.array([r[3] for r in rows], pa.float32()),
            "end": pa.array([r[4] for r in rows], pa.float32()),
            "duration": pa.array([r[4] - r[3] for r in rows], pa.float32()),
            "embedding": _embedding_array([r[5] for r in rows]),
        },
        schema=SPEAKER_EMBEDDINGS_SCHEMA,
    )
    lance.write_dataset(table, str(db / "speaker_embeddings.lance"), data_storage_version="2.2")


def _write_speakers(db: Path, rows: list[tuple[str, str, int, float, np.ndarray]]) -> None:
    """rows: (doc_id, speaker_label, n_turns, total_duration, centroid)."""
    table = pa.table(
        {
            "doc_id": pa.array([r[0] for r in rows], pa.string()),
            "speaker_label": pa.array([r[1] for r in rows], pa.string()),
            "n_turns": pa.array([r[2] for r in rows], pa.int32()),
            "total_duration": pa.array([r[3] for r in rows], pa.float32()),
            "embedding": _embedding_array([r[4] for r in rows]),
            "speaker_cluster": pa.array([-1] * len(rows), pa.int32()),
            "speaker_name": pa.array([None] * len(rows), pa.string()),
        },
        schema=SPEAKERS_SCHEMA,
    )
    lance.write_dataset(table, str(db / "speakers.lance"), data_storage_version="2.2")


def _doc_ids_by_path(db: Path) -> dict[str, str]:
    rows = (
        lance.dataset(str(db / "chunks.lance"))
        .to_table(columns=["doc_id", "audio_path"])
        .to_pylist()
    )
    return {r["audio_path"]: r["doc_id"] for r in rows}


def _voice_db(tmp_path: Path) -> tuple[Path, str, str, str]:
    """Chunks for three docs + their doc ids (A has two 5s chunks; B/C one each)."""
    db = tmp_path / "transcripts.lance"
    ingest_many(
        db,
        [
            make_doc("input/a.mp4", ["alpha zero", "alpha one"]),
            make_doc("input/b.mp4", ["bravo zero"]),
            make_doc("input/c.mp4", ["charlie zero"]),
        ],
    )
    ids = _doc_ids_by_path(db)
    return db, ids["input/a.mp4"], ids["input/b.mp4"], ids["input/c.mp4"]


def _voice_rows(a: str, b: str, c: str) -> list[tuple[str, int, str, float, float, np.ndarray]]:
    # A's turn 0 spans both of A's chunks: overlap 2s with chunk 0 ([0,5)) and
    # 4s with chunk 1 ([5,10)) — the join must pick chunk 1 ("alpha one").
    return [
        (a, 0, "SPEAKER_00", 3.0, 9.0, VX_A),
        (a, 1, "SPEAKER_01", 0.5, 2.5, VY_A),
        (b, 0, "SPEAKER_00", 1.0, 4.0, VX),
        (c, 0, "SPEAKER_00", 1.0, 4.0, VY_C),
    ]


@pytest.fixture
def client_unbuilt(tmp_path):
    db, _, _, _ = _voice_db(tmp_path)
    return TestClient(create_app(db))


@pytest.fixture
def client_built(tmp_path):
    db, a, b, c = _voice_db(tmp_path)
    _write_speaker_embeddings(db, _voice_rows(a, b, c))
    _write_speakers(
        db,
        [
            (a, "SPEAKER_00", 1, 6.0, VX_A),
            (a, "SPEAKER_01", 1, 2.0, VY_A),
            (b, "SPEAKER_00", 1, 3.0, VX),
            (c, "SPEAKER_00", 1, 3.0, VY_C),
        ],
    )
    return TestClient(create_app(db)), (a, b, c)


@pytest.fixture
def client_embeddings_only(tmp_path):
    db, a, b, c = _voice_db(tmp_path)
    _write_speaker_embeddings(db, _voice_rows(a, b, c))
    return TestClient(create_app(db)), (a, b, c)


def _similar(client: TestClient, **params) -> dict:
    r = client.get("/api/voice/similar", params=params)
    assert r.status_code == 200, (params, r.status_code, r.text)
    return r.json()


class TestStatus:
    def test_unbuilt(self, client_unbuilt) -> None:
        r = client_unbuilt.get("/api/voice/status")
        assert r.status_code == 200
        assert r.json() == {"built": False, "turns": 0, "speakers": 0}

    def test_built_counts(self, client_built) -> None:
        client, _ = client_built
        assert client.get("/api/voice/status").json() == {
            "built": True,
            "turns": 4,
            "speakers": 4,
        }

    def test_embeddings_without_speakers(self, client_embeddings_only) -> None:
        client, _ = client_embeddings_only
        assert client.get("/api/voice/status").json() == {
            "built": True,
            "turns": 4,
            "speakers": 0,
        }


class TestAnchors:
    def test_turn_anchor_finds_same_voice_elsewhere(self, client_built) -> None:
        client, (a, b, _) = client_built
        body = _similar(client, doc_id=b, turn_id=0)
        assert body["query"] == {
            "doc_id": b,
            "speaker_label": "SPEAKER_00",
            "turn_id": 0,
            "turn_start": 1.0,
            "turn_end": 4.0,
        }
        top = body["hits"][0]
        assert (top["doc_id"], top["speaker_label"]) == (a, "SPEAKER_00")
        assert top["_distance"] < 0.01  # voice X in A is ~the anchor voice

    def test_speaker_anchor_uses_centroid(self, client_built) -> None:
        client, (a, b, _) = client_built
        body = _similar(client, doc_id=b, speaker="SPEAKER_00")
        assert body["query"]["speaker_label"] == "SPEAKER_00"
        assert body["query"]["turn_id"] is None  # centroid anchor, not one turn
        top = body["hits"][0]
        assert (top["doc_id"], top["speaker_label"]) == (a, "SPEAKER_00")

    def test_time_anchor_resolves_covering_turn(self, client_built) -> None:
        client, (a, b, _) = client_built
        body = _similar(client, doc_id=a, t=6.0)  # inside A's turn 0 [3, 9]
        assert body["query"] == {
            "doc_id": a,
            "speaker_label": "SPEAKER_00",
            "turn_id": 0,
            "turn_start": 3.0,
            "turn_end": 9.0,
        }
        assert body["hits"][0]["doc_id"] == b  # voice X's other doc

    def test_hits_keep_voice_ranking(self, client_built) -> None:
        client, (_, b, _) = client_built
        hits = _similar(client, doc_id=b, turn_id=0)["hits"]
        distances = [h["_distance"] for h in hits]
        assert distances == sorted(distances)


class TestAnchorValidation:
    def test_missing_anchor_is_400(self, client_built) -> None:
        client, (_, b, _) = client_built
        assert client.get("/api/voice/similar", params={"doc_id": b}).status_code == 400

    @pytest.mark.parametrize(
        "extra",
        [
            {"turn_id": 0, "speaker": "SPEAKER_00"},
            {"turn_id": 0, "t": 1.5},
            {"speaker": "SPEAKER_00", "t": 1.5},
            {"turn_id": 0, "speaker": "SPEAKER_00", "t": 1.5},
        ],
    )
    def test_ambiguous_anchor_is_400(self, client_built, extra: dict) -> None:
        client, (_, b, _) = client_built
        r = client.get("/api/voice/similar", params={"doc_id": b, **extra})
        assert r.status_code == 400

    def test_non_whitelisted_doc_id_is_400(self, client_built) -> None:
        client, _ = client_built
        r = client.get("/api/voice/similar", params={"doc_id": "x';DROP--", "turn_id": 0})
        assert r.status_code == 400

    def test_unknown_turn_is_404(self, client_built) -> None:
        client, (_, b, _) = client_built
        assert (
            client.get("/api/voice/similar", params={"doc_id": b, "turn_id": 99}).status_code == 404
        )

    def test_unknown_speaker_is_404(self, client_built) -> None:
        client, (_, b, _) = client_built
        r = client.get("/api/voice/similar", params={"doc_id": b, "speaker": "SPEAKER_42"})
        assert r.status_code == 404

    def test_uncovered_time_is_404(self, client_built) -> None:
        client, (a, _, _) = client_built
        assert client.get("/api/voice/similar", params={"doc_id": a, "t": 99.0}).status_code == 404

    def test_unknown_doc_is_404(self, client_built) -> None:
        client, _ = client_built
        r = client.get("/api/voice/similar", params={"doc_id": "ffffffffffffffff", "turn_id": 0})
        assert r.status_code == 404


class TestUnbuilt:
    def test_similar_unbuilt_is_503(self, client_unbuilt) -> None:
        r = client_unbuilt.get(
            "/api/voice/similar", params={"doc_id": "0123456789abcdef", "turn_id": 0}
        )
        assert r.status_code == 503

    def test_speaker_anchor_without_speakers_table_is_503(self, client_embeddings_only) -> None:
        client, (_, b, _) = client_embeddings_only
        r = client.get("/api/voice/similar", params={"doc_id": b, "speaker": "SPEAKER_00"})
        assert r.status_code == 503


class TestExcludeSameDoc:
    def test_default_excludes_anchor_doc(self, client_built) -> None:
        client, (_, b, _) = client_built
        hits = _similar(client, doc_id=b, turn_id=0)["hits"]
        assert hits and all(h["doc_id"] != b for h in hits)

    def test_off_returns_anchor_turn_first(self, client_built) -> None:
        client, (_, b, _) = client_built
        hits = _similar(client, doc_id=b, turn_id=0, exclude_same_doc=False)["hits"]
        assert hits[0]["doc_id"] == b
        assert hits[0]["_distance"] < 1e-5  # the anchor turn itself
        assert hits[0]["turn_score"] > 0.99999


class TestHitShape:
    def test_uniform_hit_plus_voice_fields(self, client_built) -> None:
        client, (a, b, _) = client_built
        top = _similar(client, doc_id=b, turn_id=0)["hits"][0]
        # The uniform search Hit shape (chunks payload + parsed alignments) …
        assert {
            "doc_id",
            "audio_path",
            "speech_id",
            "chunk_id",
            "start",
            "end",
            "duration",
            "text",
            "language",
            "alignments",
        } <= top.keys()
        assert top["alignments"] == []  # alignments_json isn't projected, like /api/search
        # … augmented with the matched turn's voice fields.
        assert top["speaker_label"] == "SPEAKER_00"
        assert (top["turn_start"], top["turn_end"], top["turn_id"]) == (3.0, 9.0, 0)
        assert top["turn_score"] == pytest.approx(1.0 - top["_distance"])
        assert top["doc_id"] == a

    def test_join_picks_max_overlap_chunk(self, client_built) -> None:
        client, (_, b, _) = client_built
        top = _similar(client, doc_id=b, turn_id=0)["hits"][0]
        # A's turn [3, 9] overlaps chunk 0 ([0,5)) by 2s and chunk 1 ([5,10)) by 4s.
        assert (top["chunk_id"], top["text"]) == (1, "alpha one")


class TestResultCap:
    @pytest.fixture
    def client_many(self, tmp_path):
        db = tmp_path / "transcripts.lance"
        n_chunks = 120
        ingest_many(db, [make_doc("input/d.mp4", [f"chunk {i}" for i in range(n_chunks)])])
        d = _doc_ids_by_path(db)["input/d.mp4"]
        # One turn per 5s chunk, each strictly inside its chunk → 1:1 join.
        _write_speaker_embeddings(
            db,
            [(d, i, "SPEAKER_00", i * 5 + 1.0, i * 5 + 4.0, _unit(i)) for i in range(n_chunks)],
        )
        return TestClient(create_app(db)), d

    def test_n_is_capped_at_100(self, client_many) -> None:
        client, d = client_many
        body = _similar(client, doc_id=d, turn_id=0, n=500, exclude_same_doc=False)
        assert len(body["hits"]) == 100  # 120 candidates, n clamped (not 422)
