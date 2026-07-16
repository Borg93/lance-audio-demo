"""Voice endpoints (media_api port) over synthetic voice tables + a minimal descriptor.

Port of ``tests/test_backend_voice.py`` to the descriptor-driven
``backend.media_api.voice`` router: status built/unbuilt, the three anchor
forms (turn_id | speaker | t), exactly-one-anchor validation, the same-doc
exclusion toggle, the turn→max-overlap-chunk join, the uniform Hit shape
(+ voice fields), the result-count cap, the upload anchor form, identity
clusters, and the per-hit ``speaker_cluster``. No encoder runs anywhere:
anchors are read from Lance, so planted one-hot voiceprints make the cosine
ranking exact. All table/column names reach the router through a tmp-dir
descriptor — nothing is baked into the served code.
"""

from __future__ import annotations

import io
import json
import wave
from pathlib import Path

import httpx
import lance
import numpy as np
import pyarrow as pa
import pytest
from backend.core.config import Settings
from backend.core.handlers import register_handlers
from backend.media_api import voice, voice_service
from backend.media_api.wespeaker import VOICE_EMBED_DIM
from backend.state import AppState
from fastapi import FastAPI
from fastapi.testclient import TestClient

DOC_A = "aaaaaaaaaaaaaaaa"
DOC_B = "bbbbbbbbbbbbbbbb"
DOC_C = "cccccccccccccccc"

_DECLARED = {
    "identity": {
        "key_fields": ["doc_id", "speech_id", "chunk_id"],
        "doc_key": "doc_id",
        "doc_key_pattern": "^[a-f0-9]{16}$",
    },
    "time": {"start": "start", "end": "end"},
    "display": {
        "title": ["audio_path", "doc_id"],
        "body": "text",
        "metadata": [
            {"field": "language", "label": "Language"},
            {"field": "duration", "label": "Duration"},
        ],
    },
    "search": {"row_table": "chunks"},
    "capabilities": {"voice": "speaker_embeddings.embedding", "speakers": "speakers"},
}


def _unit(axis: int, leak_axis: int | None = None, leak: float = 0.0) -> np.ndarray:
    """L2-normalized one-hot(ish) voiceprint — exact cosine geometry for tests."""
    v = np.zeros(VOICE_EMBED_DIM, dtype=np.float32)
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
    return pa.array([v.tolist() for v in vecs], type=pa.list_(pa.float32(), VOICE_EMBED_DIM))


def _write_chunks(db: Path, docs: dict[str, list[str]]) -> None:
    """One 5s chunk per text per doc (speech_id 0), mirroring the old fixtures."""
    rows = [
        (doc_id, 0, i, f"input/{doc_id}.mp4", i * 5.0, i * 5.0 + 5.0, 5.0, text, "sv")
        for doc_id, texts in docs.items()
        for i, text in enumerate(texts)
    ]
    table = pa.table(
        {
            "doc_id": pa.array([r[0] for r in rows], pa.string()),
            "speech_id": pa.array([r[1] for r in rows], pa.int32()),
            "chunk_id": pa.array([r[2] for r in rows], pa.int32()),
            "audio_path": pa.array([r[3] for r in rows], pa.string()),
            "start": pa.array([r[4] for r in rows], pa.float32()),
            "end": pa.array([r[5] for r in rows], pa.float32()),
            "duration": pa.array([r[6] for r in rows], pa.float32()),
            "text": pa.array([r[7] for r in rows], pa.string()),
            "language": pa.array([r[8] for r in rows], pa.string()),
        }
    )
    lance.write_dataset(table, str(db / "chunks.lance"))


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
        }
    )
    lance.write_dataset(table, str(db / "speaker_embeddings.lance"))


def _write_speakers(
    db: Path,
    rows: list[tuple[str, str, int, float, np.ndarray]],
    clusters: list[int] | None = None,
) -> None:
    """rows: (doc_id, speaker_label, n_turns, total_duration, centroid).

    ``clusters`` plants per-row ``speaker_cluster`` ids (what a clustering run
    would write); default = all -1 (unclustered).
    """
    table = pa.table(
        {
            "doc_id": pa.array([r[0] for r in rows], pa.string()),
            "speaker_label": pa.array([r[1] for r in rows], pa.string()),
            "n_turns": pa.array([r[2] for r in rows], pa.int32()),
            "total_duration": pa.array([r[3] for r in rows], pa.float32()),
            "embedding": _embedding_array([r[4] for r in rows]),
            "speaker_cluster": pa.array(clusters or [-1] * len(rows), pa.int32()),
            "speaker_name": pa.array([None] * len(rows), pa.string()),
        }
    )
    lance.write_dataset(table, str(db / "speakers.lance"))


def _make_app(root: Path, declared: dict) -> FastAPI:
    """A minimal media_api app over ``root/corpus.lance`` + a tmp descriptor."""
    desc_dir = root / "descriptors"
    desc_dir.mkdir(exist_ok=True)
    (desc_dir / "corpus.json").write_text(json.dumps(declared))
    db = root / "corpus.lance"
    db.mkdir(exist_ok=True)
    settings = Settings(RAUDIO_DB=db, MEDIA_DB_ROOT=root, MEDIA_DESCRIPTOR_DIR=desc_dir)
    app = FastAPI()
    register_handlers(app)
    app.include_router(voice.router)
    app.state.resources = AppState(db_path=db, names=[], chunks=None, settings=settings)
    return app


def _voice_db(tmp_path: Path) -> Path:
    """Chunks for three docs (A has two 5s chunks; B/C one each)."""
    db = tmp_path / "corpus.lance"
    db.mkdir()
    _write_chunks(
        db, {DOC_A: ["alpha zero", "alpha one"], DOC_B: ["bravo zero"], DOC_C: ["charlie zero"]}
    )
    return db


def _voice_rows() -> list[tuple[str, int, str, float, float, np.ndarray]]:
    # A's turn 0 spans both of A's chunks: overlap 2s with chunk 0 ([0,5)) and
    # 4s with chunk 1 ([5,10)) — the join must pick chunk 1 ("alpha one").
    return [
        (DOC_A, 0, "SPEAKER_00", 3.0, 9.0, VX_A),
        (DOC_A, 1, "SPEAKER_01", 0.5, 2.5, VY_A),
        (DOC_B, 0, "SPEAKER_00", 1.0, 4.0, VX),
        (DOC_C, 0, "SPEAKER_00", 1.0, 4.0, VY_C),
    ]


_SPEAKER_ROWS = [
    (DOC_A, "SPEAKER_00", 1, 6.0, VX_A),
    (DOC_A, "SPEAKER_01", 1, 2.0, VY_A),
    (DOC_B, "SPEAKER_00", 1, 3.0, VX),
    (DOC_C, "SPEAKER_00", 1, 3.0, VY_C),
]


@pytest.fixture
def client_unbuilt(tmp_path):
    _voice_db(tmp_path)
    return TestClient(_make_app(tmp_path, _DECLARED))


@pytest.fixture
def client_built(tmp_path):
    db = _voice_db(tmp_path)
    _write_speaker_embeddings(db, _voice_rows())
    _write_speakers(db, _SPEAKER_ROWS)
    return TestClient(_make_app(tmp_path, _DECLARED))


@pytest.fixture
def client_embeddings_only(tmp_path):
    db = _voice_db(tmp_path)
    _write_speaker_embeddings(db, _voice_rows())
    return TestClient(_make_app(tmp_path, _DECLARED))


@pytest.fixture
def client_clustered(tmp_path):
    """client_built + planted global identities: voice X = cluster 7 across A
    and B, C's voice Y = a singleton cluster 3, A's voice Y = noise (-1)."""
    db = _voice_db(tmp_path)
    _write_speaker_embeddings(db, _voice_rows())
    _write_speakers(db, _SPEAKER_ROWS, clusters=[7, -1, 7, 3])
    return TestClient(_make_app(tmp_path, _DECLARED))


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
        assert client_built.get("/api/voice/status").json() == {
            "built": True,
            "turns": 4,
            "speakers": 4,
        }

    def test_embeddings_without_speakers(self, client_embeddings_only) -> None:
        assert client_embeddings_only.get("/api/voice/status").json() == {
            "built": True,
            "turns": 4,
            "speakers": 0,
        }

    def test_undeclared_capability_is_built_false(self, tmp_path) -> None:
        """Tables on disk but no voice capability declared → the empty contract."""
        db = _voice_db(tmp_path)
        _write_speaker_embeddings(db, _voice_rows())
        declared = {**_DECLARED, "capabilities": {}}
        client = TestClient(_make_app(tmp_path, declared))
        assert client.get("/api/voice/status").json() == {
            "built": False,
            "turns": 0,
            "speakers": 0,
        }


class TestDatasetParam:
    def test_unknown_dataset_is_404(self, client_built) -> None:
        r = client_built.get("/api/voice/status", params={"dataset": "nope"})
        assert r.status_code == 404

    def test_explicit_default_dataset_matches(self, client_built) -> None:
        implicit = client_built.get("/api/voice/status").json()
        explicit = client_built.get("/api/voice/status", params={"dataset": "corpus"}).json()
        assert explicit == implicit


class TestAnchors:
    def test_turn_anchor_finds_same_voice_elsewhere(self, client_built) -> None:
        body = _similar(client_built, doc_id=DOC_B, turn_id=0)
        assert body["query"] == {
            "doc_id": DOC_B,
            "speaker_label": "SPEAKER_00",
            "turn_id": 0,
            "turn_start": 1.0,
            "turn_end": 4.0,
        }
        top = body["hits"][0]
        assert (top["doc_id"], top["speaker_label"]) == (DOC_A, "SPEAKER_00")
        assert top["_distance"] < 0.01  # voice X in A is ~the anchor voice

    def test_speaker_anchor_uses_centroid(self, client_built) -> None:
        body = _similar(client_built, doc_id=DOC_B, speaker="SPEAKER_00")
        assert body["query"]["speaker_label"] == "SPEAKER_00"
        assert body["query"]["turn_id"] is None  # centroid anchor, not one turn
        top = body["hits"][0]
        assert (top["doc_id"], top["speaker_label"]) == (DOC_A, "SPEAKER_00")

    def test_time_anchor_resolves_covering_turn(self, client_built) -> None:
        body = _similar(client_built, doc_id=DOC_A, t=6.0)  # inside A's turn 0 [3, 9]
        assert body["query"] == {
            "doc_id": DOC_A,
            "speaker_label": "SPEAKER_00",
            "turn_id": 0,
            "turn_start": 3.0,
            "turn_end": 9.0,
        }
        assert body["hits"][0]["doc_id"] == DOC_B  # voice X's other doc

    def test_hits_keep_voice_ranking(self, client_built) -> None:
        hits = _similar(client_built, doc_id=DOC_B, turn_id=0)["hits"]
        distances = [h["_distance"] for h in hits]
        assert distances == sorted(distances)


class TestAnchorValidation:
    def test_missing_anchor_is_400(self, client_built) -> None:
        assert client_built.get("/api/voice/similar", params={"doc_id": DOC_B}).status_code == 400

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
        r = client_built.get("/api/voice/similar", params={"doc_id": DOC_B, **extra})
        assert r.status_code == 400

    def test_non_whitelisted_doc_id_is_400(self, client_built) -> None:
        r = client_built.get("/api/voice/similar", params={"doc_id": "x';DROP--", "turn_id": 0})
        assert r.status_code == 400

    def test_unknown_turn_is_404(self, client_built) -> None:
        r = client_built.get("/api/voice/similar", params={"doc_id": DOC_B, "turn_id": 99})
        assert r.status_code == 404

    def test_unknown_speaker_is_404(self, client_built) -> None:
        r = client_built.get("/api/voice/similar", params={"doc_id": DOC_B, "speaker": "SPEAKER_42"})
        assert r.status_code == 404

    def test_uncovered_time_is_404(self, client_built) -> None:
        r = client_built.get("/api/voice/similar", params={"doc_id": DOC_A, "t": 99.0})
        assert r.status_code == 404

    def test_unknown_doc_is_404(self, client_built) -> None:
        r = client_built.get(
            "/api/voice/similar", params={"doc_id": "ffffffffffffffff", "turn_id": 0}
        )
        assert r.status_code == 404


class TestUnbuilt:
    def test_similar_unbuilt_is_503(self, client_unbuilt) -> None:
        r = client_unbuilt.get(
            "/api/voice/similar", params={"doc_id": "0123456789abcdef", "turn_id": 0}
        )
        assert r.status_code == 503

    def test_speaker_anchor_without_speakers_table_is_503(self, client_embeddings_only) -> None:
        r = client_embeddings_only.get(
            "/api/voice/similar", params={"doc_id": DOC_B, "speaker": "SPEAKER_00"}
        )
        assert r.status_code == 503


class TestExcludeSameDoc:
    def test_default_excludes_anchor_doc(self, client_built) -> None:
        hits = _similar(client_built, doc_id=DOC_B, turn_id=0)["hits"]
        assert hits and all(h["doc_id"] != DOC_B for h in hits)

    def test_off_returns_anchor_turn_first(self, client_built) -> None:
        hits = _similar(client_built, doc_id=DOC_B, turn_id=0, exclude_same_doc=False)["hits"]
        assert hits[0]["doc_id"] == DOC_B
        assert hits[0]["_distance"] < 1e-5  # the anchor turn itself
        assert hits[0]["turn_score"] > 0.99999


class TestHitShape:
    def test_uniform_hit_plus_voice_fields(self, client_built) -> None:
        top = _similar(client_built, doc_id=DOC_B, turn_id=0)["hits"][0]
        # The uniform search Hit shape (descriptor-projected payload + empty
        # alignments, exactly like /api/search) …
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
        assert top["alignments"] == []  # the timing blob is never projected
        # … augmented with the matched turn's voice fields.
        assert top["speaker_label"] == "SPEAKER_00"
        assert (top["turn_start"], top["turn_end"], top["turn_id"]) == (3.0, 9.0, 0)
        assert top["turn_score"] == pytest.approx(1.0 - top["_distance"])
        assert top["doc_id"] == DOC_A

    def test_join_picks_max_overlap_chunk(self, client_built) -> None:
        top = _similar(client_built, doc_id=DOC_B, turn_id=0)["hits"][0]
        # A's turn [3, 9] overlaps chunk 0 ([0,5)) by 2s and chunk 1 ([5,10)) by 4s.
        assert (top["chunk_id"], top["text"]) == (1, "alpha one")


class TestCaptionAttach:
    def test_hits_carry_representative_frame_caption(self, tmp_path) -> None:
        """With a captions capability declared, each hit gets its frame-0 caption
        (frame_idx matched in Python, not SQL — the known planner bug)."""
        db = _voice_db(tmp_path)
        _write_speaker_embeddings(db, _voice_rows())
        keys = [(DOC_A, 0, 0), (DOC_A, 0, 1), (DOC_B, 0, 0)]
        frames = pa.table(
            {
                "doc_id": pa.array([k[0] for k in keys for _ in (0, 1)], pa.string()),
                "speech_id": pa.array([k[1] for k in keys for _ in (0, 1)], pa.int32()),
                "chunk_id": pa.array([k[2] for k in keys for _ in (0, 1)], pa.int32()),
                "frame_idx": pa.array([i for _ in keys for i in (0, 1)], pa.int32()),
                "caption": pa.array(
                    [f"{k[0]}:{k[2]}:frame{i}" for k in keys for i in (0, 1)], pa.string()
                ),
            }
        )
        lance.write_dataset(frames, str(db / "frames.lance"))
        declared = {
            **_DECLARED,
            "capabilities": {**_DECLARED["capabilities"], "captions": "frames.caption"},
        }
        client = TestClient(_make_app(tmp_path, declared))
        top = _similar(client, doc_id=DOC_B, turn_id=0)["hits"][0]
        assert top["caption"] == f"{DOC_A}:1:frame0"  # frame 0, never frame 1


class TestResultCap:
    @pytest.fixture
    def client_many(self, tmp_path):
        db = tmp_path / "corpus.lance"
        db.mkdir()
        n_chunks = 120
        doc = "d" * 16
        _write_chunks(db, {doc: [f"chunk {i}" for i in range(n_chunks)]})
        # One turn per 5s chunk, each strictly inside its chunk → 1:1 join.
        _write_speaker_embeddings(
            db,
            [(doc, i, "SPEAKER_00", i * 5 + 1.0, i * 5 + 4.0, _unit(i)) for i in range(n_chunks)],
        )
        return TestClient(_make_app(tmp_path, _DECLARED)), doc

    def test_n_is_capped_at_100(self, client_many) -> None:
        client, doc = client_many
        body = _similar(client, doc_id=doc, turn_id=0, n=500, exclude_same_doc=False)
        assert len(body["hits"]) == 100  # 120 candidates, n clamped (not 422)


def _wav_bytes(seconds: float, sample_rate: int = 16_000) -> bytes:
    """A tiny in-memory 16 kHz mono PCM16 WAV (a quiet 220 Hz tone) — stdlib wave."""
    n = round(seconds * sample_rate)
    t = np.arange(n, dtype=np.float64)
    pcm = (0.1 * np.sin(2 * np.pi * 220.0 * t / sample_rate) * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


class _FakeVoiceEncoder:
    """TurnBatchEncoder fake — one fixed raw (deliberately non-unit) row per waveform."""

    def __init__(self, vec: np.ndarray) -> None:
        self.vec = vec
        self.calls: list[list[np.ndarray]] = []

    def embed_batch(self, waveforms: list[np.ndarray]) -> np.ndarray:
        self.calls.append(waveforms)
        return np.stack([self.vec * 3.0 for _ in waveforms]).astype(np.float32)


def _upload(client: TestClient, data: bytes, **params) -> httpx.Response:
    # n rides as a query param (like the GET); the multipart body is file-only.
    return client.post(
        "/api/voice/similar",
        params={k: str(v) for k, v in params.items()},
        files={"file": ("snippet.wav", data, "audio/wav")},
    )


class TestUploadSimilar:
    """POST /api/voice/similar — the upload anchor form.

    Hermetic + offline: a fake encoder is pre-set on the lazy ``voice_encoder``
    slot (the first thing ``ensure_voice_encoder`` checks), so no model loads
    and no network is touched. ffmpeg IS exercised for real — the WAVs are
    generated in-memory with stdlib ``wave``.
    """

    @staticmethod
    def _inject(client: TestClient, vec: np.ndarray) -> _FakeVoiceEncoder:
        fake = _FakeVoiceEncoder(vec)
        client.app.state.resources.voice_encoder = fake
        return fake

    def test_upload_ranks_hits_by_voice(self, client_built) -> None:
        fake = self._inject(client_built, VX)
        r = _upload(client_built, _wav_bytes(1.0))
        assert r.status_code == 200, r.text
        body = r.json()
        # No Lance-side anchor — the query was the snippet itself.
        assert body["query"] == {
            "doc_id": None,
            "speaker_label": None,
            "turn_id": None,
            "turn_start": None,
            "turn_end": None,
        }
        hits = body["hits"]
        assert hits and fake.calls  # ranking came from the injected encoder
        assert [h["_distance"] for h in hits] == sorted(h["_distance"] for h in hits)
        # The snippet IS voice X: B's turn matches exactly, A's leaked variant next.
        assert hits[0]["doc_id"] == DOC_B
        assert hits[0]["_distance"] < 1e-5
        assert (hits[1]["doc_id"], hits[1]["speaker_label"]) == (DOC_A, "SPEAKER_00")
        # Same uniform Hit shape (+ voice fields) as the GET path.
        assert {"doc_id", "text", "start", "end", "alignments", "turn_score"} <= hits[0].keys()

    def test_long_upload_embeds_first_30s_only(self, client_built) -> None:
        fake = self._inject(client_built, VX)
        assert _upload(client_built, _wav_bytes(31.0)).status_code == 200
        (waveforms,) = fake.calls  # one embed_batch call …
        (wav,) = waveforms  # … with the capped snippet as a single "turn"
        assert wav.shape[0] == 30 * 16_000

    def test_too_short_snippet_is_400(self, client_built) -> None:
        self._inject(client_built, VX)
        r = _upload(client_built, _wav_bytes(0.2))
        assert r.status_code == 400
        assert "too short" in r.json()["detail"]

    def test_undecodable_upload_is_400(self, client_built) -> None:
        self._inject(client_built, VX)
        r = _upload(client_built, b"\x00\x01definitely-not-audio" * 16)
        assert r.status_code == 400
        assert "decode" in r.json()["detail"]

    def test_oversize_upload_is_400(self, client_built, monkeypatch) -> None:
        self._inject(client_built, VX)
        # Shrink the cap instead of allocating 25 MB: the router and service
        # read the module global at call time, so the real length check runs.
        monkeypatch.setattr(voice_service, "_MAX_UPLOAD_BYTES", 1024)
        r = _upload(client_built, _wav_bytes(1.0))  # ~32 KB > 1 KiB cap
        assert r.status_code == 400
        assert "too large" in r.json()["detail"]

    @pytest.mark.parametrize("bad", ["zeros", "nan"])
    def test_degenerate_embedding_is_400(self, client_built, bad: str) -> None:
        # Silence-shaped failures: a zero raw embedding survives l2_normalize
        # as zeros (norm floor), NaN survives as NaN — both must 400, not feed
        # the kNN an undefined cosine query.
        fill = 0.0 if bad == "zeros" else np.nan
        self._inject(client_built, np.full(VOICE_EMBED_DIM, fill, dtype=np.float32))
        r = _upload(client_built, _wav_bytes(1.0))
        assert r.status_code == 400
        assert "voiceprint" in r.json()["detail"]

    def test_n_limits_and_clamps(self, client_built) -> None:
        self._inject(client_built, VX)
        assert len(_upload(client_built, _wav_bytes(1.0), n=2).json()["hits"]) == 2
        # Oversized n is clamped (never a 422) by the same shared
        # rank_similar_turns the GET's test_n_is_capped_at_100 pins at 100.
        r = _upload(client_built, _wav_bytes(1.0), n=500)
        assert r.status_code == 200
        assert len(r.json()["hits"]) == 4  # every planted turn, not an error

    def test_unbuilt_tables_is_503(self, client_unbuilt) -> None:
        self._inject(client_unbuilt, VX)
        assert _upload(client_unbuilt, _wav_bytes(1.0)).status_code == 503


def _identity(client: TestClient, **params) -> httpx.Response:
    return client.get("/api/voice/identity", params=params)


class TestIdentity:
    """GET /api/voice/identity — global cluster membership for one speaker."""

    def test_clustered_speaker_lists_all_appearances(self, client_clustered) -> None:
        r = _identity(client_clustered, doc_id=DOC_B, speaker="SPEAKER_00")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["speaker_cluster"] == 7
        assert body["n_videos"] == 2
        # Sorted by total_duration desc: A's SPEAKER_00 (6.0s) before B's (3.0s).
        assert [(ap["doc_id"], ap["speaker_label"]) for ap in body["appearances"]] == [
            (DOC_A, "SPEAKER_00"),
            (DOC_B, "SPEAKER_00"),
        ]
        assert body["appearances"][0] == {
            "doc_id": DOC_A,
            "speaker_label": "SPEAKER_00",
            "n_turns": 1,
            "total_duration": 6.0,
        }

    def test_singleton_cluster_counts_one_video(self, client_clustered) -> None:
        body = _identity(client_clustered, doc_id=DOC_C, speaker="SPEAKER_00").json()
        assert body["speaker_cluster"] == 3
        assert body["n_videos"] == 1
        assert [ap["doc_id"] for ap in body["appearances"]] == [DOC_C]

    def test_noise_speaker_is_null_cluster_with_self_only(self, client_clustered) -> None:
        body = _identity(client_clustered, doc_id=DOC_A, speaker="SPEAKER_01").json()
        assert body["speaker_cluster"] is None
        assert body["n_videos"] == 1
        assert [(ap["doc_id"], ap["speaker_label"]) for ap in body["appearances"]] == [
            (DOC_A, "SPEAKER_01")
        ]

    def test_uncluster_run_is_null_cluster(self, client_built) -> None:
        # The default speakers table (clustering never ran): every row -1.
        body = _identity(client_built, doc_id=DOC_B, speaker="SPEAKER_00").json()
        assert body["speaker_cluster"] is None
        assert body["n_videos"] == 1

    def test_unknown_speaker_is_404(self, client_clustered) -> None:
        assert _identity(client_clustered, doc_id=DOC_B, speaker="SPEAKER_42").status_code == 404

    def test_unknown_doc_is_404(self, client_clustered) -> None:
        r = _identity(client_clustered, doc_id="ffffffffffffffff", speaker="SPEAKER_00")
        assert r.status_code == 404

    def test_non_whitelisted_doc_id_is_400(self, client_clustered) -> None:
        r = _identity(client_clustered, doc_id="x';DROP--", speaker="SPEAKER_00")
        assert r.status_code == 400

    def test_no_speakers_table_is_503(self, client_embeddings_only) -> None:
        r = _identity(client_embeddings_only, doc_id=DOC_B, speaker="SPEAKER_00")
        assert r.status_code == 503

    def test_unbuilt_is_503(self, client_unbuilt) -> None:
        r = _identity(client_unbuilt, doc_id="0123456789abcdef", speaker="SPEAKER_00")
        assert r.status_code == 503


class TestHitSpeakerCluster:
    """The per-hit ``speaker_cluster`` joined from the speakers table."""

    def test_get_hits_carry_cluster_id(self, client_clustered) -> None:
        hits = _similar(client_clustered, doc_id=DOC_B, turn_id=0)["hits"]
        by_speaker = {(h["doc_id"], h["speaker_label"]): h["speaker_cluster"] for h in hits}
        assert by_speaker[(DOC_A, "SPEAKER_00")] == 7  # voice X's identity
        assert by_speaker[(DOC_A, "SPEAKER_01")] is None  # noise (-1) → null, key present

    def test_unclustered_table_yields_null(self, client_built) -> None:
        hits = _similar(client_built, doc_id=DOC_B, turn_id=0)["hits"]
        assert hits and all(h["speaker_cluster"] is None for h in hits)

    def test_without_speakers_table_yields_null(self, client_embeddings_only) -> None:
        hits = _similar(client_embeddings_only, doc_id=DOC_B, turn_id=0)["hits"]
        assert hits and all(h["speaker_cluster"] is None for h in hits)

    def test_upload_hits_carry_cluster_id(self, client_clustered) -> None:
        client_clustered.app.state.resources.voice_encoder = _FakeVoiceEncoder(VX)
        r = _upload(client_clustered, _wav_bytes(1.0))
        assert r.status_code == 200, r.text
        top = r.json()["hits"][0]
        assert (top["doc_id"], top["speaker_label"]) == (DOC_B, "SPEAKER_00")
        assert top["speaker_cluster"] == 7
