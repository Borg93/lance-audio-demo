"""Diarization endpoint (media_api port) over a synthetic turns table.

Port of ``tests/test_backend_diarization.py`` to the descriptor-driven
``backend.media_api.diarization`` router: ``built: false`` when the capability
is undeclared / the table or doc is absent, turns sorted by start, speakers
sorted-distinct, the doc_id whitelist (now the descriptor's
``identity.doc_key_pattern``) enforced before any filter interpolation — plus
the per-request re-read that picks a rebuild up without a restart.
"""

from __future__ import annotations

import json
from pathlib import Path

import lance
import pyarrow as pa
import pytest
from backend.core.config import Settings
from backend.core.handlers import register_handlers
from backend.media_api import diarization
from backend.state import AppState
from fastapi import FastAPI
from fastapi.testclient import TestClient

DOC_ID = "0123456789abcdef"
# (turn_id, speaker_label, start, end) — deliberately NOT pre-sorted by start.
TURNS = [
    (0, "SPEAKER_01", 5.0, 6.0),
    (1, "SPEAKER_00", 1.0, 2.0),
    (2, "SPEAKER_01", 3.0, 4.0),
]

_DECLARED = {
    "identity": {"key_fields": ["doc_id"], "doc_key": "doc_id", "doc_key_pattern": "^[a-f0-9]{16}$"},
    "capabilities": {"diarization": "speaker_turns"},
}


def _write_turns(db: Path) -> None:
    table = pa.table(
        {
            "doc_id": pa.array([DOC_ID] * len(TURNS), type=pa.string()),
            "turn_id": pa.array([t[0] for t in TURNS], type=pa.int32()),
            "speaker_label": pa.array([t[1] for t in TURNS], type=pa.string()),
            "start": pa.array([t[2] for t in TURNS], type=pa.float32()),
            "end": pa.array([t[3] for t in TURNS], type=pa.float32()),
        }
    )
    lance.write_dataset(table, str(db / "speaker_turns.lance"))


def _make_app(root: Path, declared: dict) -> FastAPI:
    desc_dir = root / "descriptors"
    desc_dir.mkdir(exist_ok=True)
    (desc_dir / "corpus.json").write_text(json.dumps(declared))
    db = root / "corpus.lance"
    db.mkdir(exist_ok=True)
    settings = Settings(RAUDIO_DB=db, MEDIA_DB_ROOT=root, MEDIA_DESCRIPTOR_DIR=desc_dir)
    app = FastAPI()
    register_handlers(app)
    app.include_router(diarization.router)
    app.state.resources = AppState(db_path=db, names=[], chunks=None, settings=settings)
    return app


@pytest.fixture
def client_no_turns(tmp_path):
    return TestClient(_make_app(tmp_path, _DECLARED))


@pytest.fixture
def client_with_turns(tmp_path):
    db = tmp_path / "corpus.lance"
    db.mkdir()
    _write_turns(db)
    return TestClient(_make_app(tmp_path, _DECLARED))


class TestBuiltContract:
    def test_no_table_is_built_false(self, client_no_turns) -> None:
        r = client_no_turns.get(f"/api/diarization/{DOC_ID}")
        assert r.status_code == 200
        assert r.json() == {"built": False, "doc_id": DOC_ID, "turns": [], "speakers": []}

    def test_undeclared_capability_is_built_false(self, tmp_path) -> None:
        """A turns table on disk but no declared capability → the empty contract."""
        db = tmp_path / "corpus.lance"
        db.mkdir()
        _write_turns(db)
        declared = {**_DECLARED, "capabilities": {}}
        client = TestClient(_make_app(tmp_path, declared))
        r = client.get(f"/api/diarization/{DOC_ID}")
        assert r.status_code == 200
        assert r.json()["built"] is False

    def test_unknown_doc_is_built_false(self, client_with_turns) -> None:
        r = client_with_turns.get("/api/diarization/ffffffffffffffff")
        assert r.status_code == 200
        assert r.json()["built"] is False

    def test_turns_sorted_by_start_and_speakers_distinct(self, client_with_turns) -> None:
        body = client_with_turns.get(f"/api/diarization/{DOC_ID}").json()
        assert body["built"] is True
        starts = [t["start"] for t in body["turns"]]
        assert starts == [1.0, 3.0, 5.0]  # sorted ascending, not write order
        assert body["speakers"] == ["SPEAKER_00", "SPEAKER_01"]  # sorted-distinct

    def test_rebuild_is_picked_up_without_restart(self, client_no_turns, tmp_path) -> None:
        """The table is re-read per request: a diarization pass that lands after
        startup starts serving immediately."""
        assert client_no_turns.get(f"/api/diarization/{DOC_ID}").json()["built"] is False
        _write_turns(tmp_path / "corpus.lance")
        assert client_no_turns.get(f"/api/diarization/{DOC_ID}").json()["built"] is True


class TestDatasetParam:
    def test_unknown_dataset_is_404(self, client_with_turns) -> None:
        r = client_with_turns.get(f"/api/diarization/{DOC_ID}", params={"dataset": "nope"})
        assert r.status_code == 404

    def test_explicit_default_dataset_matches(self, client_with_turns) -> None:
        implicit = client_with_turns.get(f"/api/diarization/{DOC_ID}").json()
        explicit = client_with_turns.get(
            f"/api/diarization/{DOC_ID}", params={"dataset": "corpus"}
        ).json()
        assert explicit == implicit


class TestDocIdGuard:
    @pytest.mark.parametrize(
        "bad",
        [
            "x';DROP--",  # SQL/filter-injection payload
            "x' OR '1'='1",
            "ZZZZZZZZZZZZZZZZ",  # 16 chars but not hex
            "abc",  # too short
            "0123456789abcdef0",  # too long
        ],
    )
    def test_non_whitelisted_doc_id_is_400(self, client_with_turns, bad: str) -> None:
        assert client_with_turns.get(f"/api/diarization/{bad}").status_code == 400

    def test_pattern_comes_from_descriptor(self, tmp_path) -> None:
        """A dataset with a different identity pattern accepts its own keys."""
        db = tmp_path / "corpus.lance"
        db.mkdir()
        doc = "clip_007"
        table = pa.table(
            {
                "doc_id": pa.array([doc], pa.string()),
                "turn_id": pa.array([0], pa.int32()),
                "speaker_label": pa.array(["SPEAKER_00"], pa.string()),
                "start": pa.array([0.0], pa.float32()),
                "end": pa.array([1.0], pa.float32()),
            }
        )
        lance.write_dataset(table, str(db / "speaker_turns.lance"))
        declared = {
            "identity": {
                "key_fields": ["doc_id"],
                "doc_key": "doc_id",
                "doc_key_pattern": "^[A-Za-z0-9_]{1,32}$",
            },
            "capabilities": {"diarization": "speaker_turns"},
        }
        client = TestClient(_make_app(tmp_path, declared))
        assert client.get(f"/api/diarization/{doc}").json()["built"] is True
        assert client.get(f"/api/diarization/{DOC_ID}").json()["built"] is False  # still valid hex
        assert client.get("/api/diarization/x';DROP--").status_code == 400
