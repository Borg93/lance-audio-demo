"""Diarization endpoint over a synthetic ``speaker_turns.lance`` table.

Covers the contract the player timeline depends on — ``built: false`` when the
table/doc is absent, turns sorted by start, speakers sorted-distinct — and the
security guard: ``doc_id`` is whitelisted (16-char hex) before it is inlined into
the Lance filter literal, so a crafted value is rejected (400), never
interpolated.
"""

from __future__ import annotations

from pathlib import Path

import lance
import pyarrow as pa
import pytest
from backend import create_app
from fastapi.testclient import TestClient

from fakes import make_doc
from raudio.ingest.ingest import ingest_many
from raudio.model.schema import SPEAKER_TURNS_SCHEMA, SPEAKER_TURNS_STORAGE_VERSION

DOC_ID = "0123456789abcdef"
# (turn_id, speaker_label, start, end) — deliberately NOT pre-sorted by start.
TURNS = [
    (0, "SPEAKER_01", 5.0, 6.0),
    (1, "SPEAKER_00", 1.0, 2.0),
    (2, "SPEAKER_01", 3.0, 4.0),
]


def _write_turns(db: Path) -> None:
    table = pa.table(
        {
            "doc_id": pa.array([DOC_ID] * len(TURNS), type=pa.string()),
            "turn_id": pa.array([t[0] for t in TURNS], type=pa.int32()),
            "speaker_label": pa.array([t[1] for t in TURNS], type=pa.string()),
            "start": pa.array([t[2] for t in TURNS], type=pa.float32()),
            "end": pa.array([t[3] for t in TURNS], type=pa.float32()),
        },
        schema=SPEAKER_TURNS_SCHEMA,
    )
    lance.write_dataset(
        table,
        str(db / "speaker_turns.lance"),
        data_storage_version=SPEAKER_TURNS_STORAGE_VERSION,
    )


def _db_with_chunks(tmp_path: Path) -> Path:
    db = tmp_path / "transcripts.lance"
    ingest_many(db, [make_doc("input/x.mp4", ["hello world"])])  # chunks table is required
    return db


@pytest.fixture
def client_no_turns(tmp_path):
    return TestClient(create_app(_db_with_chunks(tmp_path)))


@pytest.fixture
def client_with_turns(tmp_path):
    db = _db_with_chunks(tmp_path)
    _write_turns(db)
    return TestClient(create_app(db))


class TestBuiltContract:
    def test_no_table_is_built_false(self, client_no_turns) -> None:
        r = client_no_turns.get(f"/api/diarization/{DOC_ID}")
        assert r.status_code == 200
        assert r.json() == {"built": False, "doc_id": DOC_ID, "turns": [], "speakers": []}

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
