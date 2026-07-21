"""Atlas endpoints over a synthetic tmp dataset — Arrow decode round-trip.

The descriptor declares two projection spaces on one row table: ``main`` is
built (non-null x/y for 4 of 5 rows) and ``alt`` exists but is all-NULL, so
/status must report built-ness from non-null presence, not column existence
(descriptor validation already guarantees the columns exist). The /points
stream is decoded back with pyarrow to assert the exact wire format: f16
coords, int32 keys, int64 rowid, DICTIONARY<int32,utf8> doc + channels, and
docFiles metadata aligned with the doc dictionary. The rowids then round-trip
through POST /chunks.
"""

from __future__ import annotations

import json
from pathlib import Path

import lance
import pyarrow as pa
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from common.core.config import Settings
from viewer.main import create_viewer_app

DOC_A = "d001"
DOC_B = "d002"
SPANS = json.dumps([{"words": [{"text": "aa", "start": 0.0, "end": 0.2}]}])

DESCRIPTOR = {
    "identity": {
        "key_fields": ["doc_id", "speech_id", "chunk_id"],
        "doc_key": "doc_id",
        "doc_key_pattern": "^[a-z0-9]{4}$",
    },
    "time": {"start": "start", "end": "end"},
    "display": {"title": ["clip_path", "doc_id"], "body": "text"},
    "search": {"row_table": "chunks", "filterable": ["language"]},
    "atlas": [
        {
            "name": "main",
            "x": "px",
            "y": "py",
            "cluster": "pc",
            "source_column": "emb",
            "table": "chunks",
        },
        {
            "name": "alt",
            "x": "qx",
            "y": "qy",
            "cluster": "qc",
            "source_column": "emb",
            "table": "chunks",
        },
    ],
    "capabilities": {"alignments": "chunks.alignments_json"},
}


def _write_chunks(db: Path) -> None:
    db.mkdir(parents=True)
    chunks = pa.table(
        {
            "doc_id": [DOC_A, DOC_A, DOC_A, DOC_B, DOC_B],
            "speech_id": pa.array([0, 0, 1, 0, 0], pa.int32()),
            "chunk_id": pa.array([0, 1, 0, 0, 1], pa.int32()),
            "start": pa.array([0.0, 5.0, 10.0, 0.0, 5.0], pa.float64()),
            "end": pa.array([5.0, 10.0, 15.0, 5.0, 10.0], pa.float64()),
            "text": ["a", "b", "c", "d", "e"],
            "alignments_json": pa.array([SPANS, None, None, None, None], pa.string()),
            "language": ["sv", "sv", "en", "en", "sv"],
            "clip_path": ["clips/aaa.mp4"] * 3 + ["clips/bbb.mp4"] * 2,
            # `main` space: row 2 unprojected (NULL x) → 4 built points.
            "px": pa.array([0.5, 1.5, None, 2.5, 3.5], pa.float64()),
            "py": pa.array([0.1, 0.2, None, 0.3, 0.4], pa.float64()),
            "pc": pa.array([0, 0, None, 1, 1], pa.int32()),
            # `alt` space: declared but never built (all NULL).
            "qx": pa.array([None] * 5, pa.float64()),
            "qy": pa.array([None] * 5, pa.float64()),
            "qc": pa.array([None] * 5, pa.int32()),
        }
    )
    lance.write_dataset(chunks, str(db / "chunks.lance"))


@pytest.fixture
def app_client(tmp_path: Path) -> tuple[FastAPI, TestClient]:
    root = tmp_path / "dbs"
    root.mkdir()
    descriptor_dir = tmp_path / "descriptors"
    descriptor_dir.mkdir()
    _write_chunks(root / "atl.lance")
    (descriptor_dir / "atl.json").write_text(json.dumps(DESCRIPTOR))
    settings = Settings(
        MEDIA_DB=root / "atl.lance",
        MEDIA_DB_ROOT=root,
        MEDIA_DESCRIPTOR_DIR=descriptor_dir,
    )
    app = create_viewer_app(settings)
    return app, TestClient(app)


@pytest.fixture
def client(app_client: tuple[FastAPI, TestClient]) -> TestClient:
    return app_client[1]


def _decode(content: bytes) -> pa.Table:
    return pa.ipc.open_stream(content).read_all()


class TestStatus:
    def test_default_space_reports_nonnull_builtness(self, client: TestClient) -> None:
        r = client.get("/api/atlas/status")
        assert r.status_code == 200
        assert r.json() == {
            "projected": True,
            "rows": 4,
            "space": "main",
            "spaces": {"main": True, "alt": False},
        }

    def test_unbuilt_space_reports_zero_rows(self, client: TestClient) -> None:
        r = client.get("/api/atlas/status", params={"space": "alt"})
        assert r.status_code == 200
        body = r.json()
        assert body["projected"] is False
        assert body["rows"] == 0

    def test_unknown_space_is_400(self, client: TestClient) -> None:
        assert client.get("/api/atlas/status", params={"space": "zzz"}).status_code == 400

    def test_unknown_dataset_param_is_404(self, client: TestClient) -> None:
        assert client.get("/api/atlas/status", params={"dataset": "nope"}).status_code == 404


class TestPoints:
    def test_arrow_stream_round_trip(self, client: TestClient) -> None:
        r = client.get("/api/atlas/points")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("application/vnd.apache.arrow.stream")
        table = _decode(r.content)
        assert table.num_rows == 4
        assert table.schema.names == [
            "x", "y", "speech_id", "chunk_id", "rowid", "doc", "cluster", "language",
        ]
        assert table.schema.field("x").type == pa.float16()
        assert table.schema.field("speech_id").type == pa.int32()
        assert table.schema.field("rowid").type == pa.int64()
        assert table.schema.field("doc").type == pa.dictionary(pa.int32(), pa.string())
        assert table.schema.field("language").type == pa.dictionary(pa.int32(), pa.string())
        assert [float(v) for v in table.column("x").to_pylist()] == [0.5, 1.5, 2.5, 3.5]
        assert table.column("chunk_id").to_pylist() == [0, 1, 0, 1]
        assert table.column("doc").to_pylist() == [DOC_A, DOC_A, DOC_B, DOC_B]
        assert table.column("cluster").to_pylist() == [0, 0, 1, 1]
        assert table.column("language").to_pylist() == ["sv", "sv", "en", "sv"]

    def test_doc_files_metadata_aligned_with_doc_dictionary(self, client: TestClient) -> None:
        table = _decode(client.get("/api/atlas/points").content)
        meta = table.schema.metadata
        assert meta[b"count"] == b"4"
        assert meta[b"space"] == b"main"
        doc_labels = table.column("doc").chunk(0).dictionary.to_pylist()
        doc_files = json.loads(meta[b"docFiles"])
        # One label per distinct doc, in dictionary order: the stem of the
        # first declared title field (clip_path).
        assert dict(zip(doc_labels, doc_files, strict=True)) == {DOC_A: "aaa", DOC_B: "bbb"}

    def test_unbuilt_space_is_400(self, client: TestClient) -> None:
        assert client.get("/api/atlas/points", params={"space": "alt"}).status_code == 400

    def test_points_memoized_per_dataset_space_version(
        self, app_client: tuple[FastAPI, TestClient]
    ) -> None:
        app, client = app_client
        first = client.get("/api/atlas/points").content
        second = client.get("/api/atlas/points").content
        assert first == second
        cache = app.state.resources.points_cache
        assert len(cache) == 1
        ((dataset_id, space, version),) = cache.keys()
        assert (dataset_id, space) == ("atl", "main")
        assert isinstance(version, int)


class TestChunkFetch:
    def test_single_chunk_hit(self, client: TestClient) -> None:
        r = client.get(f"/api/atlas/chunk/{DOC_A}/0/0")
        assert r.status_code == 200
        hit = r.json()
        assert hit["text"] == "a"
        assert hit["clip_path"] == "clips/aaa.mp4"
        assert hit["alignments"] == json.loads(SPANS)
        assert "alignments_json" not in hit

    def test_unknown_chunk_is_404(self, client: TestClient) -> None:
        assert client.get("/api/atlas/chunk/zzzz/9/9").status_code == 404

    def test_invalid_doc_key_is_400(self, client: TestClient) -> None:
        assert client.get("/api/atlas/chunk/abcdefgh/0/0").status_code == 400

    def test_rowids_round_trip_from_points(self, client: TestClient) -> None:
        table = _decode(client.get("/api/atlas/points").content)
        rowids = table.column("rowid").to_pylist()
        r = client.post("/api/atlas/chunks", json={"rowids": rowids[:2]})
        assert r.status_code == 200
        rows = r.json()
        assert [h["text"] for h in rows] == ["a", "b"]
        assert rows[0]["alignments"] == json.loads(SPANS)
        assert rows[1]["alignments"] == []

    def test_empty_selection_is_empty(self, client: TestClient) -> None:
        r = client.post("/api/atlas/chunks", json={"rowids": []})
        assert r.status_code == 200
        assert r.json() == []
