"""media_api core endpoints over a synthetic multi-table tmp dataset.

Every table and column name here deliberately differs from the old corpus
(``utterances``/``media_docs``/``framestore``, keys ``vid``/``seg``/``part``,
blobs ``payload``/``poster``/``img_blob``) — the routers must resolve them all
from the descriptor config file, proving the package carries no corpus
constants. Covers datasets/descriptor endpoints, Blob V2 Range serving,
thumbnails, per-chunk frames (frame_idx matched in Python), transcript
ordering, alignments, and the system endpoints.
"""

from __future__ import annotations

import json
from pathlib import Path

import lance
import pyarrow as pa
import pytest
from backend.core.config import Settings
from backend.media_api import create_media_app
from fastapi.testclient import TestClient
from lance import blob_array, blob_field

DOC = "abcd1234"  # matches the descriptor's ^[a-f0-9]{8}$ doc-key pattern
MEDIA = bytes(range(256)) * 8  # 2048 deterministic bytes
POSTER = b"\x89PNG-FAKE-POSTER"
FRAME0 = b"WEBP-FRAME-0"
FRAME1 = b"WEBP-FRAME-1"
SPANS = json.dumps([{"words": [{"text": "hej", "start": 0.0, "end": 0.4}]}])

DESCRIPTOR = {
    "identity": {
        "key_fields": ["vid", "seg", "part"],
        "doc_key": "vid",
        "doc_key_pattern": "^[a-f0-9]{8}$",
    },
    "document": {
        "table": "media_docs",
        "media_blob": "payload",
        "mime": "payload_mime",
        "thumbnail": "poster",
        "thumbnail_mime": "poster_mime",
    },
    "time": {"start": "t0", "end": "t1"},
    "display": {
        "title": ["file_name", "vid"],
        "body": "utterance",
        "metadata": [
            {"field": "file_name", "label": "File"},
            {"field": "lang", "label": "Language"},
        ],
    },
    "search": {
        "row_table": "utterances",
        "fts": {"table": "utterances", "column": "utterance"},
        "filterable": ["lang", "file_name"],
    },
    "capabilities": {
        "alignments": "utterances.word_spans",
        "frames": "framestore.img_blob",
        "captions": "framestore.scene_txt",
    },
}


def _write_tables(db: Path, media_file: Path) -> None:
    db.mkdir(parents=True)
    # Row table, written OUT of time order so /doc-transcript must sort by t0.
    utterances = pa.table(
        {
            "vid": [DOC, DOC, DOC],
            "seg": pa.array([0, 0, 1], pa.int32()),
            "part": pa.array([1, 0, 0], pa.int32()),
            "t0": pa.array([5.0, 0.0, 10.0], pa.float64()),
            "t1": pa.array([10.0, 5.0, 15.0], pa.float64()),
            "utterance": ["second", "first", "third"],
            "word_spans": pa.array([None, SPANS, None], pa.string()),
            "lang": ["sv", "sv", "en"],
            "file_name": ["x.mp4", "x.mp4", "x.mp4"],
            "vec": pa.array([[0.0] * 4] * 3, pa.list_(pa.float32(), 4)),
        }
    )
    lance.write_dataset(utterances, str(db / "utterances.lance"))

    docs_schema = pa.schema(
        [
            pa.field("vid", pa.string()),
            pa.field("file_name", pa.string()),
            pa.field("duration", pa.float64()),
            pa.field("lang", pa.string()),
            blob_field("payload"),
            pa.field("payload_mime", pa.string()),
            blob_field("poster"),
            pa.field("poster_mime", pa.string()),
        ]
    )
    docs = pa.table(
        {
            "vid": [DOC],
            "file_name": ["x.mp4"],
            "duration": [15.0],
            "lang": ["sv"],
            "payload": blob_array([media_file.resolve().as_uri()]),  # External: URI
            "payload_mime": ["video/mp4"],
            "poster": blob_array([POSTER]),  # Inline: bytes
            "poster_mime": ["image/png"],
        },
        schema=docs_schema,
    )
    lance.write_dataset(
        docs,
        str(db / "media_docs.lance"),
        data_storage_version="2.2",
        allow_external_blob_outside_bases=True,
    )

    frames_schema = pa.schema(
        [
            pa.field("vid", pa.string()),
            pa.field("seg", pa.int32()),
            pa.field("part", pa.int32()),
            pa.field("frame_idx", pa.int32()),
            blob_field("img_blob"),
            pa.field("img_mime", pa.string()),
            pa.field("scene_txt", pa.string()),
        ]
    )
    frames = pa.table(
        {
            "vid": [DOC, DOC, DOC],
            "seg": pa.array([0, 0, 0], pa.int32()),
            "part": pa.array([0, 0, 1], pa.int32()),
            "frame_idx": pa.array([0, 1, 0], pa.int32()),
            "img_blob": blob_array([FRAME0, FRAME1, b"WEBP-OTHER"]),
            "img_mime": ["image/webp", "image/webp", "image/webp"],
            "scene_txt": ["a red square", None, "a blue circle"],
        },
        schema=frames_schema,
    )
    lance.write_dataset(frames, str(db / "framestore.lance"), data_storage_version="2.2")


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    root = tmp_path / "dbs"
    root.mkdir()
    descriptor_dir = tmp_path / "descriptors"
    descriptor_dir.mkdir()
    media_file = tmp_path / "media.bin"
    media_file.write_bytes(MEDIA)
    _write_tables(root / "demo.lance", media_file)
    (descriptor_dir / "demo.json").write_text(json.dumps(DESCRIPTOR))
    settings = Settings(
        MEDIA_DB=root / "demo.lance",  # stem = the default dataset id
        MEDIA_DB_ROOT=root,
        MEDIA_DESCRIPTOR_DIR=descriptor_dir,
    )
    return TestClient(create_media_app(settings))


class TestDatasets:
    def test_lists_tables_and_available_capabilities(self, client: TestClient) -> None:
        r = client.get("/api/datasets")
        assert r.status_code == 200
        datasets = r.json()["datasets"]
        assert [d["id"] for d in datasets] == ["demo"]
        tables = datasets[0]["tables"]
        assert tables["utterances"]["row_count"] == 3
        assert tables["media_docs"]["n_columns"] == 8
        assert tables["framestore"]["version"] >= 1
        assert set(datasets[0]["capabilities"]) == {"alignments", "frames", "captions"}

    def test_descriptor_roundtrip(self, client: TestClient) -> None:
        r = client.get("/api/datasets/demo/descriptor")
        assert r.status_code == 200
        body = r.json()
        assert body["id"] == "demo"
        assert body["declared"]["identity"]["doc_key"] == "vid"
        assert body["declared"]["search"]["row_table"] == "utterances"
        assert "utterances" in body["tables"]

    def test_unknown_dataset_is_problem_404(self, client: TestClient) -> None:
        r = client.get("/api/datasets/missing/descriptor")
        assert r.status_code == 404
        assert r.headers["content-type"] == "application/problem+json"


class TestMediaRange:
    def test_full_get_streams_all_bytes(self, client: TestClient) -> None:
        r = client.get(f"/api/media/{DOC}")
        assert r.status_code == 200
        assert r.content == MEDIA
        assert r.headers["content-type"] == "video/mp4"  # declared mime column value
        assert r.headers["accept-ranges"] == "bytes"
        assert r.headers["content-length"] == str(len(MEDIA))

    def test_range_returns_206_slice(self, client: TestClient) -> None:
        r = client.get(f"/api/media/{DOC}", headers={"Range": "bytes=0-99"})
        assert r.status_code == 206
        assert r.content == MEDIA[:100]
        assert r.headers["content-range"] == f"bytes 0-99/{len(MEDIA)}"
        assert r.headers["content-length"] == "100"

    def test_suffix_range_counts_from_end(self, client: TestClient) -> None:
        r = client.get(f"/api/media/{DOC}", headers={"Range": "bytes=-64"})
        assert r.status_code == 206
        assert r.content == MEDIA[-64:]

    def test_unsatisfiable_range_is_416(self, client: TestClient) -> None:
        total = len(MEDIA)
        r = client.get(f"/api/media/{DOC}", headers={"Range": f"bytes={total + 10}-{total + 20}"})
        assert r.status_code == 416
        assert r.headers["content-range"] == f"bytes */{total}"

    def test_doc_key_pattern_comes_from_descriptor(self, client: TestClient) -> None:
        # 16-hex was the OLD corpus pattern; this descriptor whitelists 8-hex.
        assert client.get("/api/media/0123456789abcdef").status_code == 400

    def test_unknown_doc_is_404(self, client: TestClient) -> None:
        assert client.get("/api/media/ffffffff").status_code == 404

    def test_explicit_dataset_param(self, client: TestClient) -> None:
        assert client.get(f"/api/media/{DOC}", params={"dataset": "demo"}).status_code == 200
        assert client.get(f"/api/media/{DOC}", params={"dataset": "nope"}).status_code == 404


class TestThumbnail:
    def test_returns_inline_bytes_with_declared_mime(self, client: TestClient) -> None:
        r = client.get(f"/api/thumbnail/{DOC}")
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/png"
        assert r.content == POSTER

    def test_unknown_doc_is_404(self, client: TestClient) -> None:
        assert client.get("/api/thumbnail/ffffffff").status_code == 404


class TestChunkFrame:
    def test_default_returns_representative_frame(self, client: TestClient) -> None:
        r = client.get(f"/api/chunk-frame/{DOC}/0/0")
        assert r.status_code == 200
        assert r.content == FRAME0
        assert r.headers["content-type"] == "image/webp"  # discovered *_mime column

    def test_frame_idx_selects_a_specific_frame(self, client: TestClient) -> None:
        r = client.get(f"/api/chunk-frame/{DOC}/0/0", params={"frame_idx": 1})
        assert r.status_code == 200
        assert r.content == FRAME1

    def test_out_of_range_frame_idx_is_404(self, client: TestClient) -> None:
        assert client.get(f"/api/chunk-frame/{DOC}/0/0", params={"frame_idx": 5}).status_code == 404

    def test_missing_chunk_is_404(self, client: TestClient) -> None:
        assert client.get(f"/api/chunk-frame/{DOC}/9/9").status_code == 404


class TestTranscripts:
    def test_transcript_ordered_by_declared_start_time(self, client: TestClient) -> None:
        r = client.get(f"/api/doc-transcript/{DOC}")
        assert r.status_code == 200
        body = r.json()
        assert body["doc_id"] == DOC
        chunks = body["chunks"]
        assert [c["utterance"] for c in chunks] == ["first", "second", "third"]
        assert [c["t0"] for c in chunks] == [0.0, 5.0, 10.0]
        assert set(chunks[0]) == {"seg", "part", "t0", "t1", "utterance", "alignments"}
        assert chunks[0]["alignments"] == json.loads(SPANS)
        assert chunks[1]["alignments"] == []

    def test_unknown_doc_gives_empty_transcript(self, client: TestClient) -> None:
        r = client.get("/api/doc-transcript/ffffffff")
        assert r.status_code == 200
        assert r.json()["chunks"] == []

    def test_invalid_doc_key_is_400(self, client: TestClient) -> None:
        assert client.get("/api/doc-transcript/not-a-key").status_code == 400

    def test_chunk_alignments_parsed_from_capability_column(self, client: TestClient) -> None:
        r = client.get(f"/api/chunk-alignments/{DOC}/0/0")
        assert r.status_code == 200
        assert r.json() == {"alignments": json.loads(SPANS)}

    def test_missing_chunk_alignments_are_empty(self, client: TestClient) -> None:
        r = client.get(f"/api/chunk-alignments/{DOC}/9/9")
        assert r.status_code == 200
        assert r.json() == {"alignments": []}


class TestSystem:
    def test_health_reports_handle_facts(self, client: TestClient) -> None:
        r = client.get("/api/health")
        assert r.status_code == 200
        body = r.json()
        assert body["db"]["path"].endswith("demo.lance")
        assert body["db"]["chunks"] == 3
        assert body["db"]["documents"] == 1
        assert set(body["db"]["tables"]) == {"utterances", "media_docs", "framestore"}
        assert {"ok", "url", "error"} <= set(body["embed"])

    def test_columns_exclude_vector_blob_and_alignments(self, client: TestClient) -> None:
        r = client.get("/api/columns")
        assert r.status_code == 200
        by_name = {c["name"]: c["type"] for c in r.json()}
        assert by_name["utterance"] == "text"
        assert by_name["t0"] == "number"
        assert by_name["seg"] == "number"
        assert "vec" not in by_name  # vector column
        assert "word_spans" not in by_name  # declared alignments column

    def test_documents_gallery_projects_declared_fields(self, client: TestClient) -> None:
        r = client.get("/api/documents")
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 1
        assert body["page"] == 1
        doc = body["docs"][0]
        assert doc["vid"] == DOC
        assert doc["file_name"] == "x.mp4"
        assert doc["duration"] == 15.0
        assert "payload" not in doc  # blobs never enter the gallery projection

    def test_unknown_dataset_param_is_404(self, client: TestClient) -> None:
        assert client.get("/api/health", params={"dataset": "nope"}).status_code == 404


class TestAtlasWithoutSpaces:
    def test_status_is_404_when_no_spaces_declared(self, client: TestClient) -> None:
        assert client.get("/api/atlas/status").status_code == 404

    def test_atlas_chunk_attaches_capability_caption(self, client: TestClient) -> None:
        # /atlas/chunk works without atlas spaces (it is a row fetch); the
        # caption rides in from the `captions` capability's frame-0 row.
        r = client.get(f"/api/atlas/chunk/{DOC}/0/0")
        assert r.status_code == 200
        hit = r.json()
        assert hit["utterance"] == "first"
        assert hit["scene_txt"] == "a red square"
        assert hit["alignments"] == json.loads(SPANS)
        assert "word_spans" not in hit
