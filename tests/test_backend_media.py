"""Media endpoints over a synthetic documents table with real Blob V2 columns.

Exercises the ``take_blobs`` + HTTP-Range streaming path (thumbnail, ranged
media, chunk-frame 404) without the local dataset the smoke tests need: the
documents table is built in tmp with an External ``media_blob`` URI and an Inline
``thumbnail``, mirroring how ingest writes them.
"""

from __future__ import annotations

from pathlib import Path

import lance
import pyarrow as pa
import pytest
from backend import create_app
from fastapi.testclient import TestClient
from lance import blob_array

from fakes import make_doc
from rmedia.ingest.ingest import ingest_many
from rmedia.modalities.av.frames import ExtractedFrame, write_chunk_frames
from rmedia.model.schema import DOC_SCHEMA, DOC_STORAGE_VERSION

DOC_ID = "0123456789abcdef"
MEDIA_BYTES = bytes(range(256)) * 16  # 4096 deterministic bytes
THUMB_BYTES = b"\xff\xd8\xffFAKE-JPEG-THUMBNAIL"
FRAME_BYTES = b"\xff\xd8\xffCHUNK-FRAME-BYTES"


def _write_documents(db: Path, media_path: Path) -> None:
    """Replace the documents table with a single blob-bearing row for DOC_ID."""
    values: dict[str, object] = {
        "doc_id": DOC_ID,
        "audio_path": "input/x.mp4",
        "sample_rate": 16000,
        "duration": 1.0,
        "referenskod": "R1",
        "namn": "Talare",
        "bildid": None,
        "extraid": None,
        "language": "sv",
        "media_mime": "video/mp4",
        "media_blob": media_path.resolve().as_uri(),  # External: URI string
        "thumbnail": THUMB_BYTES,  # Inline: bytes
        "thumbnail_mime": "image/jpeg",
        "metadata": None,
    }
    arrays: list[pa.Array] = []
    for field in DOC_SCHEMA:
        if field.name in ("media_blob", "thumbnail"):
            arrays.append(blob_array([values[field.name]]))
        else:
            arrays.append(pa.array([values[field.name]], type=field.type))
    table = pa.Table.from_arrays(arrays, schema=DOC_SCHEMA)
    lance.write_dataset(
        table,
        str(db / "documents.lance"),
        mode="overwrite",
        data_storage_version=DOC_STORAGE_VERSION,
        allow_external_blob_outside_bases=True,
    )


@pytest.fixture
def client(tmp_path):
    db = tmp_path / "transcripts.lance"
    ingest_many(db, [make_doc("input/x.mp4", ["hello world"])])
    media_path = tmp_path / "media.bin"
    media_path.write_bytes(MEDIA_BYTES)
    _write_documents(db, media_path)
    return TestClient(create_app(db))


class TestThumbnail:
    def test_returns_inline_jpeg(self, client) -> None:
        r = client.get(f"/api/thumbnail/{DOC_ID}")
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/jpeg"
        assert r.content == THUMB_BYTES

    def test_unknown_doc_id_is_404(self, client) -> None:
        assert client.get("/api/thumbnail/ffffffffffffffff").status_code == 404

    def test_invalid_doc_id_is_400(self, client) -> None:
        assert client.get("/api/thumbnail/not-hex").status_code == 400


class TestMediaRange:
    def test_full_get_streams_all_bytes(self, client) -> None:
        r = client.get(f"/api/media/{DOC_ID}")
        assert r.status_code == 200
        assert r.content == MEDIA_BYTES
        assert r.headers["accept-ranges"] == "bytes"
        assert r.headers["content-length"] == str(len(MEDIA_BYTES))

    def test_range_returns_206_slice(self, client) -> None:
        r = client.get(f"/api/media/{DOC_ID}", headers={"Range": "bytes=0-1023"})
        assert r.status_code == 206
        assert r.content == MEDIA_BYTES[:1024]
        assert r.headers["content-range"] == f"bytes 0-1023/{len(MEDIA_BYTES)}"
        assert r.headers["content-length"] == "1024"

    def test_suffix_range_counts_from_end(self, client) -> None:
        r = client.get(f"/api/media/{DOC_ID}", headers={"Range": "bytes=-100"})
        assert r.status_code == 206
        assert r.content == MEDIA_BYTES[-100:]

    def test_unsatisfiable_range_is_416(self, client) -> None:
        total = len(MEDIA_BYTES)
        r = client.get(f"/api/media/{DOC_ID}", headers={"Range": f"bytes={total + 10}-{total + 20}"})
        assert r.status_code == 416
        assert r.headers["content-range"] == f"bytes */{total}"

    def test_unknown_doc_id_is_404(self, client) -> None:
        assert client.get("/api/media/ffffffffffffffff").status_code == 404

    def test_invalid_doc_id_is_400(self, client) -> None:
        assert client.get("/api/media/zzz").status_code == 400


@pytest.fixture
def client_with_frame(tmp_path):
    db = tmp_path / "transcripts.lance"
    ingest_many(db, [make_doc("input/x.mp4", ["hello world"])])
    media_path = tmp_path / "media.bin"
    media_path.write_bytes(MEDIA_BYTES)
    _write_documents(db, media_path)
    write_chunk_frames(
        db / "chunk_frames.lance",
        [
            ExtractedFrame(
                doc_id=DOC_ID,
                speech_id=0,
                chunk_id=0,
                time_sec=0.0,
                jpeg_bytes=FRAME_BYTES,
                width=8,
                height=8,
            )
        ],
        create=True,
    )
    return TestClient(create_app(db))


class TestChunkFrame:
    def test_no_frames_table_is_404(self, client) -> None:
        # No chunk_frames extracted in this fixture → 404, never 500.
        assert client.get(f"/api/chunk-frame/{DOC_ID}/0/0").status_code == 404

    def test_returns_inline_frame_bytes(self, client_with_frame) -> None:
        r = client_with_frame.get(f"/api/chunk-frame/{DOC_ID}/0/0")
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/jpeg"
        assert r.content == FRAME_BYTES

    def test_missing_chunk_is_404(self, client_with_frame) -> None:
        assert client_with_frame.get(f"/api/chunk-frame/{DOC_ID}/9/9").status_code == 404


FRAME_0 = b"\xff\xd8\xffFRAME-IDX-0"
FRAME_1 = b"\xff\xd8\xffFRAME-IDX-1"


@pytest.fixture
def client_with_multiframe(tmp_path):
    db = tmp_path / "transcripts.lance"
    ingest_many(db, [make_doc("input/x.mp4", ["hello world"])])
    media_path = tmp_path / "media.bin"
    media_path.write_bytes(MEDIA_BYTES)
    _write_documents(db, media_path)
    write_chunk_frames(
        db / "chunk_frames.lance",
        [
            ExtractedFrame(
                doc_id=DOC_ID, speech_id=0, chunk_id=0, frame_idx=0,
                time_sec=0.0, jpeg_bytes=FRAME_0, width=8, height=8,
            ),
            ExtractedFrame(
                doc_id=DOC_ID, speech_id=0, chunk_id=0, frame_idx=1,
                time_sec=2.0, jpeg_bytes=FRAME_1, width=8, height=8,
            ),
        ],
        create=True,
    )
    return TestClient(create_app(db))


class TestMultiFrame:
    def test_default_returns_frame_0(self, client_with_multiframe) -> None:
        r = client_with_multiframe.get(f"/api/chunk-frame/{DOC_ID}/0/0")
        assert r.status_code == 200
        assert r.content == FRAME_0

    def test_frame_idx_selects_a_specific_frame(self, client_with_multiframe) -> None:
        r = client_with_multiframe.get(f"/api/chunk-frame/{DOC_ID}/0/0", params={"frame_idx": 1})
        assert r.status_code == 200
        assert r.content == FRAME_1

    def test_out_of_range_frame_idx_is_404(self, client_with_multiframe) -> None:
        r = client_with_multiframe.get(f"/api/chunk-frame/{DOC_ID}/0/0", params={"frame_idx": 5})
        assert r.status_code == 404
