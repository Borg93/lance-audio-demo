"""The ``documents`` table write path through ingest (Blob V2 columns).

``test_backend_media`` exercises the *read* side over a hand-built documents
table; this covers the *write* side — ``ingest_many`` composing the media URI and
loading the thumbnail into Blob V2 columns. A bug here silently corrupts the
documents table, breaking media/thumbnail fetch for the whole DB with no error at
ingest time. The table is only written when a media param is supplied.
"""

from __future__ import annotations

from pathlib import Path

import lance
import pytest

from fakes import make_doc
from ratch.ingest.ingest import ingest_many

THUMB_BYTES = b"\xff\xd8\xffINGEST-THUMB"


def _doc(tmp_path) -> Path:
    db = tmp_path / "transcripts.lance"
    return db


def test_no_media_params_writes_no_documents_table(tmp_path) -> None:
    db = _doc(tmp_path)
    ingest_many(db, [make_doc("input/x.mp4", ["hello"])])  # no audio_root/base_uri/thumbnail
    assert not (db / "documents.lance").exists()


def test_thumbnail_dir_loads_inline_thumbnail_bytes(tmp_path) -> None:
    db = _doc(tmp_path)
    thumbs = tmp_path / "thumbs"
    thumbs.mkdir()
    (thumbs / "x.jpg").write_bytes(THUMB_BYTES)  # discovered by audio_path stem

    ingest_many(db, [make_doc("input/x.mp4", ["hello"])], thumbnail_dir=thumbs)

    ds = lance.dataset(str(db / "documents.lance"))
    assert ds.count_rows() == 1
    blob = ds.take_blobs("thumbnail", indices=[0])[0]
    with blob as f:
        assert f.read() == THUMB_BYTES  # inline Blob V2 round-trips through ingest


def test_missing_thumbnail_file_leaves_thumbnail_null(tmp_path) -> None:
    db = _doc(tmp_path)
    thumbs = tmp_path / "thumbs"
    thumbs.mkdir()  # empty — no x.jpg

    ingest_many(db, [make_doc("input/x.mp4", ["hello"])], thumbnail_dir=thumbs)

    row = lance.dataset(str(db / "documents.lance")).to_table(columns=["thumbnail_mime"]).to_pylist()
    assert row[0]["thumbnail_mime"] is None


def test_media_base_uri_composes_media_uri_and_mime(tmp_path) -> None:
    db = _doc(tmp_path)
    ingest_many(db, [make_doc("input/x.mp4", ["hello"])], media_base_uri="s3://bucket/videos")

    rows = (
        lance.dataset(str(db / "documents.lance"))
        .to_table(columns=["doc_id", "media_mime"])
        .to_pylist()
    )
    # media_mime is only set when a media URI was composed → proxy that the
    # External media_blob path ran (no network fetch needed to assert it).
    assert rows[0]["media_mime"] == "video/mp4"
    assert len(rows[0]["doc_id"]) == 16  # stable hex doc_id written


@pytest.mark.parametrize("param", ["audio_root", "media_base_uri", "thumbnail_dir"])
def test_any_media_param_triggers_documents_write(tmp_path, param: str) -> None:
    db = _doc(tmp_path)
    kwargs = {
        "audio_root": tmp_path,
        "media_base_uri": "s3://b/v",
        "thumbnail_dir": tmp_path,
    }
    ingest_many(db, [make_doc("input/x.mp4", ["hello"])], **{param: kwargs[param]})
    assert (db / "documents.lance").exists()
