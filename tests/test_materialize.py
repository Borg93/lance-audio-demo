"""External blob-v2 → managed materialization (the lance-ns re-wrap)."""

from __future__ import annotations

from pathlib import Path

import lance
import pyarrow as pa
from lance import blob_array, blob_field

from ratch.ingest.materialize import materialize_blobs


def _external_blob_documents(db: Path, payload: bytes) -> tuple[Path, Path]:
    """A documents.lance whose ``media_blob`` is an EXTERNAL file:// pointer."""
    source = db.parent / "payload.mp4"
    source.write_bytes(payload)
    table = pa.table(
        {"doc_id": pa.array(["abc0123456789abc"]), "media_blob": blob_array([f"file://{source}"])},
        schema=pa.schema([pa.field("doc_id", pa.string()), blob_field("media_blob")]),
    )
    documents = db / "documents.lance"
    lance.write_dataset(
        table,
        str(documents),
        data_storage_version="2.2",
        enable_stable_row_ids=True,
        allow_external_blob_outside_bases=True,
    )
    return documents, source


def test_materialize_flips_external_blob_to_managed(tmp_path: Path) -> None:
    payload = b"FTYP-fake-mp4-bytes-" * 500
    db = tmp_path / "db.lance"
    documents, source = _external_blob_documents(db, payload)

    # BEFORE: external descriptor (kind 3), bytes live only in the source file.
    before = lance.dataset(str(documents)).to_table(columns=["media_blob"]).column("media_blob")[0].as_py()
    assert before["kind"] == 3

    stats = materialize_blobs(db, table="documents")
    assert stats["media_blob"] == {"rows": 1, "bytes": len(payload)}

    # AFTER: managed (kind != external) AND resolvable with the source DELETED —
    # the whole point: the dataset is now self-contained (copies to S3 intact).
    source.unlink()
    ds = lance.dataset(str(documents))
    after = ds.to_table(columns=["media_blob"]).column("media_blob")[0].as_py()
    assert after["kind"] != 3
    blob = ds.take_blobs("media_blob", indices=[0])[0]
    assert blob.read_range(0, blob.size()) == payload


def test_materialize_preserves_null_blobs(tmp_path: Path) -> None:
    """A null blob row (e.g. a doc with no thumbnail) must survive — read_blobs
    skips nulls, so a naive rewrite would crash on a length mismatch."""
    db = tmp_path / "db.lance"
    documents = db / "documents.lance"
    payload = b"THUMB-bytes-" * 40
    table = pa.table(
        {
            "doc_id": pa.array(["a", "b", "c"]),
            # middle row has NO thumbnail (null) — the common shape
            "thumbnail": blob_array([payload, None, payload]),
        },
        schema=pa.schema([pa.field("doc_id", pa.string()), blob_field("thumbnail")]),
    )
    lance.write_dataset(table, str(documents), data_storage_version="2.2", enable_stable_row_ids=True)

    stats = materialize_blobs(db, table="documents")
    # only the 2 non-null rows are counted as materialized
    assert stats["thumbnail"] == {"rows": 2, "bytes": 2 * len(payload)}

    ds = lance.dataset(str(documents))
    assert ds.count_rows() == 3  # row count preserved, no crash
    desc = ds.to_table(columns=["thumbnail"]).column("thumbnail").combine_chunks()
    assert [s > 0 for s in desc.field("size").to_pylist()] == [True, False, True]
    # the present rows still resolve to their bytes, managed
    blobs = ds.take_blobs("thumbnail", indices=[0, 2])
    assert all(b.read_range(0, b.size()) == payload for b in blobs)


def test_materialize_noop_on_tables_without_blobs(tmp_path: Path) -> None:
    db = tmp_path / "db.lance"
    plain = db / "chunks.lance"
    lance.write_dataset(pa.table({"doc_id": ["x"]}), str(plain), data_storage_version="2.2")
    assert materialize_blobs(db, table="chunks") == {}
