"""External blob-v2 → managed materialization (the lance-ns re-wrap)."""

from __future__ import annotations

from pathlib import Path

import lance
import pyarrow as pa
from lance import blob_array, blob_field

from rmedia.ingest.materialize import materialize_blobs


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


def test_materialize_noop_on_tables_without_blobs(tmp_path: Path) -> None:
    db = tmp_path / "db.lance"
    plain = db / "chunks.lance"
    lance.write_dataset(pa.table({"doc_id": ["x"]}), str(plain), data_storage_version="2.2")
    assert materialize_blobs(db, table="chunks") == {}
