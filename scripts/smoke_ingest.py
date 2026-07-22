"""P1.4 mixed-media smoke: SourceAdapter → agnostic documents table → blob probes.

Ingests the committed synthetic fixtures (mp4 + wav + png) through the
:class:`~ratch.ingest.sources.LocalDirSource` seam into a FRESH smoke DB via
``create_dataset`` (storage 2.2 + stable row ids + descriptor stamp): a
media-agnostic ``documents`` table (doc_id, source_uri, media_mime, media_blob)
plus a tiny synthetic ``chunks`` table (one probe sentence per doc) so a real
Ray column stage can run against this DB.

Evidence printed (per LANCE_MEDIA_MERGE §5.5 a bare marker doesn't count):
each doc's sniffed MIME + doc_id, the first/last-row 1-byte ``read_range``
probe byte counts, then ``SMOKE OK``.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import lance
import pyarrow as pa
from lance import blob_array, blob_field

from ratch.core.dataset import create_dataset, read_descriptor
from ratch.ingest.sources import LocalDirSource, sniff_mime

DOCUMENTS_SCHEMA = pa.schema(
    [
        pa.field("doc_id", pa.string(), nullable=False),
        pa.field("source_uri", pa.string(), nullable=False),
        pa.field("media_mime", pa.string(), nullable=False),
        blob_field("media_blob"),
    ]
)
CHUNKS_SCHEMA = pa.schema(
    [
        pa.field("doc_id", pa.string(), nullable=False),
        pa.field("speech_id", pa.int32(), nullable=False),
        pa.field("chunk_id", pa.int32(), nullable=False),
        pa.field("text", pa.string()),
    ]
)
DESCRIPTOR = {
    "identity": {"key_fields": ["doc_id", "speech_id", "chunk_id"]},
    "document": {"table": "documents", "media_blob": "media_blob", "mime": "media_mime"},
    "display": {"title": ["source_uri", "doc_id"], "body": "text"},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", default="tests/fixtures/media")
    parser.add_argument("--db", default="smoke.lance")
    args = parser.parse_args()

    db = Path(args.db)
    if db.exists():
        shutil.rmtree(db)
    db.mkdir(parents=True)

    objects = list(LocalDirSource(Path(args.fixtures)).iter_objects())
    if not objects:
        raise SystemExit(f"no fixtures under {args.fixtures}")

    doc_ids = [hashlib.sha1(o.uri.encode()).hexdigest()[:16] for o in objects]
    mimes = [sniff_mime(o.data, uri=o.uri) for o in objects]

    create_dataset(
        db / "documents.lance",
        DOCUMENTS_SCHEMA,
        descriptor=DESCRIPTOR,
        data=pa.table(
            {
                "doc_id": pa.array(doc_ids, pa.string()),
                "source_uri": pa.array([o.uri for o in objects], pa.string()),
                "media_mime": pa.array(mimes, pa.string()),
                "media_blob": blob_array([o.data for o in objects]),
            },
            schema=DOCUMENTS_SCHEMA,
        ),
    )
    create_dataset(
        db / "chunks.lance",
        CHUNKS_SCHEMA,
        descriptor=DESCRIPTOR,
        data=pa.table(
            {
                "doc_id": pa.array(doc_ids, pa.string()),
                "speech_id": pa.array([0] * len(objects), pa.int32()),
                "chunk_id": pa.array([0] * len(objects), pa.int32()),
                "text": pa.array([f"synthetic probe chunk for {m} document" for m in mimes]),
            },
            schema=CHUNKS_SCHEMA,
        ),
    )

    for doc_id, mime, obj in zip(doc_ids, mimes, objects, strict=True):
        print(f"  doc {doc_id}  mime={mime}  uri={Path(obj.uri).name}")

    ds = lance.dataset(str(db / "documents.lance"))
    rows = ds.count_rows()
    probe_counts = []
    for row in sorted({0, rows - 1}):
        [handle] = ds.take_blobs("media_blob", indices=[row])
        with handle as blob:
            data = blob.read_range(0, 1)
            probe_counts.append(len(data))
    print(f"  read_range probes (first,last row): {probe_counts} byte(s) each, all resolving")

    assert len(set(mimes)) >= 3, f"expected >=3 distinct MIME types, got {mimes}"
    assert read_descriptor(db / "documents.lance") == DESCRIPTOR
    print(f"  descriptor stamp verified on {db}/documents.lance")
    print("SMOKE OK")


if __name__ == "__main__":
    main()
