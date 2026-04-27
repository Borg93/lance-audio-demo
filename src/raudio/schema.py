"""PyArrow schemas for the two Lance tables that `raudio` produces.

* :data:`CHUNK_SCHEMA` — one row per :class:`AudioChunk`. Lightweight, contains
  the ``text`` column that the Tantivy FTS index is built on. Joined back to
  the source media via ``doc_id``.

* :data:`DOC_SCHEMA` — one row per source media file. Carries the **raw media
  bytes** (mp3 / mp4 / wav / …) as a Lance **blob v2** column (``media_blob``).
  Blob columns store large binary out-of-line and load lazily: scanning
  metadata never touches the audio bytes, and a search hit can `take_blobs`
  one document to stream back to the player.

The blob v2 encoding requires ``data_storage_version="2.2"`` on the
``documents`` dataset. The ``chunks`` dataset has no blob column and uses
LanceDB's default format.
"""

from __future__ import annotations

import pyarrow as pa
from lance import blob_field

# ──────────────────────────── Nested struct types ────────────────────────────

word_struct: pa.DataType = pa.struct(
    [
        pa.field("text", pa.string()),
        pa.field("start", pa.float32()),
        pa.field("end", pa.float32()),
        pa.field("score", pa.float32()),
    ]
)

alignment_struct: pa.DataType = pa.struct(
    [
        pa.field("start", pa.float32()),
        pa.field("end", pa.float32()),
        pa.field("text", pa.string()),
        pa.field("duration", pa.float32()),
        pa.field("score", pa.float32()),
        pa.field("words", pa.list_(word_struct)),
    ]
)


# ───────────────────────────── Chunk-centric schema ─────────────────────────
# One row per AudioChunk. No audio bytes here — those live on the documents
# table, keyed by `doc_id`. This keeps chunk scans (FTS, metadata queries)
# cheap regardless of how much media is stored in the DB.

CHUNK_SCHEMA: pa.Schema = pa.schema(
    [
        # Identity
        pa.field("doc_id", pa.string(), nullable=False),
        pa.field("speech_id", pa.int32(), nullable=False),
        pa.field("chunk_id", pa.int32(), nullable=False),
        # Document-level context (denormalised for filter-free retrieval).
        pa.field("audio_path", pa.string(), nullable=False),
        pa.field("sample_rate", pa.int32()),
        pa.field("audio_duration", pa.float32()),
        # Riksarkivet archival metadata (populated from a video_batcher CSV
        # at ingest time, keyed by bildid = audio_path stem).
        pa.field("referenskod", pa.string()),
        pa.field("namn", pa.string()),
        pa.field("bildid", pa.string()),
        pa.field("extraid", pa.string()),
        # Chunk payload — `text` is what the FTS index is built on.
        pa.field("start", pa.float32(), nullable=False),
        pa.field("end", pa.float32(), nullable=False),
        pa.field("duration", pa.float32()),
        pa.field("text", pa.string(), nullable=False),
        pa.field("audio_frames", pa.int64()),
        pa.field("num_logits", pa.int64()),
        pa.field("language", pa.string()),
        pa.field("language_prob", pa.float32()),
        # Word-level alignments that fall inside this chunk's [start, end),
        # stored as Lance JSONB (`pa.json_()`). Writers pass JSON strings;
        # Lance encodes them as compact binary internally. Readers get the
        # JSON text back — no binding-specific nested-struct decode. Lets
        # us add scalar/FTS indexes on JSON paths later if we ever want to.
        # Schema of each element:
        #   {"start": float, "end": float, "text": str, "duration": float,
        #    "score": float, "words": [{"text": str, "start": float,
        #    "end": float, "score": float}, ...]}
        pa.field("alignments_json", pa.json_()),
        pa.field("metadata", pa.string()),
    ]
)


# ───────────────────────────── Document-centric schema ──────────────────────
# One row per source media file.
#
# `media_blob` is a Lance Blob V2 External column — it stores URIs (not
# bytes), and a Python reader uses `ds.take_blobs("media_blob", ...)` to
# transparently fetch + range-read the underlying objects.
#   file:///abs/path/input/T0001417_00001.mp4    ← local dev
#   hf://buckets/you/raudio-videos/T0001417_00001.mp4
#   s3://bucket/videos/T0001417_00001.mp4
# Writes use `blob_array([uri, …])`. Requires data_storage_version="2.2".

DOC_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("doc_id", pa.string(), nullable=False),
        pa.field("audio_path", pa.string(), nullable=False),
        pa.field("sample_rate", pa.int32()),
        pa.field("duration", pa.float32()),
        # Riksarkivet archival metadata (mirrored from the CSV at ingest).
        pa.field("referenskod", pa.string()),
        pa.field("namn", pa.string()),
        pa.field("bildid", pa.string()),
        pa.field("extraid", pa.string()),
        # ISO 639-1 language code for the whole document (sv, en, de, fr, …).
        # Populated at ingest time either from --doc-language or inferred from
        # the alignments directory name (e.g. output/sv/alignments → "sv").
        pa.field("language", pa.string()),
        pa.field("media_mime", pa.string()),
        # Lance Blob V2 External: `blob_array(["file://…", "s3://…", "hf://…"])`.
        # Lance returns file-like BlobFile handles via `ds.take_blobs(...)`.
        blob_field("media_blob", nullable=True),
        # Lance Blob V2 (Inline mode for small objects). JPEG thumbnail bytes
        # live inside the main data file; no sidecar. Read via `take_blobs`.
        blob_field("thumbnail", nullable=True),
        pa.field("thumbnail_mime", pa.string()),
        pa.field("metadata", pa.string()),
    ]
)


#: Lance file format version required for Blob V2.
DOC_STORAGE_VERSION: str = "2.2"
