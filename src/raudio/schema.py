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
        # ── Multimodal embeddings (Phase 1+2) ─────────────────────────────
        # `text_embedding` is the Qwen3-VL-Embedding-8B vector for `text`,
        # MRL-truncated to 1024 dims and L2-normalized. Lance vector index
        # (IVF_PQ, cosine) is built on this column by `raudio embed-chunks`.
        # Nullable so existing ingest paths produce all-null vectors that
        # `embed-chunks` then populates batch-by-batch.
        pa.field("text_embedding", pa.list_(pa.float32(), 1024), nullable=True),
        # ── One representative video frame per chunk (Phase 2) ───────────
        # Captured at `chunk.start` via ffmpeg, JPEG ~50–80 KB. Stored Blob
        # V2 Inline (auto-routes <64 KB into the main data page). Read via
        # `chunks_ds.take_blobs("frame_blob", indices=[idx])`.
        blob_field("frame_blob", nullable=True),
        pa.field("frame_mime", pa.string()),
        pa.field("frame_width", pa.int32()),
        pa.field("frame_height", pa.int32()),
        # Qwen3-VL-Embedding-8B vector for the frame, same MRL=1024 space
        # as text_embedding so cross-modal search is a direct cosine compare.
        pa.field("frame_embedding", pa.list_(pa.float32(), 1024), nullable=True),
    ]
)


#: Lance file format version required for Blob V2 columns on `chunks`.
#: Bumped from the LanceDB default once we added `frame_blob` (Phase 2).
CHUNK_STORAGE_VERSION: str = "2.2"


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


# ─────────────────────── Chunk-frames table (Phase 2 v2) ────────────────────
# A NEW separate table, keyed by (doc_id, speech_id, chunk_id), built via
# append-only writes during `extract-chunk-frames`. We keep it separate from
# `chunks` because Lance 4.0's `merge_insert` crashes on wide schemas with
# multiple extension types when filling blob columns post-hoc:
#   `Invalid user input: there were more fields in the schema than provided
#    column indices / infos` (decoder.rs:438) — confirmed at row counts 1,
#   100, and 145k. Lance docs explicitly recommend a separate table + append
#   for "data evolution"-style workloads (Lance file format 2.2 docs):
#   "Adding a column with backfilled data writes new data alongside existing
#    files; the original data is never touched."
#
# The frame_embedding column is added later via `dataset.add_columns(...)`
# from the embed-chunk-frames step — also append-only at the column level,
# bypasses the merge_insert join entirely.
CHUNK_FRAMES_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("doc_id", pa.string(), nullable=False),
        pa.field("speech_id", pa.int32(), nullable=False),
        pa.field("chunk_id", pa.int32(), nullable=False),
        # ~50 KB JPEG → goes into Blob V2 Inline tier (≤64 KB threshold)
        # per the file format 2.2 spec. Read via `ds.take_blobs("frame_blob")`.
        blob_field("frame_blob", nullable=False),
        pa.field("frame_mime", pa.string(), nullable=False),
        pa.field("frame_width", pa.int32(), nullable=False),
        pa.field("frame_height", pa.int32(), nullable=False),
    ]
)


#: Lance file format version required for chunk_frames (blob_field).
CHUNK_FRAMES_STORAGE_VERSION: str = "2.2"
