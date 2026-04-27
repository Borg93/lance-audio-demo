"""Ingest easytranscriber AudioMetadata JSON into a Lance database.

Two tables are written into the same LanceDB directory:

* ``chunks`` — one row per :class:`AudioChunk`, with ``text`` indexed for
  Tantivy FTS. Lightweight; safe to scan in full.
* ``documents`` — one row per source media file, with the **raw bytes** stored
  in a Lance **blob v2** column (``media_blob``) so the table is fully
  self-contained. Written with ``data_storage_version="2.2"``. The documents
  table is only produced when ``audio_root`` is supplied.

Reading the blob back (Python):

    import lance
    ds = lance.dataset("./transcripts.lance/documents.lance")
    blob = ds.take_blobs("media_blob", indices=[0])[0]
    with blob as f: ...
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Iterable, Iterator

import lance
import lancedb
import msgspec
import pyarrow as pa
from lance import blob_array

from .audio import compose_media_uri, guess_mime, resolve_source
from .datamodel import AlignmentSegment, AudioMetadata
from .schema import (
    CHUNK_SCHEMA,
    CHUNK_STORAGE_VERSION,
    DOC_SCHEMA,
    DOC_STORAGE_VERSION,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────── Helpers ────────────────────────────────────


def _doc_id(audio_path: str) -> str:
    """Short, deterministic id derived from the audio path."""
    return hashlib.sha1(audio_path.encode("utf-8")).hexdigest()[:16]


# Archival-metadata columns populated from the video_batcher CSV.
METADATA_COLUMNS: tuple[str, ...] = ("referenskod", "namn", "bildid", "extraid")


def load_metadata_csv(csv_path: str | Path) -> dict[str, dict[str, str]]:
    """Load a ``video_batcher`` CSV into ``{bildid: {col: value}}``.

    The CSV is semicolon-separated with columns
    ``referenskod;namn;extraid;bildid``. Returned dicts carry the four
    :data:`METADATA_COLUMNS` fields and nothing else.
    """
    out: dict[str, dict[str, str]] = {}
    with Path(csv_path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter=";"):
            bildid = (row.get("bildid") or "").strip()
            if not bildid:
                continue
            out[bildid] = {col: (row.get(col) or "").strip() for col in METADATA_COLUMNS}
    return out


def _metadata_for(audio_path: str, index: dict[str, dict[str, str]] | None) -> dict[str, str | None]:
    """Look up metadata for one transcript by its audio file stem (= bildid)."""
    if not index:
        return {col: None for col in METADATA_COLUMNS}
    stem = Path(audio_path).stem
    found = index.get(stem)
    if not found:
        return {col: None for col in METADATA_COLUMNS}
    return {col: (found.get(col) or None) for col in METADATA_COLUMNS}


def _pick_alignments(
    alignments: list[AlignmentSegment], start: float, end: float
) -> list[dict[str, Any]]:
    """Alignments whose ``[a.start, a.end]`` is contained in ``[start, end)``."""
    out: list[dict[str, Any]] = []
    for a in alignments:
        if a.start is None or a.end is None:
            continue
        if a.start >= start and a.end <= end:
            out.append(
                {
                    "start": float(a.start),
                    "end": float(a.end),
                    "text": a.text or "",
                    "duration": a.duration,
                    "score": a.score,
                    "words": [
                        {
                            "text": w.text,
                            "start": float(w.start),
                            "end": float(w.end),
                            "score": w.score,
                        }
                        for w in a.words
                    ],
                }
            )
    return out


def load_transcript(path: str | Path) -> AudioMetadata:
    """Parse one easytranscriber JSON into an :class:`AudioMetadata`."""
    raw = Path(path).read_bytes()
    return msgspec.json.decode(raw, type=AudioMetadata)


# ─────────────────────────────── Row builders ───────────────────────────────


def flatten_chunks(
    doc: AudioMetadata,
    *,
    metadata_index: dict[str, dict[str, str]] | None = None,
    doc_language: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield one chunk row per :class:`AudioChunk`, matching :data:`CHUNK_SCHEMA`."""
    audio_path = doc.audio_path
    doc_id = _doc_id(audio_path)
    extra = _metadata_for(audio_path, metadata_index)

    for speech in doc.speeches or []:
        alignments = speech.alignments or []
        for chunk_id, ch in enumerate(speech.chunks or []):
            yield {
                "doc_id": doc_id,
                "speech_id": int(speech.speech_id) if speech.speech_id is not None else 0,
                "chunk_id": chunk_id,
                "audio_path": audio_path,
                "sample_rate": doc.sample_rate,
                "audio_duration": doc.duration,
                "referenskod": extra["referenskod"],
                "namn": extra["namn"],
                "bildid": extra["bildid"],
                "extraid": extra["extraid"],
                "start": float(ch.start),
                "end": float(ch.end),
                "duration": ch.duration,
                "text": ch.text or "",
                "audio_frames": ch.audio_frames,
                "num_logits": ch.num_logits,
                # Per-chunk language from Whisper auto-detect if present,
                # else the document-level language stamp.
                "language": ch.language or doc_language,
                "language_prob": ch.language_prob,
                "alignments_json": json.dumps(_pick_alignments(alignments, ch.start, ch.end)),
                "metadata": json.dumps(speech.metadata or {}),
            }


def _document_row(
    doc: AudioMetadata,
    *,
    audio_root: str | Path | None,
    media_base_uri: str | None,
    metadata_index: dict[str, dict[str, str]] | None = None,
    thumbnail_dir: str | Path | None = None,
    doc_language: str | None = None,
) -> dict[str, Any]:
    """Build one document row.

    ``media_blob`` is a Blob V2 External URI (string); ``thumbnail`` is Blob V2
    Inline bytes (auto-routes into the main data page when <64 KB).
    """
    audio_path = doc.audio_path
    source = resolve_source(audio_path, audio_root)
    extra = _metadata_for(audio_path, metadata_index)

    media_uri = compose_media_uri(
        audio_path=audio_path,
        source_path=source,
        base_uri=media_base_uri,
    )
    media_mime = guess_mime(audio_path) if media_uri else None

    thumb_bytes: bytes | None = None
    thumb_mime: str | None = None
    if thumbnail_dir is not None:
        candidate = Path(thumbnail_dir) / f"{Path(audio_path).stem}.jpg"
        if candidate.exists():
            thumb_bytes = candidate.read_bytes()
            thumb_mime = "image/jpeg"

    return {
        "doc_id": _doc_id(audio_path),
        "audio_path": audio_path,
        "sample_rate": doc.sample_rate,
        "duration": doc.duration,
        "referenskod": extra["referenskod"],
        "namn": extra["namn"],
        "bildid": extra["bildid"],
        "extraid": extra["extraid"],
        "language": doc_language,
        "media_mime": media_mime,
        "media_blob": media_uri,          # string URI — Blob V2 External
        "thumbnail": thumb_bytes,         # bytes — Blob V2 Inline
        "thumbnail_mime": thumb_mime,
        "metadata": json.dumps({}),
    }


# ─────────────────────────────── Ingestion ──────────────────────────────────


def _write_documents_table(
    db_path: str | Path,
    rows: list[dict[str, Any]],
    *,
    mode: str = "append",
) -> None:
    """Write the ``documents`` table with Blob V2 columns.

    - ``media_blob`` is a Blob V2 External URI: we wrap the string URIs with
      ``blob_array(...)``. Lance writes them as Arrow blob descriptors and,
      on read, ``ds.take_blobs(...)`` returns lazy ``BlobFile`` handles that
      range-read the underlying URI.
    - ``thumbnail`` is Blob V2 Inline bytes: same ``blob_array(...)`` wrapper;
      Lance auto-routes values under 64 KB into the main data page.
    """
    if not rows:
        return

    cols: dict[str, Any] = {name: [] for name in DOC_SCHEMA.names}
    for r in rows:
        for k in cols:
            cols[k].append(r[k])

    media_col = blob_array(cols.pop("media_blob"))
    thumb_col = blob_array(cols.pop("thumbnail"))
    arrays: list[pa.Array] = []
    for field in DOC_SCHEMA:
        if field.name == "media_blob":
            arrays.append(media_col)
        elif field.name == "thumbnail":
            arrays.append(thumb_col)
        else:
            arrays.append(pa.array(cols[field.name], type=field.type))
    table = pa.Table.from_arrays(arrays, schema=DOC_SCHEMA)

    dataset_path = str(Path(db_path) / "documents.lance")
    lance.write_dataset(
        table,
        dataset_path,
        mode=mode,
        data_storage_version=DOC_STORAGE_VERSION,
        # Required because our URIs (file://…, hf://…) don't map to registered
        # base paths yet. TODO: register base paths for true multi-base layout
        # lifecycle governance per the blog.
        allow_external_blob_outside_bases=True,
    )


def _build_chunks_table(rows: list[dict[str, Any]]) -> pa.Table:
    """Materialise chunk rows into a pa.Table that matches CHUNK_SCHEMA.

    Critical for `alignments_json`: from a list[dict] PyArrow infers
    `large_string`, but the table column is `pa.json_()` (extension type
    backed by large_string). Lance's append refuses the schema mismatch.
    Building each column with its declared field type promotes the JSON
    strings into the extension array.

    Also critical for Phase 2: ``frame_blob`` is a Lance ``blob_field``
    extension that cannot be constructed via ``pa.array(values, type=...)``.
    We wrap that column with :func:`lance.blob_array` (mirrors
    :func:`_write_documents_table`). On the create-from-empty path every
    chunk's frame is ``None``; ``embed-chunk-frames`` populates it later.
    """
    # Names of all blob-extension columns that need ``blob_array`` wrapping.
    blob_cols = {"frame_blob"}

    cols: dict[str, list[Any]] = {name: [] for name in CHUNK_SCHEMA.names}
    for r in rows:
        for k in cols:
            cols[k].append(r.get(k))

    arrays: list[pa.Array] = []
    for f in CHUNK_SCHEMA:
        if f.name in blob_cols:
            arrays.append(blob_array(cols[f.name]))
        else:
            arrays.append(pa.array(cols[f.name], type=f.type))
    return pa.Table.from_arrays(arrays, schema=CHUNK_SCHEMA)


def ingest_many(
    db_path: str | Path,
    docs: Iterable[AudioMetadata],
    *,
    audio_root: str | Path | None = None,
    media_base_uri: str | None = None,
    table_name: str = "chunks",
    metadata_csv: str | Path | None = None,
    thumbnail_dir: str | Path | None = None,
    fts_language: str = "English",
    doc_language: str | None = None,
) -> "lancedb.table.Table":
    """Ingest many transcripts in one pass.

    Writes the ``chunks`` table (always) and the ``documents`` table (when
    ``audio_root`` is supplied). Builds the FTS + scalar indexes once at the
    end.

    Parameters
    ----------
    db_path
        Filesystem path to the Lance database directory.
    docs
        Iterable of parsed :class:`AudioMetadata`.
    audio_root
        If set, each document's source media is slurped and written into the
        ``documents`` table as a Lance blob-v2 column. Pass ``None`` to skip
        the documents table entirely.
    table_name
        Name of the chunks table. Defaults to ``"chunks"``.
    metadata_csv
        Optional ``video_batcher`` CSV (``referenskod;namn;extraid;bildid``).
        Rows are joined to transcripts by ``bildid == Path(audio_path).stem``;
        matched values populate the ``referenskod`` / ``namn`` / ``bildid`` /
        ``extraid`` columns in both tables.
    """
    db = lancedb.connect(str(db_path))

    metadata_index = load_metadata_csv(metadata_csv) if metadata_csv else None
    if metadata_csv:
        logger.info("Loaded metadata for %d bildid(s) from %s", len(metadata_index or {}), metadata_csv)

    docs_list = list(docs)
    chunk_rows: list[dict[str, Any]] = []
    for doc in docs_list:
        chunk_rows.extend(
            flatten_chunks(doc, metadata_index=metadata_index, doc_language=doc_language)
        )

    if not chunk_rows:
        raise ValueError("No chunks produced from any of the supplied transcripts.")

    chunks_table = _build_chunks_table(chunk_rows)
    table_exists = table_name in db.table_names()
    if table_exists:
        table = db.open_table(table_name)
        table.add(chunks_table)
    else:
        # Phase 2 added a Blob V2 Inline column (`frame_blob`) to
        # CHUNK_SCHEMA, which requires Lance file format 2.2. lancedb's
        # `create_table` doesn't expose the storage version directly, so
        # we write the dataset via `lance.write_dataset(...)` (mirroring
        # `_write_documents_table`) and re-open it through lancedb.
        chunks_path = str(Path(db_path) / f"{table_name}.lance")
        lance.write_dataset(
            chunks_table,
            chunks_path,
            mode="create",
            data_storage_version=CHUNK_STORAGE_VERSION,
            # Blob V2 Inline (frame_blob) is on a chunks-relative path,
            # but external blob columns may show up later (e.g. if we
            # ever spill very large frames). Match the documents-table
            # behaviour for safety.
            allow_external_blob_outside_bases=True,
        )
        table = db.open_table(table_name)

    # `with_position=True` is required for phrase queries.
    # `remove_stop_words=False` keeps "of", "the", etc. in the index so phrases
    # like `"spring of hope"` match verbatim instead of silently dropping to
    # `"spring hope"` tokens.
    # `language=…` picks the stemmer and stop-word list. For Swedish text the
    # English stemmer can't reduce forms like `ministern`/`vägen`/`ansåg` to a
    # shared stem, so those queries return zero hits — use "Swedish" to fix.
    table.create_fts_index(
        "text",
        replace=True,
        with_position=True,
        remove_stop_words=False,
        language=fts_language,
    )
    for col in ("doc_id", "audio_path"):
        try:
            table.create_scalar_index(col, index_type="BTREE", replace=True)
        except Exception as e:  # noqa: BLE001
            logger.debug("scalar index (%s) skipped: %s", col, e)

    # Always write the documents table so metadata + media_uri + thumbnail
    # travel with the chunks. `audio_root` is used to resolve `file://` URIs
    # for local deploys; `media_base_uri` overrides to produce e.g. hf://…/s3://…
    if audio_root is not None or media_base_uri is not None or thumbnail_dir is not None:
        doc_rows = [
            _document_row(
                d,
                audio_root=audio_root,
                media_base_uri=media_base_uri,
                metadata_index=metadata_index,
                thumbnail_dir=thumbnail_dir,
                doc_language=doc_language,
            )
            for d in docs_list
        ]
        doc_mode = "append" if (Path(db_path) / "documents.lance").exists() else "create"
        _write_documents_table(db_path, doc_rows, mode=doc_mode)

    return table


def ingest_document(
    db_path: str | Path,
    doc: AudioMetadata,
    *,
    audio_root: str | Path | None = None,
    media_base_uri: str | None = None,
    table_name: str = "chunks",
    metadata_csv: str | Path | None = None,
    thumbnail_dir: str | Path | None = None,
    fts_language: str = "English",
    doc_language: str | None = None,
) -> "lancedb.table.Table":
    """Convenience wrapper around :func:`ingest_many` for a single transcript."""
    return ingest_many(
        db_path,
        [doc],
        audio_root=audio_root,
        media_base_uri=media_base_uri,
        table_name=table_name,
        metadata_csv=metadata_csv,
        thumbnail_dir=thumbnail_dir,
        fts_language=fts_language,
        doc_language=doc_language,
    )


def reindex_fts(
    db_path: str | Path,
    *,
    table_name: str = "chunks",
    language: str = "Swedish",
    with_position: bool = True,
    remove_stop_words: bool = False,
    ascii_folding: bool = True,
) -> "lancedb.table.Table":
    """Rebuild the FTS index on an existing chunks table with a different config.

    Use this when you want to change the stemmer/language without re-ingesting
    all JSON files. Fast — only the inverted index is rewritten.
    """
    db = lancedb.connect(str(db_path))
    table = db.open_table(table_name)
    table.create_fts_index(
        "text",
        replace=True,
        with_position=with_position,
        remove_stop_words=remove_stop_words,
        ascii_folding=ascii_folding,
        language=language,
    )
    return table
