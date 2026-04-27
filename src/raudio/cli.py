"""CLI: `raudio {transcribe, detect-language, ingest, search}` — built with Typer."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated

import typer

from .ingest import ingest_many, load_transcript, reindex_fts
from .search import extract_query_terms, iter_matching_words, nearest_chunks, timecode

app = typer.Typer(
    name="raudio",
    help="Audio transcription → Lance ingestion → full-text search.",
    no_args_is_help=True,
    add_completion=False,
)


# Global state carried between the root callback and subcommands.
class _Ctx:
    db: Path = Path("./transcripts.lance")
    table: str = "chunks"


@app.callback()
def _root(
    ctx: typer.Context,
    db: Annotated[
        Path,
        typer.Option("--db", help="Path to the Lance database."),
    ] = Path("./transcripts.lance"),
    table: Annotated[
        str,
        typer.Option("--table", help="Table name."),
    ] = "chunks",
) -> None:
    _Ctx.db = db
    _Ctx.table = table


@app.command("transcribe")
def cmd_transcribe(
    audio_dir: Annotated[Path, typer.Option("--audio-dir", exists=True, file_okay=False)],
    language: Annotated[str, typer.Option("--language", help="ISO-639-1 code (sv, en, …).")] = "sv",
    model: Annotated[str, typer.Option("--model")] = "KBLab/kb-whisper-large",
    emissions_model: Annotated[str | None, typer.Option("--emissions-model")] = None,
    vad: Annotated[str, typer.Option("--vad", help="pyannote or silero.")] = "pyannote",
    backend: Annotated[str, typer.Option("--backend", help="ct2 or hf.")] = "ct2",
    device: Annotated[str, typer.Option("--device")] = "cuda",
    cache_dir: Annotated[Path, typer.Option("--cache-dir")] = Path("models"),
    output_root: Annotated[Path, typer.Option("--output-root")] = Path("output"),
    batch_size_features: Annotated[
        int,
        typer.Option(
            "--batch-size-features",
            help="Batch size for Whisper/wav2vec2 inference. 64 fits ~25 GB on a 96 GB GPU.",
        ),
    ] = 64,
    num_workers_features: Annotated[int, typer.Option("--num-workers-features")] = 8,
    batch_size_files: Annotated[int, typer.Option("--batch-size-files")] = 1,
    num_workers_files: Annotated[int, typer.Option("--num-workers-files")] = 2,
    beam_size: Annotated[
        int,
        typer.Option(
            "--beam-size",
            help=(
                "Whisper beam size. 1 is ~3-5× faster than the default 5 "
                "with negligible quality loss on clean audio. Bump to 5 if "
                "you see obviously garbled transcripts."
            ),
        ),
    ] = 1,
) -> None:
    """Run easytranscriber on a directory of audio/video files → alignment JSONs."""
    # Lazy import — the `[transcribe]` extra is optional.
    from .transcribe import run_transcribe

    if vad not in {"pyannote", "silero"}:
        raise typer.BadParameter("--vad must be 'pyannote' or 'silero'")
    if backend not in {"ct2", "hf"}:
        raise typer.BadParameter("--backend must be 'ct2' or 'hf'")

    run_transcribe(
        audio_dir=audio_dir,
        language=language,
        model=model,
        emissions_model=emissions_model,
        vad=vad,
        backend=backend,
        device=device,
        cache_dir=cache_dir,
        output_root=output_root,
        batch_size_features=batch_size_features,
        num_workers_features=num_workers_features,
        batch_size_files=batch_size_files,
        num_workers_files=num_workers_files,
        beam_size=beam_size,
    )


@app.command("detect-language")
def cmd_detect_language(
    audio_dir: Annotated[Path, typer.Option("--audio-dir", exists=True, file_okay=False)],
    model: Annotated[
        str,
        typer.Option(
            "--model",
            help=(
                "Classifier model. Default facebook/mms-lid-256 (SOTA for "
                "language ID). Also supports multilingual Whisper like "
                "openai/whisper-large-v3. Never use language-fine-tuned "
                "models (e.g. KBLab/kb-whisper-large) — they over-predict."
            ),
        ),
    ] = "openai/whisper-large-v3",
    cache_dir: Annotated[Path, typer.Option("--cache-dir")] = Path("models"),
    sample_seconds: Annotated[
        float,
        typer.Option("--sample-seconds", help="Audio clip length fed to Whisper per sample."),
    ] = 30.0,
    sample_offset: Annotated[
        float,
        typer.Option(
            "--sample-offset",
            help=(
                "Base offset (seconds). We actually sample at 1x/3x/5x this "
                "value and vote, so default 60s → samples at 60s/180s/300s."
            ),
        ),
    ] = 60.0,
    device: Annotated[str, typer.Option("--device")] = "cuda",
    no_move: Annotated[
        bool,
        typer.Option("--no-move", help="Report detected languages without moving files."),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Show planned moves without executing them."),
    ] = False,
) -> None:
    """Detect language per file via Whisper and sort into <audio-dir>/<lang>/ subfolders."""
    from .detect_language import detect_and_sort

    detect_and_sort(
        audio_dir=audio_dir,
        model=model,
        cache_dir=cache_dir,
        sample_seconds=sample_seconds,
        sample_offset=sample_offset,
        device=device,
        move=not no_move,
        dry_run=dry_run,
    )


@app.command("thumbnail")
def cmd_thumbnail(
    input_dir: Annotated[Path, typer.Option("--input-dir", exists=True, file_okay=False)] = Path("input"),
    output_dir: Annotated[Path, typer.Option("--output-dir")] = Path("thumbnails"),
    at_sec: Annotated[
        float,
        typer.Option("--at", help="Seek this many seconds into each video before grabbing a frame."),
    ] = 5.0,
    width: Annotated[int, typer.Option("--width", help="Target thumbnail width in pixels.")] = 480,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Regenerate thumbnails that already exist."),
    ] = False,
) -> None:
    """Extract a JPEG thumbnail per media file (via ffmpeg) into <output-dir>/{stem}.jpg."""
    from .thumbnails import generate_thumbnails

    generate_thumbnails(
        input_dir=input_dir,
        output_dir=output_dir,
        at_sec=at_sec,
        width=width,
        overwrite=overwrite,
    )


@app.command("download")
def cmd_download(
    csv_path: Annotated[Path, typer.Option("--csv", exists=True, dir_okay=False)],
    output_dir: Annotated[Path, typer.Option("--output-dir")] = Path("input"),
    limit: Annotated[
        int | None,
        typer.Option("--limit", help="Only download the first N rows (for testing)."),
    ] = None,
    concurrency: Annotated[
        int,
        typer.Option("--concurrency", help="Simultaneous downloads."),
    ] = 4,
    timeout: Annotated[
        float,
        typer.Option("--timeout", help="Per-request timeout (seconds)."),
    ] = 600.0,
) -> None:
    """Bulk-download Riksarkivet media from a ``video_batcher`` CSV into <output-dir>/{bildid}.mp4."""
    from .download import download_manifest

    download_manifest(
        csv_path=csv_path,
        output_dir=output_dir,
        limit=limit,
        concurrency=concurrency,
        timeout=timeout,
    )


@app.command("ingest")
def cmd_ingest(
    json_paths: Annotated[list[Path], typer.Argument(metavar="JSON...", help="AudioMetadata JSON files.")],
    audio_root: Annotated[
        Path | None,
        typer.Option(
            "--audio-root",
            help=(
                "Local directory holding the source media files. When set "
                "(and --media-base-uri is not), each row's media_uri is "
                "generated as file:///abs/path/<filename>."
            ),
        ),
    ] = None,
    media_base_uri: Annotated[
        str | None,
        typer.Option(
            "--media-base-uri",
            help=(
                "Base URI under which to reference videos in the documents "
                "table. Overrides --audio-root. Examples: "
                "'hf://buckets/you/videos/', 's3://bucket/videos/', "
                "'https://cdn.example.com/videos/'."
            ),
        ),
    ] = None,
    metadata_csv: Annotated[
        Path | None,
        typer.Option(
            "--metadata-csv",
            help=(
                "Optional video_batcher CSV (referenskod;namn;extraid;bildid). "
                "Joined to transcripts by bildid == audio_path stem."
            ),
        ),
    ] = None,
    thumbnail_dir: Annotated[
        Path | None,
        typer.Option(
            "--thumbnail-dir",
            help=(
                "Directory of {stem}.jpg thumbnails (see `raudio thumbnail`). "
                "If set, each document row stores the path to its thumbnail; "
                "the viewer can then serve them for a gallery."
            ),
        ),
    ] = None,
    fts_language: Annotated[
        str,
        typer.Option(
            "--fts-language",
            help=(
                "Stemmer/stop-word language for the FTS index. "
                "Use 'Swedish' for Swedish text — default 'English' mis-stems "
                "forms like 'ministern'/'vägen'/'ansåg'. Supported: English, "
                "Swedish, Norwegian, Danish, Finnish, French, German, Spanish, "
                "Italian, Portuguese, Dutch, Russian, and more."
            ),
        ),
    ] = "English",
    doc_language: Annotated[
        str | None,
        typer.Option(
            "--doc-language",
            help=(
                "2-letter ISO 639-1 language code stamped on every ingested "
                "row (documents.language + chunks.language). If omitted, we "
                "infer from the alignments dir: output/sv/alignments → 'sv'."
            ),
        ),
    ] = None,
) -> None:
    """Ingest one or more easytranscriber AudioMetadata JSON files."""
    docs = [load_transcript(p) for p in json_paths]

    # Infer doc_language from the alignments dir if not explicitly passed.
    # `output/sv/alignments/foo.json` → parent.parent.name == 'sv'.
    if doc_language is None and json_paths:
        candidate = json_paths[0].parent.parent.name
        if len(candidate) == 2 and candidate.isalpha():
            doc_language = candidate.lower()

    table = ingest_many(
        _Ctx.db,
        docs,
        audio_root=audio_root,
        media_base_uri=media_base_uri,
        table_name=_Ctx.table,
        metadata_csv=metadata_csv,
        thumbnail_dir=thumbnail_dir,
        fts_language=fts_language,
        doc_language=doc_language,
    )
    suffix = ""
    if doc_language:
        suffix += f" + language={doc_language}"
    if media_base_uri:
        suffix += f" + media URIs under {media_base_uri}"
    elif audio_root:
        suffix += f" + media URIs (file://) from {audio_root}"
    if metadata_csv:
        suffix += f" + metadata from {metadata_csv.name}"
    if thumbnail_dir:
        suffix += f" + thumbnails from {thumbnail_dir}"
    suffix += f" + FTS({fts_language})"
    typer.echo(
        f"Ingested {len(docs)} transcript(s) → '{_Ctx.table}' now has "
        f"{table.count_rows()} chunk row(s){suffix}.",
        err=True,
    )


@app.command("serve")
def cmd_serve(
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port")] = 8000,
) -> None:
    """Launch the API-only FastAPI backend against the Lance DB.

    The Bun frontend in ./frontend/ proxies /api/* to this server.
    """
    from backend import run

    run(db_path=_Ctx.db, host=host, port=port)


@app.command("reindex-fts")
def cmd_reindex_fts(
    language: Annotated[
        str,
        typer.Option(
            "--language",
            help="Stemmer/stop-word language. Use 'Swedish' for Swedish text.",
        ),
    ] = "Swedish",
    with_position: Annotated[
        bool,
        typer.Option("--with-position/--no-with-position", help="Required for phrase queries."),
    ] = True,
    remove_stop_words: Annotated[
        bool,
        typer.Option("--remove-stop-words/--keep-stop-words"),
    ] = False,
    ascii_folding: Annotated[
        bool,
        typer.Option("--ascii-folding/--no-ascii-folding"),
    ] = True,
) -> None:
    """Rebuild only the FTS index on an existing chunks table. No re-ingest."""
    reindex_fts(
        _Ctx.db,
        table_name=_Ctx.table,
        language=language,
        with_position=with_position,
        remove_stop_words=remove_stop_words,
        ascii_folding=ascii_folding,
    )
    typer.echo(
        f"Rebuilt FTS index on '{_Ctx.table}' "
        f"(language={language}, with_position={with_position}, "
        f"remove_stop_words={remove_stop_words}, ascii_folding={ascii_folding}).",
        err=True,
    )


@app.command("search")
def cmd_search(
    query: Annotated[str, typer.Argument()],
    limit: Annotated[int, typer.Option("-n", "--limit")] = 10,
    where: Annotated[
        str | None,
        typer.Option("--where", help="Optional SQL filter, e.g. language = 'en'."),
    ] = None,
    words: Annotated[
        bool,
        typer.Option(
            "--words/--no-words",
            help="List each matching word with its exact timestamp (ms precision).",
        ),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json-output", help="Emit raw JSON hits to stdout."),
    ] = False,
) -> None:
    """Run a full-text (Tantivy BM25) query."""
    hits = nearest_chunks(
        _Ctx.db,
        query,
        table_name=_Ctx.table,
        limit=limit,
        where=where,
        include_alignments=words or json_output,
    )

    if json_output:
        json.dump(hits, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    if not hits:
        typer.echo("(no hits)", err=True)
        return

    terms = extract_query_terms(query) if words else []
    for i, h in enumerate(hits, 1):
        typer.echo(
            f"{i:>2}. [{timecode(h['start'])}→{timecode(h['end'])}] "
            f"{Path(h['audio_path']).name}"
        )
        typer.echo(f"     {h['text']}")
        if words:
            matches = iter_matching_words(h, terms)
            if matches:
                for w in matches:
                    typer.echo(
                        f"     • [{timecode(w['start'], millis=True)}] "
                        f"{w['text'].strip()}"
                    )
            else:
                typer.echo("     (chunk matched, no exact word hit — phrase/stemming?)")


# ──────────────────────────────────────────────────────────────────────────
# Multimodal embedding subcommands (Phase 1+2 of the multimodal plan)
# ──────────────────────────────────────────────────────────────────────────


@app.command("embed-chunks")
def cmd_embed_chunks(
    backend: Annotated[
        str, typer.Option("--backend", help="Embedding backend ('vllm').")
    ] = "vllm",
    embed_url: Annotated[
        str, typer.Option("--embed-url", help="vLLM embedding server base URL.")
    ] = "http://127.0.0.1:8001",
    batch_size: Annotated[
        int, typer.Option("--batch-size", help="Texts per outer batch.")
    ] = 256,
    only_null: Annotated[
        bool,
        typer.Option(
            "--only-null/--all",
            help="Skip rows that already have an embedding (resumable).",
        ),
    ] = True,
    create_index: Annotated[
        bool,
        typer.Option(
            "--create-index/--no-create-index",
            help="Build IVF_PQ vector index after the embed pass.",
        ),
    ] = True,
    num_partitions: Annotated[int, typer.Option("--num-partitions")] = 256,
    num_sub_vectors: Annotated[int, typer.Option("--num-sub-vectors")] = 64,
) -> None:
    """Embed `chunks.text` → `chunks.text_embedding` (Qwen3-VL-8B, MRL=1024).

    Resumable: only rows where `text_embedding IS NULL` are processed.
    Builds an IVF_PQ cosine index on completion (skip with `--no-create-index`).
    """
    import lancedb
    import numpy as np
    import pyarrow as pa
    from tqdm import tqdm

    from .embeddings import make_client

    db = lancedb.connect(str(_Ctx.db))
    if _Ctx.table not in db.table_names():
        raise typer.Exit(f"Table '{_Ctx.table}' not found in {_Ctx.db}.")
    table = db.open_table(_Ctx.table)

    where = "text_embedding IS NULL" if only_null else None
    total = table.count_rows(filter=where) if where else table.count_rows()
    if total == 0:
        typer.echo("Nothing to embed (all rows already have text_embedding).", err=True)
        if create_index:
            _ensure_vector_index(table, "text_embedding", num_partitions, num_sub_vectors)
        return
    typer.echo(f"Embedding {total} chunk(s) via {backend} at {embed_url}.", err=True)

    client = make_client(backend=backend, embed_url=embed_url)

    # Stream rows in pages, embed in sub-batches, accumulate updates.
    cursor = table.search().select(["doc_id", "speech_id", "chunk_id", "text"])
    if where:
        cursor = cursor.where(where, prefilter=False)
    rows = cursor.limit(total).to_list()

    pbar = tqdm(total=total, unit="chunk", smoothing=0.05)
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        vectors = client.embed_text([r["text"] for r in batch])
        _merge_insert_vectors(
            table,
            keys=[(r["doc_id"], r["speech_id"], r["chunk_id"]) for r in batch],
            column="text_embedding",
            vectors=vectors,
        )
        pbar.update(len(batch))
    pbar.close()

    if create_index:
        _ensure_vector_index(table, "text_embedding", num_partitions, num_sub_vectors)


@app.command("extract-chunk-frames")
def cmd_extract_chunk_frames(
    audio_root: Annotated[
        Path,
        typer.Option(
            "--audio-root", exists=True, file_okay=False,
            help="Root directory holding the source MP4s.",
        ),
    ] = Path("input/sv"),
    width: Annotated[int, typer.Option("--width")] = 448,
    jpeg_quality: Annotated[int, typer.Option("--quality")] = 4,
    jobs: Annotated[int, typer.Option("--jobs", help="Parallel ffmpeg workers.")] = 4,
    timeout: Annotated[float, typer.Option("--timeout", help="Per-frame timeout (s).")] = 30.0,
    only_null: Annotated[
        bool,
        typer.Option("--only-null/--all", help="Resumable: skip chunks with frame_blob set."),
    ] = True,
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size",
            help=(
                "Chunks per merge_insert call. Default 0 = single flush at end. "
                "merge_insert on a multi-GB blob table rewrites whole fragments "
                "per call, so batching costs more than it saves."
            ),
        ),
    ] = 0,
) -> None:
    """Extract one JPEG per chunk at chunk.start → `chunks.frame_blob`.

    Resumable: only rows where `frame_blob IS NULL` are processed.
    Failed extractions are logged to stderr and skipped (the row stays NULL).
    """
    import lancedb
    import pyarrow as pa
    from lance import blob_array
    from tqdm import tqdm

    from .audio import resolve_source
    from .frames import extract_chunk_frames_parallel

    db = lancedb.connect(str(_Ctx.db))
    if _Ctx.table not in db.table_names():
        raise typer.Exit(f"Table '{_Ctx.table}' not found in {_Ctx.db}.")
    table = db.open_table(_Ctx.table)

    # Lance 4.0 panics on `IS NULL` against lance.blob.v2 columns. Filter on
    # `frame_mime` (regular string, populated atomically with frame_blob) instead.
    where = "frame_mime IS NULL" if only_null else None
    total = table.count_rows(filter=where) if where else table.count_rows()
    if total == 0:
        typer.echo("Nothing to extract (all rows already have frame_blob).", err=True)
        return
    typer.echo(
        f"Extracting frames for {total} chunk(s) from {audio_root} (jobs={jobs}).",
        err=True,
    )

    cursor = table.search().select(["doc_id", "speech_id", "chunk_id", "audio_path", "start"])
    if where:
        cursor = cursor.where(where, prefilter=False)
    rows = cursor.limit(total).to_list()

    # Resolve source path once per audio_path.
    src_cache: dict[str, Path | None] = {}
    work: list[tuple[str, int, int, float, Path]] = []
    missing = 0
    for r in rows:
        ap = r["audio_path"]
        if ap not in src_cache:
            src_cache[ap] = resolve_source(ap, audio_root)
        src = src_cache[ap]
        if src is None:
            missing += 1
            continue
        work.append((r["doc_id"], r["speech_id"], r["chunk_id"], float(r["start"]), src))

    if missing:
        typer.echo(f"  warning: {missing} chunk(s) had no resolvable source MP4 — skipped.", err=True)
    if not work:
        typer.echo("Nothing extractable.", err=True)
        return

    pbar = tqdm(total=len(work), unit="frame", smoothing=0.05)
    pending: list = []  # buffer ExtractedFrame results before a merge_insert
    n_ok = 0
    n_fail = 0

    def _flush(buf: list) -> None:
        nonlocal n_ok
        good = [f for f in buf if f.jpeg_bytes]
        if not good:
            return
        update = pa.table(
            {
                "doc_id":       pa.array([f.doc_id for f in good], pa.string()),
                "speech_id":    pa.array([f.speech_id for f in good], pa.int32()),
                "chunk_id":     pa.array([f.chunk_id for f in good], pa.int32()),
                "frame_blob":   blob_array([f.jpeg_bytes for f in good]),
                "frame_mime":   pa.array(["image/jpeg"] * len(good), pa.string()),
                "frame_width":  pa.array([f.width for f in good], pa.int32()),
                "frame_height": pa.array([f.height for f in good], pa.int32()),
            }
        )
        (
            table.merge_insert(["doc_id", "speech_id", "chunk_id"])
            .when_matched_update_all()
            .execute(update)
        )
        n_ok += len(good)

    for frame in extract_chunk_frames_parallel(
        work, width=width, jpeg_quality=jpeg_quality, timeout=timeout, jobs=jobs,
    ):
        if frame.error:
            n_fail += 1
            if n_fail <= 5:
                typer.echo(f"  ffmpeg failed: {frame.doc_id}@{frame.time_sec:.2f}s — {frame.error}", err=True)
        pending.append(frame)
        # batch_size==0 means accumulate everything and flush once at the end —
        # avoids re-rewriting the same multi-GB fragments per batch.
        if batch_size > 0 and len(pending) >= batch_size:
            _flush(pending)
            pending = []
        pbar.update(1)
    pbar.close()
    if pending:
        typer.echo(
            f"Writing {len(pending):,} frames to Lance via single merge_insert "
            f"(this may take several minutes on a large table)…",
            err=True,
        )
        _flush(pending)
    typer.echo(f"  ok={n_ok}  failed={n_fail}", err=True)


@app.command("embed-chunk-frames")
def cmd_embed_chunk_frames(
    backend: Annotated[str, typer.Option("--backend")] = "vllm",
    embed_url: Annotated[str, typer.Option("--embed-url")] = "http://127.0.0.1:8001",
    batch_size: Annotated[int, typer.Option("--batch-size")] = 16,
    only_null: Annotated[bool, typer.Option("--only-null/--all")] = True,
    create_index: Annotated[bool, typer.Option("--create-index/--no-create-index")] = True,
    num_partitions: Annotated[int, typer.Option("--num-partitions")] = 256,
    num_sub_vectors: Annotated[int, typer.Option("--num-sub-vectors")] = 64,
) -> None:
    """Embed each chunk's frame → `chunks.frame_embedding` (Qwen3-VL, MRL=1024).

    Reads `frame_blob` for chunks where the frame exists but no embedding
    has been computed yet. Resumable; builds IVF_PQ cosine index on completion.
    """
    import io

    import lance
    import lancedb
    import pyarrow as pa
    from tqdm import tqdm

    from .embeddings import make_client

    db = lancedb.connect(str(_Ctx.db))
    if _Ctx.table not in db.table_names():
        raise typer.Exit(f"Table '{_Ctx.table}' not found in {_Ctx.db}.")
    table = db.open_table(_Ctx.table)

    # We need raw lance.LanceDataset for take_blobs() — chunks lives at
    # <db>/<table>.lance/.
    ds = lance.dataset(str(_Ctx.db / f"{_Ctx.table}.lance"))

    # Lance 4.0 panics on `IS [NOT] NULL` against lance.blob.v2 columns.
    # `frame_mime` is the sibling regular string column populated atomically
    # with frame_blob, so it's a safe proxy for "has a frame".
    if only_null:
        where = "frame_embedding IS NULL AND frame_mime IS NOT NULL"
    else:
        where = "frame_mime IS NOT NULL"
    total = table.count_rows(filter=where)
    if total == 0:
        typer.echo("Nothing to embed (no chunks have frame_blob without frame_embedding).", err=True)
        if create_index:
            _ensure_vector_index(table, "frame_embedding", num_partitions, num_sub_vectors)
        return
    typer.echo(f"Embedding {total} frame(s) via {backend} at {embed_url}.", err=True)

    client = make_client(backend=backend, embed_url=embed_url)

    # We use take_blobs() against the raw dataset, so we need _rowid for
    # the rows that match `where`. lancedb's Table doesn't expose row ids
    # directly, but `lance.LanceDataset.to_table(..., with_row_id=True)`
    # does — and the filter syntax is the same.
    keyed = ds.to_table(
        columns=["doc_id", "speech_id", "chunk_id"],
        filter=where,
        with_row_id=True,
    )
    rowids = keyed.column("_rowid").to_pylist()
    doc_ids = keyed.column("doc_id").to_pylist()
    speech_ids = keyed.column("speech_id").to_pylist()
    chunk_ids = keyed.column("chunk_id").to_pylist()

    pbar = tqdm(total=len(rowids), unit="frame", smoothing=0.05)
    for start in range(0, len(rowids), batch_size):
        end = start + batch_size
        batch_rowids = rowids[start : end]
        jpeg_bytes_list: list[bytes] = []
        for b in ds.take_blobs("frame_blob", indices=batch_rowids):
            with b as f:
                jpeg_bytes_list.append(f.read())
        vectors = client.embed_image(jpeg_bytes_list)
        _merge_insert_vectors(
            table,
            keys=list(zip(doc_ids[start:end], speech_ids[start:end], chunk_ids[start:end])),
            column="frame_embedding",
            vectors=vectors,
        )
        pbar.update(len(batch_rowids))
    pbar.close()

    if create_index:
        _ensure_vector_index(table, "frame_embedding", num_partitions, num_sub_vectors)


def _merge_insert_vectors(
    table: "object",
    *,
    keys: list[tuple[str, int, int]],
    column: str,
    vectors,           # np.ndarray of shape (N, dim) or list[list[float]]
    dim: int = 1024,
) -> None:
    """Merge a batch of (doc_id, speech_id, chunk_id) → vector updates.

    Used by both `embed-chunks` (text vectors) and `embed-chunk-frames`
    (image vectors). Both share the same chunk-keying scheme; only the
    vector column name and the source data differ.
    """
    import pyarrow as pa

    update = pa.table(
        {
            "doc_id":    pa.array([k[0] for k in keys], pa.string()),
            "speech_id": pa.array([k[1] for k in keys], pa.int32()),
            "chunk_id":  pa.array([k[2] for k in keys], pa.int32()),
            column:      pa.array(
                [v.tolist() if hasattr(v, "tolist") else list(v) for v in vectors],
                pa.list_(pa.float32(), dim),
            ),
        }
    )
    (
        table.merge_insert(["doc_id", "speech_id", "chunk_id"])
        .when_matched_update_all()
        .execute(update)
    )


def _ensure_vector_index(
    table: "object", column: str, num_partitions: int, num_sub_vectors: int
) -> None:
    """Build IVF_PQ cosine index on ``column`` once embeddings are populated.

    Refuses to run while the column still has nulls — Lance's index
    builder doesn't handle partial-NULL vector columns gracefully.
    """
    null_filter = f"{column} IS NULL"
    nulls = table.count_rows(filter=null_filter)
    if nulls > 0:
        typer.echo(
            f"  skipping index on {column}: {nulls} row(s) still NULL (run with --only-null disabled?).",
            err=True,
        )
        return
    typer.echo(
        f"Building IVF_PQ cosine index on '{column}' "
        f"(num_partitions={num_partitions}, num_sub_vectors={num_sub_vectors}) …",
        err=True,
    )
    table.create_index(
        metric="cosine",
        vector_column_name=column,
        index_type="IVF_PQ",
        num_partitions=num_partitions,
        num_sub_vectors=num_sub_vectors,
        replace=True,
    )
    typer.echo("  done.", err=True)


if __name__ == "__main__":
    app()
