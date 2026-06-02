"""CLI for raudio — built with Typer.

Exposes the full pipeline as subcommands: ``transcribe``, ``detect-language``,
``thumbnail``, ``download``, ``ingest``, ``reindex-fts``, ``search``, ``serve``,
``embed-chunks``, ``extract-chunk-frames``, ``embed-chunk-frames``, ``compact``.
Run ``raudio --help`` for the authoritative list.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, NoReturn

import typer

from .ingest.ingest import ingest_many, load_transcript, reindex_fts
from .retrieval.search import extract_query_terms, iter_matching_words, nearest_chunks, timecode

if TYPE_CHECKING:
    import lancedb


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


def _configure_logging(log_file: Path | None) -> None:
    """Wire up where library log records go: always the terminal (stderr), and
    additionally ``log_file`` when given.

    Modules log via ``logging.getLogger(__name__)``; the CLI is the single place
    that decides the destination, per the writing-python logging convention.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )


def _die(message: str) -> NoReturn:
    """Print ``message`` to stderr and exit non-zero.

    ``typer.Exit`` takes an integer *exit code*, not a message — passing a
    string sets a bogus code and silently drops the text. This prints the
    message first, then exits 1.
    """
    typer.echo(message, err=True)
    raise typer.Exit(code=1)


def _require_table(db: lancedb.DBConnection, table: str) -> None:
    """Abort with a clear message if ``table`` is missing from ``db``."""
    if table not in db.list_tables().tables:
        _die(f"Table '{table}' not found in {_Ctx.db}.")


@app.callback()
def _root(
    db: Annotated[
        Path,
        typer.Option("--db", help="Path to the Lance database."),
    ] = Path("./transcripts.lance"),
    table: Annotated[
        str,
        typer.Option("--table", help="Table name."),
    ] = "chunks",
    log_file: Annotated[
        Path | None,
        typer.Option("--log-file", help="Also write logs to this file (terminal output stays on)."),
    ] = None,
) -> None:
    _configure_logging(log_file)
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
    chunk_size: Annotated[
        int,
        typer.Option(
            "--chunk-size",
            help="Max VAD chunk length in seconds. Lower → finer-grained chunks (default 30).",
        ),
    ] = 30,
    alignment_strategy: Annotated[
        str,
        typer.Option(
            "--alignment-strategy",
            help="'chunk' uses VAD segments; 'speech' splits each speech into fixed chunk-size windows.",
        ),
    ] = "chunk",
) -> None:
    """Run easytranscriber on a directory of audio/video files → alignment JSONs."""
    # Lazy import — the `[transcribe]` extra is optional.
    from .asr.transcribe import run_transcribe

    if vad not in {"pyannote", "silero"}:
        raise typer.BadParameter("--vad must be 'pyannote' or 'silero'")
    if backend not in {"ct2", "hf"}:
        raise typer.BadParameter("--backend must be 'ct2' or 'hf'")
    if alignment_strategy not in {"chunk", "speech"}:
        raise typer.BadParameter("--alignment-strategy must be 'chunk' or 'speech'")

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
        num_workers_files=num_workers_files,
        beam_size=beam_size,
        chunk_size=chunk_size,
        alignment_strategy=alignment_strategy,
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
    num_windows: Annotated[
        int,
        typer.Option(
            "--num-windows",
            help="Clips sampled per file, spread evenly across the whole recording (duration-aware).",
        ),
    ] = 8,
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
    from .asr.detect_language import detect_and_sort

    detect_and_sort(
        audio_dir=audio_dir,
        model=model,
        cache_dir=cache_dir,
        sample_seconds=sample_seconds,
        num_windows=num_windows,
        device=device,
        move=not no_move,
        dry_run=dry_run,
    )


@app.command("thumbnail")
def cmd_thumbnail(
    input_dir: Annotated[Path, typer.Option("--input-dir", exists=True, file_okay=False)] = Path(
        "input"
    ),
    output_dir: Annotated[Path, typer.Option("--output-dir")] = Path("thumbnails"),
    at_sec: Annotated[
        float,
        typer.Option(
            "--at", help="Seek this many seconds into each video before grabbing a frame."
        ),
    ] = 5.0,
    width: Annotated[int, typer.Option("--width", help="Target thumbnail width in pixels.")] = 480,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Regenerate thumbnails that already exist."),
    ] = False,
) -> None:
    """Extract a JPEG thumbnail per media file (via ffmpeg) into <output-dir>/{stem}.jpg."""
    from .media.thumbnails import generate_thumbnails

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
    from .media.download import download_manifest

    download_manifest(
        csv_path=csv_path,
        output_dir=output_dir,
        limit=limit,
        concurrency=concurrency,
        timeout=timeout,
    )


@app.command("ingest")
def cmd_ingest(
    json_paths: Annotated[
        list[Path], typer.Argument(metavar="JSON...", help="AudioMetadata JSON files.")
    ],
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
            f"{i:>2}. [{timecode(h['start'])}→{timecode(h['end'])}] {Path(h['audio_path']).name}"
        )
        typer.echo(f"     {h['text']}")
        if words:
            matches = iter_matching_words(h, terms)
            if matches:
                for w in matches:
                    typer.echo(f"     • [{timecode(w['start'], millis=True)}] {w['text'].strip()}")
            else:
                typer.echo("     (chunk matched, no exact word hit — phrase/stemming?)")


# ──────────────────────────────────────────────────────────────────────────
# Feature columns — derive new columns via Lance data evolution
# ──────────────────────────────────────────────────────────────────────────


@app.command("feature")
def cmd_feature(
    name: Annotated[str, typer.Argument(help="Feature column to build (see the list below).")],
    url: Annotated[
        str | None,
        typer.Option("--url", help="Model server base URL (default: the feature's own)."),
    ] = None,
    batch_size: Annotated[int, typer.Option("--batch-size", help="Rows per embed batch.")] = 256,
    only_null: Annotated[
        bool,
        typer.Option("--only-null/--all", help="Top up unpopulated rows (resumable) vs rebuild."),
    ] = True,
    create_index: Annotated[
        bool,
        typer.Option("--create-index/--no-create-index", help="Build IVF_PQ index (vector features)."),
    ] = True,
    num_partitions: Annotated[int, typer.Option("--num-partitions")] = 256,
    num_sub_vectors: Annotated[int, typer.Option("--num-sub-vectors")] = 64,
    checkpoint: Annotated[
        Path | None,
        typer.Option("--checkpoint", help="batch_udf checkpoint file → crash-resumable."),
    ] = None,
) -> None:
    """Build a derived feature column via Lance data evolution (add_columns).

    Available features:

      * text_embedding  — chunks.text  → 2048-d vector (semantic search)
      * frame_embedding — chunk_frames → 2048-d vector (visual search)
      * summary         — chunks.text  → one-line LLM summary
      * caption         — chunk_frames → VLM caption

    Attaches one new column file, no fragment rewrites. `--only-null` (default)
    tops up rows a later ingest added; `--all` drops and rebuilds.
    """
    import lancedb
    from tqdm import tqdm

    from .features.columns import FEATURES, FeatureRunOptions

    feature = FEATURES.get(name)
    if feature is None:
        _die(f"Unknown feature '{name}'. Available: {', '.join(FEATURES)}.")

    if feature.table == "chunks":
        _require_table(lancedb.connect(str(_Ctx.db)), "chunks")
    elif not (_Ctx.db / f"{feature.table}.lance").exists():
        _die(f"Table '{feature.table}' missing — run `raudio extract-chunk-frames` first.")

    options = FeatureRunOptions(
        url=url,
        batch_rows=batch_size,
        overwrite=not only_null,
        create_index=create_index,
        num_partitions=num_partitions,
        num_sub_vectors=num_sub_vectors,
        checkpoint=checkpoint,
    )
    typer.echo(f"Building feature '{name}' on {feature.table} …", err=True)
    with tqdm(unit="row", smoothing=0.05) as pbar:
        written = feature.run(_Ctx.db, options, pbar.update)
    typer.echo(f"  done: {written} row(s) written.", err=True)


@app.command("extract-chunk-frames")
def cmd_extract_chunk_frames(
    audio_root: Annotated[
        Path,
        typer.Option(
            "--audio-root",
            exists=True,
            file_okay=False,
            help="Root directory holding the source MP4s.",
        ),
    ] = Path("input/sv"),
    width: Annotated[int, typer.Option("--width")] = 448,
    jpeg_quality: Annotated[int, typer.Option("--quality")] = 4,
    jobs: Annotated[int, typer.Option("--jobs", help="Parallel ffmpeg workers.")] = 4,
    timeout: Annotated[float, typer.Option("--timeout", help="Per-frame timeout (s).")] = 30.0,
    every_seconds: Annotated[
        float,
        typer.Option(
            "--every-seconds",
            help="Sample a frame every N seconds across each chunk (0 = one frame at chunk.start).",
        ),
    ] = 0.0,
    only_null: Annotated[
        bool,
        typer.Option("--only-null/--all", help="Resumable: skip chunks that already have frames."),
    ] = True,
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size",
            help=(
                "Frames per append flush. Default 0 = flush every 2000 frames "
                "during extraction. Appends are cheap (~100 ms/call) and never "
                "rewrite existing fragments, so frequent flushes just limit how "
                "much work a crash can lose."
            ),
        ),
    ] = 0,
    limit: Annotated[
        int,
        typer.Option(
            "--limit",
            help="Debug: extract only the first N chunks (0 = no limit).",
        ),
    ] = 0,
) -> None:
    """Extract JPEG frame(s) per chunk → `chunk_frames.lance` (NEW table).

    By default grabs one frame at `chunk.start`; with `--every-seconds N` it
    samples a frame every N seconds across each chunk's [start, end], numbered
    `frame_idx` 0..K-1. Writes a separate append-only `chunk_frames` table keyed
    by (doc_id, speech_id, chunk_id, frame_idx) — no `merge_insert` against the
    wide `chunks` schema (which crashes the Lance 4.0 decoder). Resumable: skips
    chunks that already have any frame.
    """
    import lancedb
    from tqdm import tqdm

    from .ingest.audio import resolve_source
    from .media.frames import (
        FrameJob,
        existing_frame_keys,
        extract_chunk_frames_parallel,
        sample_times,
        write_chunk_frames,
    )

    db = lancedb.connect(str(_Ctx.db))
    _require_table(db, _Ctx.table)
    chunks_tbl = db.open_table(_Ctx.table)
    frames_path = _Ctx.db / "chunk_frames.lance"
    frames_exists = "chunk_frames" in db.list_tables().tables

    if frames_exists and not only_null:
        # `--all` → drop up front so the rebuild is clean even if extraction
        # yields nothing (append-mode would otherwise duplicate every frame).
        typer.echo("  --all: dropping existing chunk_frames for a clean rebuild.", err=True)
        db.drop_table("chunk_frames")
        frames_exists = False

    # Resume at chunk granularity: skip any chunk that already has ≥1 frame.
    frame_keys = existing_frame_keys(frames_path) if (frames_exists and only_null) else set()
    already = {(d, s, c) for d, s, c, _ in frame_keys}
    if already:
        typer.echo(f"  {len(already):,} chunk(s) already have frames.", err=True)

    rows = (
        chunks_tbl.search()
        .select(["doc_id", "speech_id", "chunk_id", "audio_path", "start", "end"])
        .limit(chunks_tbl.count_rows())
        .to_list()
    )
    rows = [
        r for r in rows if (r["doc_id"], int(r["speech_id"]), int(r["chunk_id"])) not in already
    ]
    if limit > 0:
        rows = rows[:limit]
        typer.echo(f"  --limit {limit} → restricting to first {len(rows)} chunk(s).", err=True)
    if not rows:
        typer.echo("Nothing to extract.", err=True)
        return

    # Resolve each chunk's source MP4 (cached per audio_path) into frame job(s).
    src_cache: dict[str, Path | None] = {}
    frame_jobs: list[FrameJob] = []
    missing = 0
    for r in rows:
        ap = r["audio_path"]
        if ap not in src_cache:
            src_cache[ap] = resolve_source(ap, audio_root)
        src = src_cache[ap]
        if src is None:
            missing += 1
            continue
        for frame_idx, time_sec in enumerate(sample_times(r["start"], r["end"], every_seconds)):
            frame_jobs.append(
                FrameJob(
                    doc_id=r["doc_id"],
                    speech_id=r["speech_id"],
                    chunk_id=r["chunk_id"],
                    frame_idx=frame_idx,
                    time_sec=time_sec,
                    source=src,
                )
            )
    if missing:
        typer.echo(f"  warning: {missing} chunk(s) had no resolvable source MP4 — skipped.", err=True)
    if not frame_jobs:
        typer.echo("Nothing extractable.", err=True)
        return

    typer.echo(
        f"Extracting {len(frame_jobs)} frame(s) from {audio_root} (jobs={jobs}).", err=True
    )
    frames = extract_chunk_frames_parallel(
        frame_jobs, width=width, jpeg_quality=jpeg_quality, timeout=timeout, workers=jobs
    )
    with tqdm(total=len(frame_jobs), unit="frame", smoothing=0.05) as pbar:
        n_ok, n_fail = write_chunk_frames(
            frames_path,
            frames,
            create=not frames_exists,
            batch=batch_size if batch_size > 0 else 2000,
            progress=pbar.update,
        )
    typer.echo(f"  ok={n_ok}  failed={n_fail}", err=True)


@app.command("compact")
def cmd_compact(
    target_rows_per_fragment: Annotated[
        int,
        typer.Option(
            "--target-rows",
            help="Compaction target. Smaller fragments will be merged up to this size.",
        ),
    ] = 1024 * 1024,
    rebuild_indexes: Annotated[
        bool,
        typer.Option(
            "--rebuild-indexes/--no-rebuild-indexes",
            help="After compaction, rebuild IVF_PQ indexes that compaction invalidated.",
        ),
    ] = True,
    num_partitions: Annotated[int, typer.Option("--num-partitions")] = 256,
    num_sub_vectors: Annotated[int, typer.Option("--num-sub-vectors")] = 64,
) -> None:
    """Compact small fragments and (by default) rebuild IVF_PQ indexes.

    Lance recommends compacting *before* rebuilding indexes (compaction
    invalidates the row addresses an ANN index points at). The append-only
    `extract-chunk-frames` flushes and any incremental ingests leave a long
    tail of small fragments; one pass here merges them and rebuilds whichever
    vector indexes are fully populated. Operates on `--table` (default
    `chunks`); run it per table.
    """
    import lance
    import lancedb

    from .features.engine import ensure_vector_index

    db = lancedb.connect(str(_Ctx.db))
    _require_table(db, _Ctx.table)
    table = db.open_table(_Ctx.table)
    ds = lance.dataset(str(_Ctx.db / f"{_Ctx.table}.lance"))

    before = len(ds.get_fragments())
    typer.echo(
        f"Compacting {before} fragment(s) into target_rows_per_fragment={target_rows_per_fragment:,} …",
        err=True,
    )
    metrics = ds.optimize.compact_files(target_rows_per_fragment=target_rows_per_fragment)
    typer.echo(
        f"  done. fragments_removed={metrics.fragments_removed} "
        f"fragments_added={metrics.fragments_added} "
        f"files_removed={metrics.files_removed} "
        f"files_added={metrics.files_added}",
        err=True,
    )

    if not rebuild_indexes:
        return

    # Rebuild whichever vector indexes are fully populated. `ensure_vector_index`
    # skips (and logs) any column that still has NULL rows.
    for column in ("text_embedding", "frame_embedding"):
        if column in table.schema.names:
            ensure_vector_index(
                table,
                column,
                num_partitions=num_partitions,
                num_sub_vectors=num_sub_vectors,
            )


if __name__ == "__main__":
    app()
