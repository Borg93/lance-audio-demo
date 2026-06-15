"""Media + maintenance commands: ``thumbnail``, ``download``,
``extract-chunk-frames``, and ``compact``.

The speaker-diarization pipeline commands (``extract-speaker-turns`` →
``embed-speaker-turns`` → ``build-speakers`` → ``cluster-speakers`` plus their
shard merges) live in :mod:`raudio.cli.speaker`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer

from ._app import CliContext, _require_table, app

logger = logging.getLogger(__name__)


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
    from ..media.thumbnails import generate_thumbnails

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
    from ..media.download import download_manifest

    download_manifest(
        csv_path=csv_path,
        output_dir=output_dir,
        limit=limit,
        concurrency=concurrency,
        timeout=timeout,
    )


@app.command("extract-chunk-frames")
def cmd_extract_chunk_frames(
    ctx: typer.Context,
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

    from ..ingest.audio import resolve_source
    from ..media.frames import (
        FrameJob,
        existing_frame_keys,
        extract_chunk_frames_parallel,
        sample_times,
        write_chunk_frames,
    )

    cfg: CliContext = ctx.obj
    db = lancedb.connect(str(cfg.db))
    _require_table(db, cfg.table, cfg.db)
    chunks_tbl = db.open_table(cfg.table)
    frames_path = cfg.db / "chunk_frames.lance"
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
        typer.echo(
            f"  warning: {missing} chunk(s) had no resolvable source MP4 — skipped.", err=True
        )
    if not frame_jobs:
        typer.echo("Nothing extractable.", err=True)
        return

    typer.echo(f"Extracting {len(frame_jobs)} frame(s) from {audio_root} (jobs={jobs}).", err=True)
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
    ctx: typer.Context,
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

    from ..features.engine import ensure_vector_index

    cfg: CliContext = ctx.obj
    db = lancedb.connect(str(cfg.db))
    _require_table(db, cfg.table, cfg.db)
    table = db.open_table(cfg.table)
    ds = lance.dataset(str(cfg.db / f"{cfg.table}.lance"))

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

    # Compaction rewrites row addresses, which also invalidates the scalar
    # (BTREE) indexes ingest builds for per-row lookups — recreate them or
    # `doc_id`/`audio_path` filters silently fall back to full scans. Mirrors
    # ingest.py: only the chunks table carries these columns, so guard on it.
    for col in ("doc_id", "audio_path"):
        if col in table.schema.names:
            try:
                table.create_scalar_index(col, index_type="BTREE", replace=True)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"scalar index ({col}) skipped: {e}")
