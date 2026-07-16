"""Media + maintenance commands: ``thumbnail``, ``download``,
``extract-chunk-frames``, and ``compact``.

The speaker-diarization pipeline commands (``extract-speaker-turns`` →
``embed-speaker-turns`` → ``build-speakers`` → ``cluster-speakers`` plus their
shard merges) live in :mod:`rmedia.cli.speaker`.
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
    from rmedia.modalities.av.thumbnails import generate_thumbnails

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
    from rmedia.modalities.av.download import download_manifest

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
            help="Root directory holding the source media files.",
        ),
    ] = Path("input/sv"),
) -> None:
    """Extract one JPEG per chunk → `chunk_frames.lance`, on the Ray actor pool.

    Delegates to the `extract_frames` registry stage (`rmedia pipeline run
    extract_frames`): ffmpeg runs inside Ray actors, resume is the key diff
    against existing (doc_id, speech_id, chunk_id) frames, and appends go
    through the single create/append path.
    """
    from rmedia.features.ray_av import run_append_stage
    from rmedia.features.stages import STAGES

    cfg: CliContext = ctx.obj
    appended = run_append_stage(cfg.db, STAGES["extract_frames"], audio_root=str(audio_root))
    typer.echo(f"  {appended} frame row(s) appended")


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

    from rmedia.core.engine import ensure_vector_index

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
