"""Media + maintenance commands: ``thumbnail``, ``download``,
``extract-chunk-frames``, the speaker pipeline (``extract-speaker-turns`` →
``embed-speaker-turns`` → ``build-speakers`` plus their shard merges), and
``compact``."""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

if TYPE_CHECKING:
    import lancedb
    import numpy as np

    from ..media.diarize import SpeakerTurn
    from ..media.voiceprint import TurnSpan

from ._app import CliContext, _die, _require_table, app

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


@app.command("extract-speaker-turns")
def cmd_extract_speaker_turns(
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
    model: Annotated[
        str,
        typer.Option("--model", help="pyannote diarization pipeline (HF model id)."),
    ] = "pyannote/speaker-diarization-community-1",
    jobs: Annotated[
        int,
        typer.Option(
            "--jobs",
            help=(
                "Reserved for symmetry with extract-chunk-frames. Diarization "
                "runs one video at a time on the GPU; values >1 are ignored."
            ),
        ),
    ] = 1,
    ffmpeg_timeout: Annotated[
        float,
        typer.Option("--ffmpeg-timeout", help="Per-video wav-extraction timeout (s)."),
    ] = 1800.0,
    only_null: Annotated[
        bool,
        typer.Option(
            "--only-null/--all",
            help="Resumable: skip videos that already have turns (--all rebuilds clean).",
        ),
    ] = True,
    limit: Annotated[
        int,
        typer.Option(
            "--limit",
            help="Debug: diarize only the first N videos (0 = no limit). DO use this.",
        ),
    ] = 0,
    num_shards: Annotated[
        int,
        typer.Option(
            "--num-shards",
            help=(
                "Split the corpus across N parallel workers (1 = no sharding). "
                "Diarization is CPU/pipeline-bound and barely uses the GPU, so "
                "running several workers on one GPU multiplies throughput. Launch "
                "N processes, one per --shard-index, each writing a disjoint slice "
                "to speaker_turns_shard{i}.lance; fold them back with "
                "`raudio merge-speaker-turns`."
            ),
        ),
    ] = 1,
    shard_index: Annotated[
        int,
        typer.Option(
            "--shard-index",
            help="This worker's shard in [0, num-shards). Only used when --num-shards > 1.",
        ),
    ] = 0,
) -> None:
    """Diarize each video → ``speaker_turns.lance`` (NEW append-only table).

    Reads the distinct ``doc_id`` → ``audio_path`` pairs from the ``chunks`` table,
    resolves each source MP4 under ``--audio-root``, runs pyannote's diarization
    pipeline in-process (loaded once, reused across videos), and appends one row
    per speaker turn (absolute seconds) to ``speaker_turns.lance`` keyed logically
    by ``(doc_id, turn_id)``. Resumable at video granularity: ``--only-null`` (the
    default) skips any ``doc_id`` already present. Honour ``--limit`` — diarizing
    the full corpus is slow.
    """
    import lancedb
    from tqdm import tqdm

    from ..ingest.audio import resolve_source
    from ..media.diarize import Diarizer, existing_doc_ids, shard_of, write_speaker_turns

    cfg: CliContext = ctx.obj
    db = lancedb.connect(str(cfg.db))
    _require_table(db, cfg.table, cfg.db)
    chunks_tbl = db.open_table(cfg.table)

    if num_shards < 1:
        raise typer.BadParameter("--num-shards must be >= 1")
    sharded = num_shards > 1
    if sharded and not 0 <= shard_index < num_shards:
        raise typer.BadParameter(f"--shard-index must be in [0, {num_shards})")

    # Shards stage to their own table; `merge-speaker-turns` folds them into the
    # canonical `speaker_turns` afterwards (separate tables avoid concurrent-write
    # commit conflicts that N appenders to one table would hit).
    table_name = f"speaker_turns_shard{shard_index}" if sharded else "speaker_turns"
    turns_path = cfg.db / f"{table_name}.lance"
    main_path = cfg.db / "speaker_turns.lance"
    existing_tables = db.list_tables().tables
    turns_exists = table_name in existing_tables

    if jobs > 1:
        logger.info("--jobs %d ignored: diarization runs one video at a time on the GPU.", jobs)

    if turns_exists and not only_null:
        typer.echo(f"  --all: dropping existing {table_name} for a clean rebuild.", err=True)
        db.drop_table(table_name)
        turns_exists = False

    # Resume: skip videos this table already has and — in shard mode — anything
    # already merged into the canonical table, so no worker ever re-diarizes a
    # video another worker or an earlier single run already finished.
    already: set[str] = set()
    if only_null and turns_exists:
        already |= existing_doc_ids(turns_path)
    if sharded and "speaker_turns" in existing_tables:
        already |= existing_doc_ids(main_path)
    if already:
        typer.echo(f"  {len(already):,} video(s) already diarized — skipping.", err=True)

    # Distinct (doc_id, audio_path) — one diarization per source video.
    rows = (
        chunks_tbl.search()
        .select(["doc_id", "audio_path"])
        .limit(chunks_tbl.count_rows())
        .to_list()
    )
    seen: dict[str, str] = {}
    for r in rows:
        seen.setdefault(r["doc_id"], r["audio_path"])
    docs = [(d, ap) for d, ap in seen.items() if d not in already]
    if sharded:
        docs = [(d, ap) for d, ap in docs if shard_of(d, num_shards) == shard_index]
    docs.sort(key=lambda t: t[0])
    if limit > 0:
        docs = docs[:limit]
        typer.echo(f"  --limit {limit} → restricting to first {len(docs)} video(s).", err=True)
    if not docs:
        typer.echo("Nothing to diarize.", err=True)
        return

    # Resolve sources up front so a missing MP4 doesn't waste a model load.
    resolved: list[tuple[str, Path]] = []
    missing = 0
    for doc_id, audio_path in docs:
        src = resolve_source(audio_path, audio_root)
        if src is None:
            missing += 1
            continue
        resolved.append((doc_id, src))
    if missing:
        typer.echo(
            f"  warning: {missing} video(s) had no resolvable source MP4 — skipped.", err=True
        )
    if not resolved:
        typer.echo("Nothing diarizable.", err=True)
        return

    shard_tag = f" [shard {shard_index}/{num_shards}]" if sharded else ""
    typer.echo(
        f"Diarizing {len(resolved)} video(s) from {audio_root}{shard_tag} (model={model}).",
        err=True,
    )
    diarizer = Diarizer(model=model)

    def _per_video() -> Iterator[tuple[str, list[SpeakerTurn]]]:
        for doc_id, src in tqdm(resolved, unit="video", smoothing=0.05):
            try:
                turns = diarizer.diarize(src, ffmpeg_timeout=ffmpeg_timeout)
            except Exception as e:  # noqa: BLE001 — one bad video must not kill the batch
                logger.warning("diarization failed: %s (%s) — %s", doc_id, src, e)
                continue
            yield doc_id, turns

    n_turns = write_speaker_turns(turns_path, _per_video(), create=not turns_exists)
    typer.echo(f"  wrote {n_turns} turn(s) across {len(resolved)} video(s).", err=True)

    # One-time scalar BTREE index on doc_id — speeds the per-video lookup the
    # backend's GET /api/diarization/{doc_id} does at full-corpus scale. Built
    # once after the batch loop (not per-append), idempotent via replace=True,
    # and only when the table actually has rows. Mirrors build_topics.py: the
    # index is an optimization, never required, so a failure just logs a skip.
    # Shards skip the index — it is (re)built once on the canonical table by
    # `merge-speaker-turns`. A single (non-shard) run builds it inline as before.
    if not sharded and turns_path.exists():
        turns_tbl = db.open_table(table_name)
        if turns_tbl.count_rows() > 0:
            try:
                turns_tbl.create_scalar_index("doc_id", index_type="BTREE", replace=True)
                logger.info("built BTREE scalar index on speaker_turns.doc_id")
            except Exception as e:  # noqa: BLE001 — the index is an optimization, never required
                logger.debug("scalar index (speaker_turns.doc_id) skipped: %s", e)


@app.command("merge-speaker-turns")
def cmd_merge_speaker_turns(
    ctx: typer.Context,
    drop_shards: Annotated[
        bool,
        typer.Option(
            "--drop-shards/--keep-shards",
            help="Delete the speaker_turns_shard* staging tables after a successful merge.",
        ),
    ] = True,
) -> None:
    """Fold ``speaker_turns_shard*.lance`` staging tables into ``speaker_turns.lance``.

    The sharded ``extract-speaker-turns`` workers each write a disjoint slice to
    their own staging table; this concatenates them (plus any existing canonical
    rows, which win on a ``doc_id`` collision so re-running is safe), overwrites
    ``speaker_turns``, rebuilds the ``doc_id`` BTREE index the backend's
    ``GET /api/diarization/{doc_id}`` relies on, and drops the staging tables.
    """
    # NOTE (future / scale): this is a read-all + overwrite merge — correct and
    # effectively instant at our size (speaker_turns is ~10^5 rows of 5 tiny
    # columns), but it rewrites the whole canonical table each call. The native
    # Lance distributed-write path — each worker `lance.fragment.write_fragments`
    # into the canonical dataset, collect the `FragmentMetadata`, then one
    # `LanceOperation.Append` commit — avoids the rewrite and is the cleaner,
    # safer pattern to adopt if this table grows large or merges get frequent.
    # We keep staging-table + overwrite for now because per-table per-video
    # appends give crash durability (a dead worker resumes mid-shard), which the
    # commit-once-at-the-end fragment path would sacrifice.
    import lance
    import lancedb
    import pyarrow as pa

    from ..model.schema import SPEAKER_TURNS_SCHEMA, SPEAKER_TURNS_STORAGE_VERSION

    cfg: CliContext = ctx.obj
    db = lancedb.connect(str(cfg.db))
    existing_tables = db.list_tables().tables
    shard_names = sorted(t for t in existing_tables if t.startswith("speaker_turns_shard"))
    if not shard_names:
        typer.echo("No speaker_turns_shard* staging tables found — nothing to merge.", err=True)
        return

    main_path = cfg.db / "speaker_turns.lance"
    # Canonical rows first so an already-merged doc_id wins over a shard's copy.
    sources = (["speaker_turns"] if "speaker_turns" in existing_tables else []) + shard_names

    seen: set[str] = set()
    parts: list[pa.Table] = []
    for name in sources:
        ds = lance.dataset(str(cfg.db / f"{name}.lance"))
        if ds.count_rows() == 0:
            continue
        tbl = ds.to_table().cast(SPEAKER_TURNS_SCHEMA)
        doc_col = tbl["doc_id"].to_pylist()
        keep = pa.array([d not in seen for d in doc_col], pa.bool_())
        parts.append(tbl.filter(keep))
        seen.update(doc_col)

    if not parts:
        typer.echo("All source tables were empty — nothing to merge.", err=True)
        return

    merged = pa.concat_tables(parts)
    lance.write_dataset(
        merged,
        str(main_path),
        mode="overwrite",
        data_storage_version=SPEAKER_TURNS_STORAGE_VERSION,
    )
    typer.echo(
        f"  merged → speaker_turns.lance: {len(seen):,} video(s), {merged.num_rows:,} turn(s).",
        err=True,
    )

    turns_tbl = db.open_table("speaker_turns")
    if turns_tbl.count_rows() > 0:
        try:
            turns_tbl.create_scalar_index("doc_id", index_type="BTREE", replace=True)
            logger.info("rebuilt BTREE scalar index on speaker_turns.doc_id")
        except Exception as e:  # noqa: BLE001 — the index is an optimization, never required
            logger.debug("scalar index (speaker_turns.doc_id) skipped: %s", e)

    if drop_shards:
        for name in shard_names:
            db.drop_table(name)
        typer.echo(f"  dropped {len(shard_names)} staging table(s).", err=True)


def _speaker_embeddings_indexes(emb_tbl: lancedb.table.Table) -> None:
    """(Re)build the canonical ``speaker_embeddings`` indexes: BTREE doc_id + vector.

    Shared by ``embed-speaker-turns`` (single-run) and ``merge-speaker-embeddings``.
    """
    from ..features.engine import ensure_vector_index

    try:
        emb_tbl.create_scalar_index("doc_id", index_type="BTREE", replace=True)
        logger.info("built BTREE scalar index on speaker_embeddings.doc_id")
    except Exception as e:  # noqa: BLE001 — the index is an optimization, never required
        logger.debug("scalar index (speaker_embeddings.doc_id) skipped: %s", e)
    # 256-d voice vectors → 16 sub-vectors (16 dims each); skips itself while
    # the table is smaller than num_partitions (flat search is fine until then).
    ensure_vector_index(emb_tbl, "embedding", num_partitions=256, num_sub_vectors=16)


@app.command("embed-speaker-turns")
def cmd_embed_speaker_turns(
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
    model: Annotated[
        str,
        typer.Option("--model", help="HF model id whose 'embedding' subfolder is loaded."),
    ] = "pyannote/speaker-diarization-community-1",
    min_turn_duration: Annotated[
        float,
        typer.Option(
            "--min-turn-duration",
            help="Skip turns shorter than this (seconds); the encoder is unreliable below ~0.5 s.",
        ),
    ] = 0.5,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", help="Turn waveforms per encoder forward pass."),
    ] = 32,
    device: Annotated[
        str,
        typer.Option("--device", help="torch device for the encoder: auto | cpu | cuda[:N]."),
    ] = "auto",
    ffmpeg_timeout: Annotated[
        float,
        typer.Option("--ffmpeg-timeout", help="Per-video wav-extraction timeout (s)."),
    ] = 1800.0,
    only_null: Annotated[
        bool,
        typer.Option(
            "--only-null/--all",
            help="Resumable: skip videos that already have embeddings (--all rebuilds clean).",
        ),
    ] = True,
    limit: Annotated[
        int,
        typer.Option(
            "--limit",
            help="Debug: embed only the first N videos (0 = no limit). DO use this.",
        ),
    ] = 0,
    num_shards: Annotated[
        int,
        typer.Option(
            "--num-shards",
            help=(
                "Split the corpus across N parallel workers (1 = no sharding). "
                "Launch N processes, one per --shard-index, each writing a disjoint "
                "slice to speaker_embeddings_shard{i}.lance; fold them back with "
                "`raudio merge-speaker-embeddings`."
            ),
        ),
    ] = 1,
    shard_index: Annotated[
        int,
        typer.Option(
            "--shard-index",
            help="This worker's shard in [0, num-shards). Only used when --num-shards > 1.",
        ),
    ] = 0,
) -> None:
    """Embed each diarized turn's voice → ``speaker_embeddings.lance`` (NEW append-only table).

    Reads the canonical ``speaker_turns`` table, resolves each video's source MP4
    under ``--audio-root`` (via the ``chunks`` doc_id → audio_path mapping), decodes
    it once to 16 kHz mono WAV, slices the turn spans, and batch-embeds them with
    pyannote community-1's internal WeSpeaker-ResNet34 encoder (256-d, L2-normalized
    before storing). One table append per video; resumable at video granularity via
    ``--only-null``. Turns shorter than ``--min-turn-duration`` are skipped.
    """
    import lancedb
    from tqdm import tqdm

    from ..ingest.audio import resolve_source
    from ..media.diarize import _extract_wav_16k_mono, existing_doc_ids, shard_of
    from ..media.voiceprint import TurnSpan, VoiceEncoder, write_speaker_embeddings

    cfg: CliContext = ctx.obj
    db = lancedb.connect(str(cfg.db))
    _require_table(db, cfg.table, cfg.db)
    chunks_tbl = db.open_table(cfg.table)

    if num_shards < 1:
        raise typer.BadParameter("--num-shards must be >= 1")
    sharded = num_shards > 1
    if sharded and not 0 <= shard_index < num_shards:
        raise typer.BadParameter(f"--shard-index must be in [0, {num_shards})")
    if batch_size < 1:
        raise typer.BadParameter("--batch-size must be >= 1")

    existing_tables = db.list_tables().tables
    if "speaker_turns" not in existing_tables:
        _die(
            f"Table 'speaker_turns' not found in {cfg.db} — run `raudio extract-speaker-turns` "
            "first (and `raudio merge-speaker-turns` to fold its shards into the canonical "
            "table; this command only reads the canonical speaker_turns)."
        )

    # Shards stage to their own table; `merge-speaker-embeddings` folds them into
    # the canonical `speaker_embeddings` afterwards (separate tables avoid
    # concurrent-write commit conflicts that N appenders to one table would hit).
    table_name = f"speaker_embeddings_shard{shard_index}" if sharded else "speaker_embeddings"
    emb_path = cfg.db / f"{table_name}.lance"
    main_path = cfg.db / "speaker_embeddings.lance"
    emb_exists = table_name in existing_tables

    if emb_exists and not only_null:
        typer.echo(f"  --all: dropping existing {table_name} for a clean rebuild.", err=True)
        db.drop_table(table_name)
        emb_exists = False

    # Resume: skip videos this table already has and — in shard mode — anything
    # already merged into the canonical table.
    already: set[str] = set()
    if only_null and emb_exists:
        already |= existing_doc_ids(emb_path)
    if sharded and "speaker_embeddings" in existing_tables:
        already |= existing_doc_ids(main_path)
    if already:
        typer.echo(f"  {len(already):,} video(s) already embedded — skipping.", err=True)

    # All turns, grouped per video (the whole table is ~10^5 tiny rows).
    turns_tbl = db.open_table("speaker_turns")
    turn_rows = (
        turns_tbl.search()
        .select(["doc_id", "turn_id", "speaker_label", "start", "end"])
        .limit(turns_tbl.count_rows())
        .to_list()
    )
    turns_by_doc: dict[str, list[TurnSpan]] = {}
    for r in turn_rows:
        turns_by_doc.setdefault(r["doc_id"], []).append(
            TurnSpan(
                turn_id=int(r["turn_id"]),
                speaker_label=r["speaker_label"],
                start=float(r["start"]),
                end=float(r["end"]),
            )
        )
    for doc_turns in turns_by_doc.values():
        doc_turns.sort(key=lambda t: t.turn_id)

    # doc_id → audio_path from chunks (same mapping extract-speaker-turns used).
    chunk_rows = (
        chunks_tbl.search()
        .select(["doc_id", "audio_path"])
        .limit(chunks_tbl.count_rows())
        .to_list()
    )
    audio_path_of: dict[str, str] = {}
    for r in chunk_rows:
        audio_path_of.setdefault(r["doc_id"], r["audio_path"])

    docs = [d for d in turns_by_doc if d not in already and d in audio_path_of]
    if sharded:
        docs = [d for d in docs if shard_of(d, num_shards) == shard_index]
    docs.sort()
    if limit > 0:
        docs = docs[:limit]
        typer.echo(f"  --limit {limit} → restricting to first {len(docs)} video(s).", err=True)
    if not docs:
        typer.echo("Nothing to embed.", err=True)
        return

    # Resolve sources up front so a missing MP4 doesn't waste a model load.
    resolved: list[tuple[str, Path]] = []
    missing = 0
    for doc_id in docs:
        src = resolve_source(audio_path_of[doc_id], audio_root)
        if src is None:
            missing += 1
            continue
        resolved.append((doc_id, src))
    if missing:
        typer.echo(
            f"  warning: {missing} video(s) had no resolvable source MP4 — skipped.", err=True
        )
    if not resolved:
        typer.echo("Nothing embeddable.", err=True)
        return

    shard_tag = f" [shard {shard_index}/{num_shards}]" if sharded else ""
    typer.echo(
        f"Embedding turns for {len(resolved)} video(s) from {audio_root}{shard_tag} "
        f"(model={model}, min_turn_duration={min_turn_duration}s).",
        err=True,
    )
    encoder = VoiceEncoder(model=model, device=device)

    def _per_video() -> Iterator[tuple[str, list[TurnSpan], np.ndarray]]:
        for doc_id, src in tqdm(resolved, unit="video", smoothing=0.05):
            try:
                with tempfile.TemporaryDirectory(prefix="raudio-voice-") as tmp:
                    wav = Path(tmp) / "audio_16k_mono.wav"
                    _extract_wav_16k_mono(src, wav, timeout=ffmpeg_timeout)
                    kept, embeddings = encoder.embed_turns(
                        wav,
                        turns_by_doc[doc_id],
                        batch_size=batch_size,
                        min_duration=min_turn_duration,
                    )
            except Exception as e:  # noqa: BLE001 — one bad video must not kill the batch
                logger.warning("voice embedding failed: %s (%s) — %s", doc_id, src, e)
                continue
            yield doc_id, kept, embeddings

    n_embeddings = write_speaker_embeddings(emb_path, _per_video(), create=not emb_exists)
    typer.echo(f"  wrote {n_embeddings} embedding(s) across {len(resolved)} video(s).", err=True)

    # Indexes on the canonical table only — shards defer to
    # `merge-speaker-embeddings`, which rebuilds them after the fold.
    if not sharded and emb_path.exists():
        emb_tbl = db.open_table(table_name)
        if emb_tbl.count_rows() > 0:
            _speaker_embeddings_indexes(emb_tbl)


@app.command("merge-speaker-embeddings")
def cmd_merge_speaker_embeddings(
    ctx: typer.Context,
    drop_shards: Annotated[
        bool,
        typer.Option(
            "--drop-shards/--keep-shards",
            help="Delete the speaker_embeddings_shard* staging tables after a successful merge.",
        ),
    ] = True,
) -> None:
    """Fold ``speaker_embeddings_shard*.lance`` staging tables into ``speaker_embeddings.lance``.

    The sharded ``embed-speaker-turns`` workers each write a disjoint slice to
    their own staging table; this concatenates them (plus any existing canonical
    rows, which win on a ``doc_id`` collision so re-running is safe), overwrites
    ``speaker_embeddings``, rebuilds the ``doc_id`` BTREE + vector indexes, and
    drops the staging tables. Mirrors ``merge-speaker-turns`` (see its NOTE on
    the fragment-commit alternative if this table ever gets large).
    """
    import lance
    import lancedb
    import pyarrow as pa

    from ..model.schema import SPEAKER_EMBEDDINGS_SCHEMA, SPEAKER_EMBEDDINGS_STORAGE_VERSION

    cfg: CliContext = ctx.obj
    db = lancedb.connect(str(cfg.db))
    existing_tables = db.list_tables().tables
    shard_names = sorted(t for t in existing_tables if t.startswith("speaker_embeddings_shard"))
    if not shard_names:
        typer.echo(
            "No speaker_embeddings_shard* staging tables found — nothing to merge.", err=True
        )
        return

    main_path = cfg.db / "speaker_embeddings.lance"
    # Canonical rows first so an already-merged doc_id wins over a shard's copy.
    sources = (
        ["speaker_embeddings"] if "speaker_embeddings" in existing_tables else []
    ) + shard_names

    seen: set[str] = set()
    parts: list[pa.Table] = []
    for name in sources:
        ds = lance.dataset(str(cfg.db / f"{name}.lance"))
        if ds.count_rows() == 0:
            continue
        tbl = ds.to_table().cast(SPEAKER_EMBEDDINGS_SCHEMA)
        doc_col = tbl["doc_id"].to_pylist()
        keep = pa.array([d not in seen for d in doc_col], pa.bool_())
        parts.append(tbl.filter(keep))
        seen.update(doc_col)

    if not parts:
        typer.echo("All source tables were empty — nothing to merge.", err=True)
        return

    merged = pa.concat_tables(parts)
    lance.write_dataset(
        merged,
        str(main_path),
        mode="overwrite",
        data_storage_version=SPEAKER_EMBEDDINGS_STORAGE_VERSION,
    )
    typer.echo(
        f"  merged → speaker_embeddings.lance: {len(seen):,} video(s), "
        f"{merged.num_rows:,} embedding(s).",
        err=True,
    )

    emb_tbl = db.open_table("speaker_embeddings")
    if emb_tbl.count_rows() > 0:
        _speaker_embeddings_indexes(emb_tbl)

    if drop_shards:
        for name in shard_names:
            db.drop_table(name)
        typer.echo(f"  dropped {len(shard_names)} staging table(s).", err=True)


@app.command("build-speakers")
def cmd_build_speakers(ctx: typer.Context) -> None:
    """Aggregate per-turn voice embeddings → ``speakers.lance`` (overwrite).

    Groups the canonical ``speaker_embeddings`` by ``(doc_id, speaker_label)`` and
    writes one row per local speaker: turn count, total speech duration, and the
    duration-weighted mean of the turn embeddings re-L2-normalized (the per-speaker
    voiceprint the backend's ``speaker`` anchor reads). ``speaker_cluster`` starts
    at -1 for the later global-clustering pass; ``speaker_name`` starts NULL. The
    table is tiny, so each run rebuilds it wholesale.
    """
    import lance
    import lancedb
    import numpy as np
    import pyarrow as pa

    from ..media.voiceprint import duration_weighted_centroid
    from ..model.schema import SPEAKERS_SCHEMA, SPEAKERS_STORAGE_VERSION, VOICE_EMBED_DIM

    cfg: CliContext = ctx.obj
    db = lancedb.connect(str(cfg.db))
    if "speaker_embeddings" not in db.list_tables().tables:
        _die(
            f"Table 'speaker_embeddings' not found in {cfg.db} — run "
            "`raudio embed-speaker-turns` first (and `raudio merge-speaker-embeddings` "
            "if it ran sharded)."
        )

    ds = lance.dataset(str(cfg.db / "speaker_embeddings.lance"))
    tbl = ds.to_table(columns=["doc_id", "speaker_label", "duration", "embedding"])
    if tbl.num_rows == 0:
        _die("speaker_embeddings is empty — run `raudio embed-speaker-turns` first.")

    doc_ids = tbl["doc_id"].to_pylist()
    labels = tbl["speaker_label"].to_pylist()
    durations = tbl["duration"].to_pylist()
    vectors = (
        tbl["embedding"]
        .combine_chunks()
        .flatten()
        .to_numpy(zero_copy_only=False)
        .reshape(-1, VOICE_EMBED_DIM)
        .astype(np.float32)
    )

    groups: dict[tuple[str, str], list[int]] = {}
    for i, key in enumerate(zip(doc_ids, labels, strict=True)):
        groups.setdefault(key, []).append(i)

    out_keys: list[tuple[str, str]] = []
    out_n_turns: list[int] = []
    out_totals: list[float] = []
    centroids: list[np.ndarray] = []
    for (doc_id, label), idxs in sorted(groups.items()):
        turn_durations = [durations[i] for i in idxs]
        try:
            centroid = duration_weighted_centroid(vectors[idxs], turn_durations)
        except ValueError as e:
            logger.warning("skipping speaker %s/%s: %s", doc_id, label, e)
            continue
        out_keys.append((doc_id, label))
        out_n_turns.append(len(idxs))
        out_totals.append(float(sum(turn_durations)))
        centroids.append(centroid)

    if not centroids:
        _die("No speaker centroid could be computed — speaker_embeddings looks corrupt.")

    n = len(centroids)
    flat = pa.array(np.concatenate(centroids), pa.float32())
    speakers = pa.table(
        {
            "doc_id": pa.array([d for d, _ in out_keys], pa.string()),
            "speaker_label": pa.array([s for _, s in out_keys], pa.string()),
            "n_turns": pa.array(out_n_turns, pa.int32()),
            "total_duration": pa.array(out_totals, pa.float32()),
            "embedding": pa.FixedSizeListArray.from_arrays(flat, VOICE_EMBED_DIM),
            "speaker_cluster": pa.array([-1] * n, pa.int32()),
            "speaker_name": pa.array([None] * n, pa.string()),
        },
        schema=SPEAKERS_SCHEMA,
    )
    lance.write_dataset(
        speakers,
        str(cfg.db / "speakers.lance"),
        mode="overwrite",
        data_storage_version=SPEAKERS_STORAGE_VERSION,
    )
    typer.echo(
        f"  built speakers.lance: {n:,} speaker(s) across "
        f"{len({d for d, _ in groups}):,} video(s).",
        err=True,
    )

    speakers_tbl = db.open_table("speakers")
    try:
        speakers_tbl.create_scalar_index("doc_id", index_type="BTREE", replace=True)
        logger.info("built BTREE scalar index on speakers.doc_id")
    except Exception as e:  # noqa: BLE001 — the index is an optimization, never required
        logger.debug("scalar index (speakers.doc_id) skipped: %s", e)


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
