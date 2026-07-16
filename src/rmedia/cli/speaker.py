"""Speaker-diarization pipeline commands: ``extract-speaker-turns`` →
``embed-speaker-turns`` → ``build-speakers`` → ``cluster-speakers`` plus the
two shard-fold merges (``merge-speaker-turns``, ``merge-speaker-embeddings``).

Each handler parses its options, calls one library function (in
:mod:`rmedia.modalities.av.diarize`, :mod:`rmedia.modalities.av.voiceprint`, or
:mod:`rmedia.modalities.av.cluster`), and echoes a summary.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

if TYPE_CHECKING:
    from rmedia.modalities.av.diarize import SpeakerTurn

from ._app import CliContext, _die, _require_table, app

logger = logging.getLogger(__name__)


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

    from rmedia.modalities.av.diarize import (
        Diarizer,
        existing_doc_ids,
        shard_of,
        write_speaker_turns,
    )

    from ..ingest.audio import resolve_source

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
    import lancedb

    from rmedia.modalities.av.voiceprint import fold_shards

    from ..model.schema import SPEAKER_TURNS_SCHEMA, SPEAKER_TURNS_STORAGE_VERSION

    cfg: CliContext = ctx.obj
    db = lancedb.connect(str(cfg.db))

    def _rebuild(turns_tbl: lancedb.table.Table) -> None:
        try:
            turns_tbl.create_scalar_index("doc_id", index_type="BTREE", replace=True)
            logger.info("rebuilt BTREE scalar index on speaker_turns.doc_id")
        except Exception as e:  # noqa: BLE001 — the index is an optimization, never required
            logger.debug("scalar index (speaker_turns.doc_id) skipped: %s", e)

    result = fold_shards(
        db,
        cfg.db,
        "speaker_turns",
        schema=SPEAKER_TURNS_SCHEMA,
        storage_version=SPEAKER_TURNS_STORAGE_VERSION,
        drop_shards=drop_shards,
        rebuild_indexes=_rebuild,
    )
    if result is None:
        typer.echo("No speaker_turns_shard* staging tables found — nothing to merge.", err=True)
        return

    n_videos, n_rows = result
    typer.echo(
        f"  merged → speaker_turns.lance: {n_videos:,} video(s), {n_rows:,} turn(s).",
        err=True,
    )
    if drop_shards:
        typer.echo("  dropped staging table(s).", err=True)


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

    from rmedia.modalities.av.diarize import existing_doc_ids, shard_of
    from rmedia.modalities.av.voiceprint import (
        TurnSpan,
        VoiceEncoder,
        embed_videos,
        speaker_embeddings_indexes,
        write_speaker_embeddings,
    )

    from ..ingest.audio import resolve_source

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

    def _progress(videos: Sequence[tuple[str, Path]]) -> Iterable[tuple[str, Path]]:
        return tqdm(videos, unit="video", smoothing=0.05)

    rows = embed_videos(
        encoder,
        resolved,
        turns_by_doc,
        batch_size=batch_size,
        min_turn_duration=min_turn_duration,
        ffmpeg_timeout=ffmpeg_timeout,
        progress=_progress,
    )
    n_embeddings = write_speaker_embeddings(emb_path, rows, create=not emb_exists)
    typer.echo(f"  wrote {n_embeddings} embedding(s) across {len(resolved)} video(s).", err=True)

    # Indexes on the canonical table only — shards defer to
    # `merge-speaker-embeddings`, which rebuilds them after the fold.
    if not sharded and emb_path.exists():
        emb_tbl = db.open_table(table_name)
        if emb_tbl.count_rows() > 0:
            speaker_embeddings_indexes(emb_tbl)


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
    import lancedb

    from rmedia.modalities.av.voiceprint import fold_shards, speaker_embeddings_indexes

    from ..model.schema import SPEAKER_EMBEDDINGS_SCHEMA, SPEAKER_EMBEDDINGS_STORAGE_VERSION

    cfg: CliContext = ctx.obj
    db = lancedb.connect(str(cfg.db))

    result = fold_shards(
        db,
        cfg.db,
        "speaker_embeddings",
        schema=SPEAKER_EMBEDDINGS_SCHEMA,
        storage_version=SPEAKER_EMBEDDINGS_STORAGE_VERSION,
        drop_shards=drop_shards,
        rebuild_indexes=speaker_embeddings_indexes,
    )
    if result is None:
        typer.echo(
            "No speaker_embeddings_shard* staging tables found — nothing to merge.", err=True
        )
        return

    n_videos, n_rows = result
    typer.echo(
        f"  merged → speaker_embeddings.lance: {n_videos:,} video(s), {n_rows:,} embedding(s).",
        err=True,
    )
    if drop_shards:
        typer.echo("  dropped staging table(s).", err=True)


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
    import lancedb

    from rmedia.modalities.av.voiceprint import build_speakers

    cfg: CliContext = ctx.obj
    db = lancedb.connect(str(cfg.db))
    try:
        n, n_videos = build_speakers(db, cfg.db)
    except ValueError as e:
        _die(str(e))

    typer.echo(
        f"  built speakers.lance: {n:,} speaker(s) across {n_videos:,} video(s).",
        err=True,
    )


@app.command("cluster-speakers")
def cmd_cluster_speakers(
    ctx: typer.Context,
    seed: Annotated[
        int,
        typer.Option("--seed", help="EVoC random_state — the assignment is reproducible."),
    ] = 42,
    min_cluster_size: Annotated[
        int,
        typer.Option(
            "--min-cluster-size",
            help=(
                "EVoC base_min_cluster_size (EVoC's own default is 5); "
                "lower it if known identities land in noise."
            ),
        ),
    ] = 5,
    validate: Annotated[
        bool,
        typer.Option("--validate", help="Check the known same-person pairs and print PASS/FAIL."),
    ] = False,
) -> None:
    """Globally cluster speaker voiceprints → ``speakers.speaker_cluster`` (overwrite).

    Fits :class:`evoc.EVoC` (default ``n_neighbors``; ``-1`` = noise — the same
    estimator idiom as the Atlas projection, but seeded by default: identity
    assignment must be reproducible) over the ``speakers`` embedding matrix and
    rewrites the table wholesale with the new ``speaker_cluster`` column
    (mirrors ``build-speakers`` — the table is tiny). The written assignment is
    the layer the identity-layer selector picks, NOT EVoC's persistence-max
    ``labels_`` (channel-scale here — see the helper). Cluster ids are a
    partition, not stable names: a re-run with other parameters renumbers them.
    """
    import lancedb
    import numpy as np

    from rmedia.modalities.av.cluster import (
        MAX_SAME_DOC_MERGE_RATE,
        cluster_speakers,
        validate_known_identities,
    )

    cfg: CliContext = ctx.obj
    db = lancedb.connect(str(cfg.db))

    try:
        result = cluster_speakers(
            db,
            cfg.db,
            seed=seed,
            min_cluster_size=min_cluster_size,
            on_start=lambda n: typer.echo(
                f"EVoC clustering {n:,} speaker voiceprint(s) "
                f"(seed={seed}, min_cluster_size={min_cluster_size}) …",
                err=True,
            ),
        )
    except ImportError as e:
        _die(str(e))
    except ValueError as e:
        _die(str(e))

    clusters = result.clusters
    if result.fallback:
        typer.echo(
            "  [warn] no EVoC layer met the ≤"
            f"{MAX_SAME_DOC_MERGE_RATE:.0%} within-video false-merge bound — "
            f"falling back to EVoC's own layer choice "
            f"(false-merge {result.fallback_merge_rate:.1%}).",
            err=True,
        )
    elif result.layer_idx is not None and result.merge_rate is not None:
        typer.echo(
            f"  identity layer: {result.layer_idx + 1}/{result.n_layers} "
            f"(fine→coarse), within-video false-merge {result.merge_rate:.1%} "
            f"(EVoC's own persistence-max layer: {result.auto_merge_rate:.1%}).",
            err=True,
        )

    n_noise = int((clusters < 0).sum())
    ids, sizes = np.unique(clusters[clusters >= 0], return_counts=True)
    typer.echo(f"  clusters found: {len(ids):,}", err=True)
    typer.echo(f"  noise (unclustered): {n_noise:,} / {len(clusters):,}", err=True)
    if len(ids):
        order = np.argsort(sizes)[::-1]
        typer.echo(f"  largest cluster: {int(sizes[order[0]]):,} speaker(s)", err=True)
        top = ", ".join(f"{int(ids[i])}: {int(sizes[i])}" for i in order[:10])
        typer.echo(f"  top 10 cluster sizes (id: size): {top}", err=True)

    if validate:
        validate_known_identities(
            cfg.db,
            cfg.table,
            result.speakers,
            clusters,
            echo=lambda msg: typer.echo(msg, err=True),
        )
