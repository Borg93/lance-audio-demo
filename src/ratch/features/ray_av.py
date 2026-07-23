"""Ray composition root for the AV append stages (frames / diarize / voiceprint).

Pure orchestration: the MODEL actor factories live in their runners
(``runners/{diarize,voiceprint}/actor.py`` — each Ray actor loads its model once,
warm per actor) and are imported lazily here; only the model-free ``frames``
factory (ffmpeg subprocess) stays in this module. Media bytes
are read from the filesystem inside the actor — per LANCE_MEDIA_MERGE §4.3 only
the small frame JPEGs ride back through Ray Data blocks.

Per-item failures warn and skip (one bad video never kills a batch); loud
failures stay reserved for correctness bugs.
"""

from __future__ import annotations

import logging
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa

from ratch.core.dataset import create_dataset
from ratch.core.driver import run_append_rows_stage
from ratch.core.registry import Stage

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

#: Frame sampled this far into a chunk — matches the old extract-chunk-frames
#: policy (the representative frame is the chunk's start).
FRAME_AT_CHUNK_START_S = 0.0


def _empty(schema: pa.Schema) -> pa.Table:
    return pa.table({f.name: pa.array([], type=f.type) for f in schema}, schema=schema)


def frames_compute(audio_root: str) -> Callable[[pa.Table], pa.Table]:
    from lance import blob_array

    from ratch.ingest.audio import resolve_source
    from ratch.modalities.av.frames import extract_chunk_frame
    from ratch.model.schema import CHUNK_FRAMES_SCHEMA

    def compute(batch: pa.Table) -> pa.Table:
        rows: list[tuple[str, int, int, bytes, int, int]] = []
        for doc_id, speech_id, chunk_id, start, audio_path in zip(
            batch["doc_id"].to_pylist(),
            batch["speech_id"].to_pylist(),
            batch["chunk_id"].to_pylist(),
            batch["start"].to_pylist(),
            batch["audio_path"].to_pylist(),
            strict=True,
        ):
            try:
                source = resolve_source(audio_path, Path(audio_root))
                if source is None:
                    raise FileNotFoundError(f"{audio_path} not under {audio_root}")
                jpeg, width, height = extract_chunk_frame(
                    source=source, time_sec=float(start) + FRAME_AT_CHUNK_START_S
                )
                rows.append((doc_id, int(speech_id), int(chunk_id), jpeg, width, height))
            except Exception as exc:  # noqa: BLE001 — per-item skip is the stage contract
                logger.warning("frame extraction failed for %s/%s/%s: %s", doc_id, speech_id, chunk_id, exc)
        if not rows:
            return _empty(CHUNK_FRAMES_SCHEMA)
        return pa.table(
            {
                "doc_id": pa.array([r[0] for r in rows], pa.string()),
                "speech_id": pa.array([r[1] for r in rows], pa.int32()),
                "chunk_id": pa.array([r[2] for r in rows], pa.int32()),
                "frame_idx": pa.array([0] * len(rows), pa.int32()),
                "frame_blob": blob_array([r[3] for r in rows]),
                "frame_mime": pa.array(["image/jpeg"] * len(rows), pa.string()),
                "frame_width": pa.array([r[4] for r in rows], pa.int32()),
                "frame_height": pa.array([r[5] for r in rows], pa.int32()),
            },
            schema=CHUNK_FRAMES_SCHEMA,
        )

    return compute






def run_append_stage(db_path: str | Path, stage: Stage, *, audio_root: str = "input/sv") -> int:
    """Dispatch an APPEND_ROWS stage to the Ray driver with its AV binding."""
    from ratch.model.schema import (
        CHUNK_FRAMES_SCHEMA,
        SPEAKER_EMBEDDINGS_SCHEMA,
        SPEAKER_TURNS_SCHEMA,
    )

    # Absolute paths throughout: a relative path would resolve against the Ray
    # workers' runtime-env working-dir copy, failing every per-item read.
    audio_root = str(Path(audio_root).resolve())
    turns_uri = str((Path(db_path) / "speaker_turns.lance").resolve())

    # Runner-backed stages: the ACTOR factories live in runners/<name>/actor.py
    # (the model's home — see Stage.runner); ratch only composes them here.
    from runners.diarize.actor import diarize_compute
    from runners.voiceprint.actor import voiceprint_compute

    bindings: dict[str, tuple[Callable[[], Callable[[pa.Table], pa.Table]], pa.Schema]] = {
        "extract_frames": (partial(frames_compute, audio_root), CHUNK_FRAMES_SCHEMA),
        "diarize": (partial(diarize_compute, audio_root), SPEAKER_TURNS_SCHEMA),
        "voiceprint": (partial(voiceprint_compute, audio_root, turns_uri), SPEAKER_EMBEDDINGS_SCHEMA),
    }
    if stage.name not in bindings:
        raise ValueError(f"no AV binding for append stage {stage.name!r}")
    factory, output_schema = bindings[stage.name]

    out_uri = str(Path(db_path) / f"{stage.output_table}.lance")
    return run_append_rows_stage(
        db_path,
        stage,
        factory=factory,
        output_schema=output_schema,
        create_output=lambda: create_dataset(out_uri, output_schema),
    )
