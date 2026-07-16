"""Ray composition root for the AV append stages (frames / diarize / voiceprint).

Module-level factories for the ``APPEND_ROWS`` stages: each Ray actor builds
its own warm tool (ffmpeg is subprocess-per-call; pyannote/WeSpeaker load once
per actor) and turns a batch of source rows into output-table rows. Media bytes
are read from the filesystem inside the actor — per LANCE_MEDIA_MERGE §4.3 only
the small frame JPEGs ride back through Ray Data blocks.

Per-item failures warn and skip (one bad video never kills a batch); loud
failures stay reserved for correctness bugs.
"""

from __future__ import annotations

import logging
import tempfile
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa

from rmedia.core.dataset import create_dataset
from rmedia.core.driver import run_append_rows_stage
from rmedia.core.registry import Stage

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

    from rmedia.ingest.audio import resolve_source
    from rmedia.modalities.av.frames import extract_chunk_frame
    from rmedia.model.schema import CHUNK_FRAMES_SCHEMA

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


def diarize_compute(audio_root: str) -> Callable[[pa.Table], pa.Table]:
    from rmedia.ingest.audio import resolve_source
    from rmedia.modalities.av.diarize import Diarizer
    from rmedia.model.schema import SPEAKER_TURNS_SCHEMA

    diarizer = Diarizer()  # pyannote loads once per actor

    def compute(batch: pa.Table) -> pa.Table:
        tables: list[pa.Table] = []
        for doc_id, audio_path in zip(
            batch["doc_id"].to_pylist(), batch["audio_path"].to_pylist(), strict=True
        ):
            try:
                source = resolve_source(audio_path, Path(audio_root))
                if source is None:
                    raise FileNotFoundError(f"{audio_path} not under {audio_root}")
                turns = diarizer.diarize(source)
            except Exception as exc:  # noqa: BLE001 — per-item skip is the stage contract
                logger.warning("diarization failed for %s: %s", doc_id, exc)
                continue
            if not turns:
                continue
            tables.append(
                pa.table(
                    {
                        "doc_id": pa.array([doc_id] * len(turns), pa.string()),
                        "turn_id": pa.array(list(range(len(turns))), pa.int32()),
                        "speaker_label": pa.array([t.speaker_label for t in turns], pa.string()),
                        "start": pa.array([t.start for t in turns], pa.float32()),
                        "end": pa.array([t.end for t in turns], pa.float32()),
                    },
                    schema=SPEAKER_TURNS_SCHEMA,
                )
            )
        return pa.concat_tables(tables) if tables else _empty(SPEAKER_TURNS_SCHEMA)

    return compute


def voiceprint_compute(audio_root: str, path_by_doc: dict[str, str]) -> Callable[[pa.Table], pa.Table]:
    import numpy as np

    from rmedia.ingest.audio import resolve_source
    from rmedia.modalities.av.diarize import extract_wav_16k_mono
    from rmedia.modalities.av.voiceprint import TurnSpan, VoiceEncoder
    from rmedia.model.schema import SPEAKER_EMBEDDINGS_SCHEMA, VOICE_EMBED_DIM

    encoder = VoiceEncoder()  # WeSpeaker loads once per actor

    def compute(batch: pa.Table) -> pa.Table:
        by_doc: dict[str, list[TurnSpan]] = {}
        for doc_id, turn_id, label, start, end in zip(
            batch["doc_id"].to_pylist(),
            batch["turn_id"].to_pylist(),
            batch["speaker_label"].to_pylist(),
            batch["start"].to_pylist(),
            batch["end"].to_pylist(),
            strict=True,
        ):
            by_doc.setdefault(doc_id, []).append(
                TurnSpan(turn_id=int(turn_id), speaker_label=label, start=float(start), end=float(end))
            )

        tables: list[pa.Table] = []
        for doc_id, turns in by_doc.items():
            try:
                source = resolve_source(path_by_doc[doc_id], Path(audio_root))
                if source is None:
                    raise FileNotFoundError(f"{path_by_doc[doc_id]} not under {audio_root}")
                with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
                    extract_wav_16k_mono(source, Path(tmp.name))
                    kept, embeddings = encoder.embed_turns(Path(tmp.name), turns)
            except Exception as exc:  # noqa: BLE001 — per-item skip is the stage contract
                logger.warning("voiceprint failed for %s: %s", doc_id, exc)
                continue
            if not kept:
                continue
            flat = pa.array(embeddings.astype(np.float32).reshape(-1), pa.float32())
            tables.append(
                pa.table(
                    {
                        "doc_id": pa.array([doc_id] * len(kept), pa.string()),
                        "turn_id": pa.array([t.turn_id for t in kept], pa.int32()),
                        "speaker_label": pa.array([t.speaker_label for t in kept], pa.string()),
                        "start": pa.array([t.start for t in kept], pa.float32()),
                        "end": pa.array([t.end for t in kept], pa.float32()),
                        "duration": pa.array([t.duration for t in kept], pa.float32()),
                        "embedding": pa.FixedSizeListArray.from_arrays(flat, VOICE_EMBED_DIM),
                    },
                    schema=SPEAKER_EMBEDDINGS_SCHEMA,
                )
            )
        return pa.concat_tables(tables) if tables else _empty(SPEAKER_EMBEDDINGS_SCHEMA)

    return compute


def run_append_stage(db_path: str | Path, stage: Stage, *, audio_root: str = "input/sv") -> int:
    """Dispatch an APPEND_ROWS stage to the Ray driver with its AV binding.

    The voiceprint stage reads turns but media lives doc-level, so its source
    read joins ``audio_path`` from the documents table first (small map).
    """
    from rmedia.model.schema import (
        CHUNK_FRAMES_SCHEMA,
        SPEAKER_EMBEDDINGS_SCHEMA,
        SPEAKER_TURNS_SCHEMA,
    )

    # Absolute: a relative root would resolve against the Ray workers'
    # runtime-env working-dir copy, failing every per-item extraction.
    audio_root = str(Path(audio_root).resolve())

    def _doc_paths() -> dict[str, str]:
        import lance

        docs = lance.dataset(str(Path(db_path) / "documents.lance")).to_table(
            columns=["doc_id", "audio_path"]
        )
        return dict(zip(docs["doc_id"].to_pylist(), docs["audio_path"].to_pylist(), strict=True))

    bindings: dict[str, tuple[Callable[[], Callable[[pa.Table], pa.Table]], pa.Schema]] = {
        "extract_frames": (partial(frames_compute, audio_root), CHUNK_FRAMES_SCHEMA),
        "diarize": (partial(diarize_compute, audio_root), SPEAKER_TURNS_SCHEMA),
        "voiceprint": (
            partial(voiceprint_compute, audio_root, _doc_paths() if stage.name == "voiceprint" else {}),
            SPEAKER_EMBEDDINGS_SCHEMA,
        ),
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
