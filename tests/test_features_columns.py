"""Chunk-level frame-vector join — the ``chunk_frame_embedding_column`` step.

The visual/caption atlases project a CHUNK-level vector that is copied from the
representative frame (``frame_idx=0``) on ``chunk_frames``, keyed on
``(doc_id, speech_id, chunk_id)``. The join is a silent-corruption risk: if the
key match is wrong, chunks get the wrong vector (or none) and the atlas is subtly
broken with no error. These tests pin the contract — matched count, orphan→NULL,
and the missing-column guard — against real Lance tables (no GPU).
"""

from __future__ import annotations

from pathlib import Path

import lance
import pytest

from fakes import FakeEmbedClient, make_doc
from rmedia.features.columns import chunk_frame_embedding_column, embed_frame_column
from rmedia.ingest.ingest import ingest_many
from rmedia.modalities.av.frames import ExtractedFrame, write_chunk_frames


def _chunk_keys(db: Path) -> list[dict]:
    return (
        lance.dataset(str(db / "chunks.lance"))
        .to_table(columns=["doc_id", "speech_id", "chunk_id"])
        .to_pylist()
    )


def _write_frames_for(db: Path, keys: list[dict]) -> None:
    write_chunk_frames(
        db / "chunk_frames.lance",
        [
            ExtractedFrame(
                doc_id=k["doc_id"],
                speech_id=int(k["speech_id"]),
                chunk_id=int(k["chunk_id"]),
                frame_idx=0,
                time_sec=0.0,
                jpeg_bytes=f"frame-{i}".encode(),
                width=8,
                height=8,
            )
            for i, k in enumerate(keys)
        ],
        create=True,
    )


@pytest.fixture
def db_three_chunks(tmp_path) -> Path:
    """A doc with 3 chunks (ids 0/1/2), no frames written yet."""
    db = tmp_path / "transcripts.lance"
    ingest_many(db, [make_doc("input/d.mp4", ["alpha", "bravo", "charlie"])])
    return db


def test_join_matches_only_chunks_with_a_frame(db_three_chunks) -> None:
    db = db_three_chunks
    keys = _chunk_keys(db)
    # Frames for the first two chunks only → chunk_id 2 is an orphan.
    _write_frames_for(db, keys[:2])
    embed_frame_column(db / "chunk_frames.lance", client=FakeEmbedClient())

    matched = chunk_frame_embedding_column(db / "chunks.lance", db / "chunk_frames.lance")

    assert matched == 2
    rows = (
        lance.dataset(str(db / "chunks.lance"))
        .to_table(columns=["chunk_id", "frame_embedding"])
        .to_pylist()
    )
    by_chunk = {r["chunk_id"]: r["frame_embedding"] for r in rows}
    assert by_chunk[0] is not None and by_chunk[1] is not None  # frame → vector attached
    assert by_chunk[2] is None  # orphan chunk stays NULL, not mis-joined


def test_missing_source_column_raises(db_three_chunks) -> None:
    db = db_three_chunks
    # Frames exist but frame_embedding was never built → the join can't run.
    _write_frames_for(db, _chunk_keys(db))
    with pytest.raises(ValueError, match="frame_embedding"):
        chunk_frame_embedding_column(db / "chunks.lance", db / "chunk_frames.lance")


def test_idempotent_without_overwrite(db_three_chunks) -> None:
    db = db_three_chunks
    _write_frames_for(db, _chunk_keys(db))
    embed_frame_column(db / "chunk_frames.lance", client=FakeEmbedClient())
    chunk_frame_embedding_column(db / "chunks.lance", db / "chunk_frames.lance")
    # Column already present → a second call is a no-op (0 matched) unless overwrite.
    again = chunk_frame_embedding_column(db / "chunks.lance", db / "chunk_frames.lance")
    assert again == 0
