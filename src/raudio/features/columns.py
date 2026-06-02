"""Concrete feature columns + the ``FEATURES`` registry.

Each column has a small **client-injectable** function (``embed_text_column``,
``caption_column``, …) that wraps the type-agnostic engine with one model client
— this is the seam tests drive with an offline fake. The :data:`FEATURES`
registry maps a name to a :class:`Feature` whose ``run`` builds the production
client from a server URL and calls that function; the ``raudio feature <name>``
CLI is a thin loop over this dict, so adding a column is one entry here.
"""

from __future__ import annotations

# Path + Callable are imported at runtime (not under TYPE_CHECKING) because the
# Pydantic ``Feature`` model resolves its ``run`` field annotation at class-build time.
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
from pydantic import BaseModel, ConfigDict

from ..model.schema import EMBED_DIM
from .engine import ensure_vector_index, upsert_blob_column, upsert_scan_column

if TYPE_CHECKING:
    import lancedb

    from ..vllm.caption import CaptionClient
    from ..vllm.embedding import EmbeddingClient
    from ..vllm.summarize import SummarizeClient

TEXT_EMBED_COLUMN = "text_embedding"
FRAME_EMBED_COLUMN = "frame_embedding"
SUMMARY_COLUMN = "summary"
CAPTION_COLUMN = "caption"

CHUNK_KEYS = ["doc_id", "speech_id", "chunk_id"]
VECTOR_TYPE = pa.list_(pa.float32(), EMBED_DIM)


def _vectors_to_arrow(vectors: np.ndarray) -> pa.FixedSizeListArray:
    """``(N, EMBED_DIM)`` float32 array → Arrow ``FixedSizeList<float32, EMBED_DIM>``."""
    if vectors.ndim != 2 or vectors.shape[1] != EMBED_DIM:
        raise ValueError(f"expected (N, {EMBED_DIM}) vectors, got {vectors.shape}")
    flat = pa.array(np.ascontiguousarray(vectors, dtype=np.float32).reshape(-1), pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, EMBED_DIM)


# ─────────────────────────── Column population (client-injectable) ──────────


def embed_text_column(
    chunks_path: str | Path,
    *,
    client: EmbeddingClient,
    batch_rows: int = 256,
    checkpoint_file: str | Path | None = None,
    overwrite: bool = False,
    progress: Callable[[int], None] | None = None,
) -> int:
    """Attach ``text_embedding`` (2048-d) to the chunks table from ``text``."""

    def compute(batch: pa.RecordBatch) -> pa.Array:
        return _vectors_to_arrow(client.embed_text([t or "" for t in batch.column("text").to_pylist()]))

    return upsert_scan_column(
        chunks_path,
        name=TEXT_EMBED_COLUMN,
        output_type=VECTOR_TYPE,
        key_columns=CHUNK_KEYS,
        read_columns=["text"],
        compute=compute,
        batch_rows=batch_rows,
        checkpoint_file=checkpoint_file,
        overwrite=overwrite,
        progress=progress,
    )


def embed_frame_column(
    frames_path: str | Path,
    *,
    client: EmbeddingClient,
    batch_rows: int = 256,
    checkpoint_file: str | Path | None = None,
    overwrite: bool = False,
    progress: Callable[[int], None] | None = None,
) -> int:
    """Attach ``frame_embedding`` (2048-d) to the chunk_frames table from ``frame_blob``."""

    def compute(jpegs: list[bytes]) -> pa.Array:
        return _vectors_to_arrow(client.embed_image(jpegs))

    return upsert_blob_column(
        frames_path,
        name=FRAME_EMBED_COLUMN,
        output_type=VECTOR_TYPE,
        blob_column="frame_blob",
        compute=compute,
        batch_rows=batch_rows,
        checkpoint_file=checkpoint_file,
        overwrite=overwrite,
        progress=progress,
    )


def summary_column(
    chunks_path: str | Path,
    *,
    client: SummarizeClient,
    batch_rows: int = 256,
    checkpoint_file: str | Path | None = None,
    overwrite: bool = False,
    progress: Callable[[int], None] | None = None,
) -> int:
    """Attach a one-line ``summary`` string to the chunks table from ``text``."""

    def compute(batch: pa.RecordBatch) -> pa.Array:
        return pa.array(client.summarize([t or "" for t in batch.column("text").to_pylist()]), pa.string())

    return upsert_scan_column(
        chunks_path,
        name=SUMMARY_COLUMN,
        output_type=pa.string(),
        key_columns=CHUNK_KEYS,
        read_columns=["text"],
        compute=compute,
        batch_rows=batch_rows,
        checkpoint_file=checkpoint_file,
        overwrite=overwrite,
        progress=progress,
    )


def caption_column(
    frames_path: str | Path,
    *,
    client: CaptionClient,
    batch_rows: int = 256,
    checkpoint_file: str | Path | None = None,
    overwrite: bool = False,
    progress: Callable[[int], None] | None = None,
) -> int:
    """Attach a ``caption`` string to the chunk_frames table from ``frame_blob``."""

    def compute(jpegs: list[bytes]) -> pa.Array:
        return pa.array(client.caption(jpegs), pa.string())

    return upsert_blob_column(
        frames_path,
        name=CAPTION_COLUMN,
        output_type=pa.string(),
        blob_column="frame_blob",
        compute=compute,
        batch_rows=batch_rows,
        checkpoint_file=checkpoint_file,
        overwrite=overwrite,
        progress=progress,
    )


# ─────────────────────────────── Registry ───────────────────────────────────


class FeatureRunOptions(BaseModel):
    """Knobs the ``raudio feature <name>`` CLI passes into a feature's ``run``."""

    url: str | None = None  # model server base URL; None → the feature's own default
    batch_rows: int = 256
    overwrite: bool = False  # --all: drop and rebuild rather than fill-NULL
    create_index: bool = True  # vector features only
    num_partitions: int = 256
    num_sub_vectors: int = 64
    checkpoint: Path | None = None


class Feature(BaseModel):
    """One derived column: where it lives + how to build it."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    name: str
    table: str  # "chunks" or "chunk_frames"
    description: str
    run: Callable[[Path, FeatureRunOptions, Callable[[int], None] | None], int]


def _open_table(db_path: Path, table: str) -> lancedb.table.Table:
    import lancedb

    return lancedb.connect(str(db_path)).open_table(table)


def _run_text_embedding(
    db_path: Path, opts: FeatureRunOptions, progress: Callable[[int], None] | None
) -> int:
    from ..vllm.embedding import DEFAULT_EMBED_URL, VLLMEmbeddingClient

    client = VLLMEmbeddingClient(opts.url or DEFAULT_EMBED_URL)
    n = embed_text_column(
        db_path / "chunks.lance",
        client=client,
        batch_rows=opts.batch_rows,
        checkpoint_file=opts.checkpoint,
        overwrite=opts.overwrite,
        progress=progress,
    )
    if opts.create_index:
        ensure_vector_index(
            _open_table(db_path, "chunks"),
            TEXT_EMBED_COLUMN,
            num_partitions=opts.num_partitions,
            num_sub_vectors=opts.num_sub_vectors,
        )
    return n


def _run_frame_embedding(
    db_path: Path, opts: FeatureRunOptions, progress: Callable[[int], None] | None
) -> int:
    from ..vllm.embedding import DEFAULT_EMBED_URL, VLLMEmbeddingClient

    client = VLLMEmbeddingClient(opts.url or DEFAULT_EMBED_URL)
    n = embed_frame_column(
        db_path / "chunk_frames.lance",
        client=client,
        batch_rows=opts.batch_rows,
        checkpoint_file=opts.checkpoint,
        overwrite=opts.overwrite,
        progress=progress,
    )
    if opts.create_index:
        ensure_vector_index(
            _open_table(db_path, "chunk_frames"),
            FRAME_EMBED_COLUMN,
            num_partitions=opts.num_partitions,
            num_sub_vectors=opts.num_sub_vectors,
        )
    return n


def _run_summary(
    db_path: Path, opts: FeatureRunOptions, progress: Callable[[int], None] | None
) -> int:
    from ..vllm.summarize import DEFAULT_SUMMARIZE_URL, VLLMSummarizeClient

    client = VLLMSummarizeClient(opts.url or DEFAULT_SUMMARIZE_URL)
    return summary_column(
        db_path / "chunks.lance",
        client=client,
        batch_rows=opts.batch_rows,
        checkpoint_file=opts.checkpoint,
        overwrite=opts.overwrite,
        progress=progress,
    )


def _run_caption(
    db_path: Path, opts: FeatureRunOptions, progress: Callable[[int], None] | None
) -> int:
    from ..vllm.caption import DEFAULT_CAPTION_URL, VLLMCaptionClient

    client = VLLMCaptionClient(opts.url or DEFAULT_CAPTION_URL)
    return caption_column(
        db_path / "chunk_frames.lance",
        client=client,
        batch_rows=opts.batch_rows,
        checkpoint_file=opts.checkpoint,
        overwrite=opts.overwrite,
        progress=progress,
    )


FEATURES: dict[str, Feature] = {
    "text_embedding": Feature(
        name="text_embedding",
        table="chunks",
        description="Qwen3-VL text embedding (2048-d) for semantic search.",
        run=_run_text_embedding,
    ),
    "frame_embedding": Feature(
        name="frame_embedding",
        table="chunk_frames",
        description="Qwen3-VL frame embedding (2048-d) for visual search.",
        run=_run_frame_embedding,
    ),
    "summary": Feature(
        name="summary",
        table="chunks",
        description="One-line LLM summary of each chunk's transcript.",
        run=_run_summary,
    ),
    "caption": Feature(
        name="caption",
        table="chunk_frames",
        description="VLM caption of each chunk's representative frame.",
        run=_run_caption,
    ),
}
