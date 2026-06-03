"""Concrete feature columns + the ``FEATURES`` registry.

Each column has a small **client-injectable** function (``embed_text_column``,
``caption_column``, …) that wraps the type-agnostic engine with one model client
— this is the seam tests drive with an offline fake. The :data:`FEATURES`
registry maps a name to a :class:`Feature` whose ``run`` builds the production
client from a server URL and calls that function; the ``raudio feature <name>``
CLI is a thin loop over this dict, so adding a column is one entry here.
"""

from __future__ import annotations

import logging

# Path + Callable are imported at runtime (not under TYPE_CHECKING) because the
# Pydantic ``Feature`` model resolves its ``run`` field annotation at class-build time.
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pyarrow as pa
from pydantic import BaseModel, ConfigDict

from ..model.schema import EMBED_DIM
from .engine import ensure_fts_index, ensure_vector_index, upsert_blob_column, upsert_scan_column

if TYPE_CHECKING:
    import lancedb

    from ..vllm.caption import CaptionClient
    from ..vllm.embedding import EmbeddingClient
    from ..vllm.summarize import SummarizeClient

logger = logging.getLogger(__name__)

TEXT_EMBED_COLUMN = "text_embedding"
FRAME_EMBED_COLUMN = "frame_embedding"
SUMMARY_COLUMN = "summary"
CAPTION_COLUMN = "caption"
CAPTION_EMBED_COLUMN = "caption_embedding"

CHUNK_KEYS = ["doc_id", "speech_id", "chunk_id"]
# chunk_frames adds frame_idx to the key (a chunk may hold several frames).
FRAME_KEYS = ["doc_id", "speech_id", "chunk_id", "frame_idx"]
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
        return _vectors_to_arrow(
            client.embed_text([t or "" for t in batch.column("text").to_pylist()])
        )

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
        return pa.array(
            client.summarize([t or "" for t in batch.column("text").to_pylist()]), pa.string()
        )

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


def embed_caption_column(
    frames_path: str | Path,
    *,
    client: EmbeddingClient,
    batch_rows: int = 256,
    checkpoint_file: str | Path | None = None,
    overwrite: bool = False,
    progress: Callable[[int], None] | None = None,
) -> int:
    """Attach ``caption_embedding`` (2048-d) to chunk_frames from the ``caption`` text.

    The text counterpart to ``frame_embedding``: it embeds each frame's Swedish
    caption string (produced by :func:`caption_column`) into the same shared
    2048-d space, so a text query can retrieve frames by *what the scene depicts*
    (``mode=scene``), complementing the raw image-similarity ``frame_embedding``.
    Reads the existing ``caption`` column — it never re-reads or re-extracts the
    frame JPEGs. Run ``raudio feature caption`` first.
    """
    import lance

    ds = lance.dataset(str(frames_path))
    if CAPTION_COLUMN not in ds.schema.names:
        raise ValueError(
            f"'{CAPTION_COLUMN}' column missing on {frames_path} — run "
            f"`raudio feature caption` before `caption_embedding`."
        )

    def compute(batch: pa.RecordBatch) -> pa.Array:
        captions = [c or "" for c in batch.column(CAPTION_COLUMN).to_pylist()]
        return _vectors_to_arrow(client.embed_text(captions))

    return upsert_scan_column(
        frames_path,
        name=CAPTION_EMBED_COLUMN,
        output_type=VECTOR_TYPE,
        key_columns=FRAME_KEYS,
        read_columns=[CAPTION_COLUMN],
        compute=compute,
        batch_rows=batch_rows,
        checkpoint_file=checkpoint_file,
        overwrite=overwrite,
        progress=progress,
    )


def chunk_frame_embedding_column(
    chunks_path: str | Path,
    frames_path: str | Path,
    *,
    column: str = FRAME_EMBED_COLUMN,
    frame_idx: int = 0,
    batch_rows: int = 4096,
    overwrite: bool = False,
    progress: Callable[[int], None] | None = None,
) -> int:
    """Attach a CHUNK-level copy of a per-frame vector ``column`` to ``chunks``.

    The visual/caption atlases project a chunk-level vector, but the source
    (``frame_embedding`` / ``caption_embedding``) lives PER-FRAME on
    ``chunk_frames`` (keyed by ``…/frame_idx``). This is a pure Lance scan+join —
    NO re-embedding: it reads the representative frame's (``frame_idx=0``, the
    same frame the UI/captions/``/chunk-frame`` already treat as canonical)
    ``column`` and attaches it to ``chunks`` keyed on
    ``(doc_id, speech_id, chunk_id)`` via ``add_columns``.

    Returns the number of chunk rows that received a vector (a chunk with no
    matching representative frame stays ``NULL``). Raises if ``chunk_frames``
    lacks ``column`` (run the matching ``raudio feature`` step first).
    """
    import lance

    frames_ds = lance.dataset(str(frames_path))
    if column not in frames_ds.schema.names:
        raise ValueError(
            f"'{column}' column missing on {frames_path} — run the matching "
            f"`raudio feature {column}` step before the chunk-level join."
        )

    chunks_ds = lance.dataset(str(chunks_path))
    if column in chunks_ds.schema.names:
        if not overwrite:
            logger.info("%s already on chunks — nothing to do (pass overwrite=True)", column)
            return 0
        chunks_ds.drop_columns([column])
        chunks_ds = lance.dataset(str(chunks_path))

    # Build a {chunk key → representative-frame vector} map from one filtered scan.
    rep = frames_ds.to_table(columns=[*CHUNK_KEYS, column], filter=f"frame_idx = {int(frame_idx)}")
    vec_by_key: dict[tuple[str, int, int], list[float]] = {}
    docs = rep.column("doc_id").to_pylist()
    speeches = rep.column("speech_id").to_pylist()
    chunk_ids = rep.column("chunk_id").to_pylist()
    vectors = rep.column(column).to_pylist()
    for d, s, c, v in zip(docs, speeches, chunk_ids, vectors, strict=True):
        if v is not None:
            vec_by_key[(d, int(s), int(c))] = v
    logger.info(
        "loaded %d representative-frame %s vector(s) (frame_idx=%d) for the chunk-level join",
        len(vec_by_key),
        column,
        frame_idx,
    )

    schema = pa.schema([pa.field(column, VECTOR_TYPE, nullable=True)])
    matched = 0

    @lance.batch_udf(output_schema=schema)
    def attach(batch: pa.RecordBatch) -> pa.RecordBatch:
        nonlocal matched
        bdocs = batch.column("doc_id").to_pylist()
        bspeech = batch.column("speech_id").to_pylist()
        bchunk = batch.column("chunk_id").to_pylist()
        values: list[list[float] | None] = []
        for d, s, c in zip(bdocs, bspeech, bchunk, strict=True):
            v = vec_by_key.get((d, int(s), int(c)))
            if v is not None:
                matched += 1
            values.append(v)
        out = pa.array(values, type=VECTOR_TYPE)
        if progress is not None:
            progress(batch.num_rows)
        return pa.RecordBatch.from_arrays([out], names=[column])

    logger.info("attaching chunk-level %s via add_columns", column)
    chunks_ds.add_columns(attach, read_columns=CHUNK_KEYS, batch_size=batch_rows)
    return matched


# ─────────────────────────────── Registry ───────────────────────────────────


class FeatureRunOptions(BaseModel):
    """Knobs the ``raudio feature <name>`` CLI passes into a feature's ``run``."""

    url: str | None = None  # model server base URL; None → the feature's own default
    # Generative-feature overrides (caption): None → the client's own default
    # (Gemma 4 / the Swedish instruction). Ignored by the embedding features.
    model: str | None = None
    instruction: str | None = None
    batch_rows: int = 256
    overwrite: bool = False  # --all: drop and rebuild rather than fill-NULL
    create_index: bool = True  # vector features only
    num_partitions: int = 256
    num_sub_vectors: int = 64
    checkpoint: Path | None = None
    # Atlas projection space — routes the 'atlas' feature, ignored by the rest:
    # 'text' (text_embedding → atlas_*), 'visual' (chunk frame_embedding →
    # atlas_img_*), or 'caption' (chunk caption_embedding → atlas_cap_*).
    space: Literal["text", "visual", "caption"] = "text"


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
    from ..vllm.caption import (
        CAPTION_INSTRUCTION,
        CAPTION_MODEL,
        DEFAULT_CAPTION_URL,
        VLLMCaptionClient,
    )

    # model/instruction fall back to the client's own defaults (Gemma 4 / Swedish)
    # when the CLI didn't override them.
    client = VLLMCaptionClient(
        opts.url or DEFAULT_CAPTION_URL,
        model=opts.model or CAPTION_MODEL,
        instruction=opts.instruction or CAPTION_INSTRUCTION,
    )
    n = caption_column(
        db_path / "chunk_frames.lance",
        client=client,
        batch_rows=opts.batch_rows,
        checkpoint_file=opts.checkpoint,
        overwrite=opts.overwrite,
        progress=progress,
    )
    # Tantivy FTS index so captions are keyword-searchable (mode=scene, keyword).
    if opts.create_index:
        ensure_fts_index(_open_table(db_path, "chunk_frames"), CAPTION_COLUMN)
    return n


def _run_caption_embedding(
    db_path: Path, opts: FeatureRunOptions, progress: Callable[[int], None] | None
) -> int:
    from ..vllm.embedding import DEFAULT_EMBED_URL, VLLMEmbeddingClient

    client = VLLMEmbeddingClient(opts.url or DEFAULT_EMBED_URL)
    n = embed_caption_column(
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
            CAPTION_EMBED_COLUMN,
            num_partitions=opts.num_partitions,
            num_sub_vectors=opts.num_sub_vectors,
        )
    return n


def _run_atlas(
    db_path: Path, opts: FeatureRunOptions, progress: Callable[[int], None] | None
) -> int:
    """Text-space atlas, or route to the visual/caption path per ``--space``."""
    if opts.space == "visual":
        return _run_atlas_visual(db_path, opts, progress)
    if opts.space == "caption":
        return _run_atlas_caption(db_path, opts, progress)
    from .projection import project_atlas_columns

    return project_atlas_columns(
        db_path / "chunks.lance",
        overwrite=opts.overwrite,
        batch_rows=opts.batch_rows,
        progress=progress,
    )


def _run_atlas_visual(
    db_path: Path, opts: FeatureRunOptions, progress: Callable[[int], None] | None
) -> int:
    """Visual-space atlas: chunk-level ``frame_embedding`` → ``atlas_img_*``.

    Two pure-Lance steps, no GPU/re-embedding: (a) ensure a chunk-level
    ``frame_embedding`` exists on ``chunks`` (joined from ``chunk_frames``'
    representative ``frame_idx=0`` vector), then (b) project it into the
    namespaced ``atlas_img_x/atlas_img_y/atlas_img_cluster`` triplet.
    """
    from .projection import project_atlas_columns

    chunks_path = db_path / "chunks.lance"
    frames_path = db_path / "chunk_frames.lance"
    if not frames_path.exists():
        raise ValueError(
            "chunk_frames table missing — run `raudio extract-chunk-frames` "
            "and `raudio feature frame_embedding` before the visual atlas."
        )
    chunk_frame_embedding_column(
        chunks_path,
        frames_path,
        batch_rows=opts.batch_rows,
        overwrite=opts.overwrite,
    )
    return project_atlas_columns(
        chunks_path,
        source_column=FRAME_EMBED_COLUMN,
        column_prefix="atlas_img_",
        overwrite=opts.overwrite,
        batch_rows=opts.batch_rows,
        progress=progress,
    )


def _run_atlas_caption(
    db_path: Path, opts: FeatureRunOptions, progress: Callable[[int], None] | None
) -> int:
    """Caption-space atlas: chunk-level ``caption_embedding`` → ``atlas_cap_*``.

    Mirrors :func:`_run_atlas_visual` but over the frame's Swedish-caption text
    vector instead of the raw image vector: (a) join each chunk's representative
    (``frame_idx=0``) ``caption_embedding`` from ``chunk_frames`` onto ``chunks``,
    then (b) project it into ``atlas_cap_x/atlas_cap_y/atlas_cap_cluster``. So the
    map clusters chunks by what their frame *depicts* (described in language),
    distinct from text (what's said) and visual (raw image similarity).
    """
    from .projection import project_atlas_columns

    chunks_path = db_path / "chunks.lance"
    frames_path = db_path / "chunk_frames.lance"
    if not frames_path.exists():
        raise ValueError(
            "chunk_frames table missing — run `raudio feature caption` and "
            "`raudio feature caption_embedding` before the caption atlas."
        )
    chunk_frame_embedding_column(
        chunks_path,
        frames_path,
        column=CAPTION_EMBED_COLUMN,
        batch_rows=opts.batch_rows,
        overwrite=opts.overwrite,
    )
    return project_atlas_columns(
        chunks_path,
        source_column=CAPTION_EMBED_COLUMN,
        column_prefix="atlas_cap_",
        overwrite=opts.overwrite,
        batch_rows=opts.batch_rows,
        progress=progress,
    )


def _run_topics(
    db_path: Path, opts: FeatureRunOptions, progress: Callable[[int], None] | None
) -> int:
    """Toponymy topic layers — runs in an ISOLATED uv env (scripts/build_topics.py).

    Toponymy pins ``transformers<5``; rather than dragging that into the raudio
    project we shell out to a PEP-723 script whose deps resolve in their own
    sealed env (same isolation rationale as the vLLM servers). The worker reads
    ``chunks.text``/``text_embedding``/``atlas_x``/``atlas_y`` and writes the
    ``topic_l*`` columns + a ``<db>.topics.json`` artifact back. ``opts.url``, if
    set, overrides the LLM (Gemma) endpoint; the embedder defaults to :8001.
    """
    import subprocess

    script = Path(__file__).resolve().parents[3] / "scripts" / "build_topics.py"
    cmd = ["uv", "run", "--no-project", str(script), "--db", str(db_path)]
    if opts.url:
        cmd += ["--llm-url", opts.url]
    # stderr inherits (live Toponymy/keyphrase progress bars); stdout carries the
    # final row count.
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"topics worker failed (exit {proc.returncode})")
    # The isolated worker wrote the topic_l*/doc_topic columns; derive the nested
    # hierarchy from them and store it as JSONB in Lance (for the Tree treemap).
    from .topic_tree import build_topic_tree

    build_topic_tree(db_path)
    lines = (proc.stdout or "").strip().splitlines()
    try:
        return int(lines[-1]) if lines else 0
    except ValueError:
        return 0


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
        description="Gemma 4 Swedish caption of each chunk's frame (one factual sentence).",
        run=_run_caption,
    ),
    "caption_embedding": Feature(
        name="caption_embedding",
        table="chunk_frames",
        description="Qwen3-VL text embedding (2048-d) of each frame's caption for scene search.",
        run=_run_caption_embedding,
    ),
    "atlas": Feature(
        name="atlas",
        table="chunks",
        description="EVōC 2-D layout + clusters (atlas_x/atlas_y/atlas_cluster) for the Atlas view. "
        "Use --space visual for the frame_embedding map (atlas_img_*).",
        run=_run_atlas,
    ),
    "atlas_visual": Feature(
        name="atlas_visual",
        table="chunks",
        description="Visual EVōC map from chunk-level frame_embedding "
        "(atlas_img_x/atlas_img_y/atlas_img_cluster). Same as `atlas --space visual`.",
        run=_run_atlas_visual,
    ),
    "atlas_caption": Feature(
        name="atlas_caption",
        table="chunks",
        description="Caption EVōC map from chunk-level caption_embedding "
        "(atlas_cap_x/atlas_cap_y/atlas_cap_cluster). Same as `atlas --space caption`.",
        run=_run_atlas_caption,
    ),
    "topics": Feature(
        name="topics",
        table="chunks",
        description="Toponymy Swedish topic layers (topic_l0…) named by Gemma 4 over the EVōC atlas "
        "map — runs in an isolated env so transformers<5 never enters this project.",
        run=_run_topics,
    ),
}
