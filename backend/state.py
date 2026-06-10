"""Shared, app-wide resources opened once at startup and read via DI.

``open_resources`` is the single place that touches the filesystem at startup
(opening the Lance handles). The result lives on ``app.state.resources`` and is
reached through the ``StateDep`` dependency — routers never touch ``app.state``
directly. The two vLLM client slots stay ``None`` until first use so an
FTS-only deployment never connects to a GPU server.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import lance
import lancedb
from pydantic import BaseModel, ConfigDict, Field

from backend.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


class AppState(BaseModel):
    """Per-app resource handles + lazy vLLM client cache.

    The Lance/lancedb handles aren't Pydantic-validatable, hence
    ``arbitrary_types_allowed``. ``embedder``/``reranker`` are mutable slots that
    :mod:`backend.clients` fills on first use and reuses thereafter. ``settings``
    carries the typed app config (vLLM URLs, host/port, CORS); it defaults via
    ``get_settings()`` so bare constructions in tests still work.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    db_path: Path
    names: list[str]
    chunks: Any  # lancedb.table.Table — the abstract stub omits .search()/.to_lance()
    # `chunks_ds` is `chunks.to_lance()` resolved once at startup. Re-wrapping per
    # request re-seeds the dataset's metadata/index cache, so we share one handle.
    # Defaults to None (like the other Lance handles) so a bare 3-arg construction
    # in tests still validates; ``open_resources`` always sets it in production.
    chunks_ds: Any = None  # lance.LanceDataset — typed Any to match the `chunks` stub
    docs_ds: lance.LanceDataset | None = None
    chunk_frames_tbl: Any | None = None  # lancedb.table.Table
    chunk_frames_ds: lance.LanceDataset | None = None
    settings: Settings = Field(default_factory=get_settings)
    embedder: Any | None = None  # raudio.vllm.embedding.VLLMEmbeddingClient
    reranker: Any | None = None  # raudio.vllm.reranker.VLLMReranker


def open_resources(db_path: str | Path) -> AppState:
    """Open every Lance handle the API serves from. Raises if ``chunks`` is absent."""
    db_path = Path(db_path)
    db = lancedb.connect(str(db_path))
    names = db.list_tables().tables
    logger.info("opened Lance DB %s — tables: %s", db_path, names)

    if "chunks" not in names:
        raise RuntimeError(f"'chunks' table missing in {db_path}")
    chunks = db.open_table("chunks")
    chunks_ds = chunks.to_lance()  # resolve once; reused by every read path

    # `documents` is optional (only present after `ingest --audio-root …`).
    docs_ds = lance.dataset(str(db_path / "documents.lance")) if "documents" in names else None

    # `chunk_frames` holds per-chunk video frames — a separate table (Lance 4.0
    # merge_insert crashes on the wide chunks schema). Optional, only present
    # after `extract-chunk-frames`. Two handles: a lancedb Table for frame-vector
    # search, and a lance.Dataset for blob reads (`take_blobs`).
    chunk_frames_tbl = None
    chunk_frames_ds = None
    chunk_frames_path = db_path / "chunk_frames.lance"
    if chunk_frames_path.exists():
        chunk_frames_tbl = db.open_table("chunk_frames")
        chunk_frames_ds = lance.dataset(str(chunk_frames_path))
        has_embeddings = "frame_embedding" in chunk_frames_ds.schema.names
        logger.info(
            f"opened chunk_frames ({chunk_frames_ds.count_rows()} row(s); "
            f"has_embeddings={has_embeddings})"
        )

    return AppState(
        db_path=db_path,
        names=names,
        chunks=chunks,
        chunks_ds=chunks_ds,
        docs_ds=docs_ds,
        chunk_frames_tbl=chunk_frames_tbl,
        chunk_frames_ds=chunk_frames_ds,
        settings=get_settings(),
    )
