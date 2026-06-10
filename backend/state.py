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

import httpx
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
    # Voice-search tables (built by embed-speaker-turns / build-speakers).
    # Optional like chunk_frames: absent until the voice pipeline has run.
    speaker_embeddings_tbl: Any | None = None  # lancedb.table.Table
    speakers_tbl: Any | None = None  # lancedb.table.Table
    settings: Settings = Field(default_factory=get_settings)
    embedder: Any | None = None  # raudio.vllm.embedding.VLLMEmbeddingClient
    reranker: Any | None = None  # raudio.vllm.reranker.VLLMReranker
    # One HTTP connection pool per process (health pings) — never per-request.
    http: Any | None = None  # httpx.Client
    # Memoized /points payloads keyed on (space, dataset version) — per-app, not
    # module-global, so two app instances (e.g. tests) can't cross-contaminate.
    # Writes are idempotent (same input → same bytes), so a plain dict suffices.
    points_cache: dict[tuple[str, int], bytes] = Field(default_factory=dict)


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
            "opened chunk_frames (%d row(s); has_embeddings=%s)",
            chunk_frames_ds.count_rows(),
            has_embeddings,
        )

    # Voice-search tables — optional, only present after `embed-speaker-turns`
    # (+ `merge-speaker-embeddings`) and `build-speakers`. `speaker_embeddings`
    # holds one 256-d voiceprint per diarized turn; `speakers` one
    # duration-weighted centroid per (doc_id, speaker_label).
    speaker_embeddings_tbl = None
    if (db_path / "speaker_embeddings.lance").exists():
        speaker_embeddings_tbl = db.open_table("speaker_embeddings")
        logger.info("opened speaker_embeddings (%d row(s))", speaker_embeddings_tbl.count_rows())
    speakers_tbl = None
    if (db_path / "speakers.lance").exists():
        speakers_tbl = db.open_table("speakers")
        logger.info("opened speakers (%d row(s))", speakers_tbl.count_rows())

    return AppState(
        db_path=db_path,
        names=names,
        chunks=chunks,
        chunks_ds=chunks_ds,
        docs_ds=docs_ds,
        chunk_frames_tbl=chunk_frames_tbl,
        chunk_frames_ds=chunk_frames_ds,
        speaker_embeddings_tbl=speaker_embeddings_tbl,
        speakers_tbl=speakers_tbl,
        settings=get_settings(),
        http=httpx.Client(),
    )
