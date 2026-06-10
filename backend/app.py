"""App factory + uvicorn launcher for the Lance-backed transcript viewer.

Every endpoint reads from Lance directly — no disk walks, no sidecar JSON.
``media_blob`` is Lance Blob V2 External (URI); ``thumbnail`` and ``frame_blob``
are Blob V2 Inline (bytes); ``alignments_json`` is Lance JSONB. Blob reads use
``ds.take_blobs(..., ids=[rowid])`` (lazy, seekable), so HTTP Range maps to
``seek(start) + read(length)``.

Search modes (`/api/search`): ``fts`` (Tantivy BM25), ``semantic`` (text
vectors), ``visual`` (frame vectors, text or image query), ``hybrid`` (Lance
native FTS+vector RRF), ``all`` (RRF over all three). ``rerank=true`` swaps the
default RRF for the Qwen3-VL cross-encoder.

Run:  raudio serve --db ./transcripts_v2.lance --port 8000
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

from backend.atlas.router import router as atlas_router
from backend.core.config import get_settings
from backend.core.handlers import register_handlers
from backend.core.lifespan import lifespan
from backend.core.middleware import register_middleware
from backend.core.probes import router as probes_router
from backend.diarization.router import router as diarization_router
from backend.graph.router import router as graph_router
from backend.media.router import router as media_router
from backend.search.router import router as search_router
from backend.state import open_resources
from backend.system.router import router as system_router
from backend.topics.router import router as topics_router


def create_app(db_path: str | Path) -> FastAPI:
    """Build the API-only FastAPI app.

    ``db_path`` is the single positional arg the CLI and tests pass; Lance
    handles are opened eagerly here (not in the lifespan) so a bare
    ``TestClient(create_app(db))`` — used without a context manager — still
    has ``app.state.resources``. The lifespan only warms caches + flips the
    readiness flags, which a lifespan-less TestClient rightly skips. We set
    the readiness flags to False here so ``/readyz`` is well-defined even
    without the lifespan.
    """
    settings = get_settings()
    app = FastAPI(title="raudio api", lifespan=lifespan)

    app.state.resources = open_resources(db_path)
    app.state.startup_complete = False
    app.state.shutting_down = False

    register_handlers(app)
    register_middleware(app, settings)

    app.include_router(probes_router)
    app.include_router(search_router)
    app.include_router(media_router)
    app.include_router(system_router)
    app.include_router(atlas_router)
    app.include_router(topics_router)
    app.include_router(diarization_router)
    app.include_router(graph_router)

    return app


def run(
    db_path: str | Path,
    *,
    host: str | None = None,
    port: int | None = None,
) -> None:
    """Start the API with uvicorn. Host/port fall back to ``Settings`` (RAUDIO_HOST/PORT)."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        create_app(db_path),
        host=host or settings.host,
        port=port or settings.port,
        log_level="info",
    )
