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

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from backend.atlas.router import router as atlas_router
from backend.diarization.router import router as diarization_router
from backend.media.router import router as media_router
from backend.search.router import router as search_router
from backend.state import open_resources
from backend.system.router import router as system_router
from backend.topics.router import router as topics_router
from backend.warmup import warm_caches


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Warm the Lance index + atlas-points caches on startup, off the event loop.

    Resources are opened eagerly in the factory body (so a bare ``TestClient`` has
    them); the lifespan only warms caches. It runs under ``raudio serve`` but not
    under a context-manager-less ``TestClient`` — which is exactly where warmup
    belongs (the real DB) and not (tiny test fixtures). Best-effort: a warmup
    failure is logged, never fatal to startup.
    """
    try:
        await run_in_threadpool(warm_caches, app.state.resources)
    except Exception:  # noqa: BLE001 — warmup must never block the server coming up
        logging.getLogger(__name__).warning("cache warmup failed", exc_info=True)
    yield


def create_app(db_path: str | Path) -> FastAPI:
    """Build the API-only FastAPI app."""
    app = FastAPI(title="raudio api", lifespan=_lifespan)

    # Open Lance handles once, synchronously, in the factory body — not the
    # lifespan: TestClient(create_app(db)) is used without the context manager,
    # so a lifespan would never run and app.state.resources would be unset.
    # (The lifespan above only warms caches, which tests rightly skip.)
    app.state.resources = open_resources(db_path)

    app.include_router(search_router)
    app.include_router(media_router)
    app.include_router(system_router)
    app.include_router(atlas_router)
    app.include_router(topics_router)
    app.include_router(diarization_router)

    # API-only — the Bun frontend serves assets and proxies /api/*. Default "*"
    # is fine behind that local proxy; set RAUDIO_CORS_ORIGINS=https://a,https://b
    # (comma-separated) to lock it down if the API is ever exposed directly.
    # expose_headers is load-bearing for browser Range seeking.
    origins = [o.strip() for o in os.getenv("RAUDIO_CORS_ORIGINS", "*").split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["Content-Range", "Content-Length", "Accept-Ranges"],
    )
    return app


def run(db_path: str | Path, *, host: str = "127.0.0.1", port: int = 8000) -> None:
    """Start the API with uvicorn."""
    import uvicorn

    uvicorn.run(create_app(db_path), host=host, port=port, log_level="info")
