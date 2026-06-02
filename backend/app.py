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

Run:  raudio serve --db ./transcripts.lance --port 8000
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.media.router import router as media_router
from backend.search.router import router as search_router
from backend.state import open_resources
from backend.system.router import router as system_router


def create_app(db_path: str | Path) -> FastAPI:
    """Build the API-only FastAPI app."""
    app = FastAPI(title="raudio api")

    # Open Lance handles once, synchronously, in the factory body — not a
    # lifespan: TestClient(create_app(db)) is used without the context manager,
    # so a lifespan would never run and app.state.resources would be unset.
    app.state.resources = open_resources(db_path)

    app.include_router(search_router)
    app.include_router(media_router)
    app.include_router(system_router)

    # API-only — the Bun frontend serves assets and proxies /api/*.
    # expose_headers is load-bearing for browser Range seeking.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["Content-Range", "Content-Length", "Accept-Ranges"],
    )
    return app


def run(db_path: str | Path, *, host: str = "127.0.0.1", port: int = 8000) -> None:
    """Start the API with uvicorn."""
    import uvicorn

    uvicorn.run(create_app(db_path), host=host, port=port, log_level="info")
