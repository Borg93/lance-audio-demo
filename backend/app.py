"""Composition root: one module-level app over the two router groups.

Per the fastapi house skill: ``app`` is module-level with ALL construction in
``lifespan`` (importing this module does zero I/O — the dataset registry is
lazy and datasets open on first request). The two groups stay import-
independent (LANCE_MEDIA_MERGE §4.4): ``backend.media_api`` (viewer role) and
``backend.search_api`` (search role) meet only here, so the rask lift re-hosts
each group without touching routers.

No ``BaseHTTPMiddleware`` anywhere — it buffers responses and breaks 206 Range
streaming; CORS with the Range ``expose_headers`` comes from
``backend.core.middleware``.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from backend.core.config import get_settings
from backend.core.handlers import register_handlers
from backend.core.middleware import register_middleware
from backend.core.probes import router as probes_router
from backend.state import AppState, dataset_handle

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    state = AppState(
        db_path=settings.db_path,
        names=[],
        chunks=None,
        settings=settings,
        http=httpx.Client(),
    )
    app.state.resources = state
    try:
        handle = dataset_handle(state)  # opens + validates the default descriptor: fail fast
        logger.info("default dataset %s ready (%d tables)", handle.id, len(handle.descriptor.tables))
    except Exception:
        # /livez must stay green and other datasets servable; per-request
        # resolution surfaces the config problem as a domain 404, never a 500.
        logger.exception("default dataset failed to open — serving degraded")
    app.state.startup_complete = True
    app.state.shutting_down = False
    yield
    app.state.shutting_down = True
    for resource in (state.http, state.embedder, state.reranker):
        close = getattr(resource, "close", None)
        if close is not None:
            try:
                close()
            except Exception:  # noqa: BLE001 — best-effort teardown
                logger.warning("error closing %s on shutdown", type(resource).__name__)


def _include_groups(app: FastAPI) -> None:
    from backend.media_api.annotate import router as annotate_router
    from backend.media_api.assist import router as assist_router
    from backend.media_api.atlas import router as atlas_router
    from backend.media_api.datasets import router as datasets_router
    from backend.media_api.diarization import router as diarization_router
    from backend.media_api.graph import router as graph_router
    from backend.media_api.media import router as media_router
    from backend.media_api.system import router as system_router
    from backend.media_api.topics import router as topics_router
    from backend.media_api.transcripts import router as transcripts_router
    from backend.media_api.voice import router as voice_router
    from backend.search_api.router import router as search_router

    for router in (
        probes_router,
        datasets_router,
        media_router,
        transcripts_router,
        system_router,
        atlas_router,
        voice_router,
        diarization_router,
        topics_router,
        graph_router,
        search_router,
        annotate_router,
        assist_router,
    ):
        app.include_router(router)


app = FastAPI(title="lance-media backend", lifespan=lifespan)
register_handlers(app)
register_middleware(app, get_settings())
_include_groups(app)


def run(db_path: str | None = None, *, host: str | None = None, port: int | None = None) -> None:
    """Start uvicorn on this module's ``app`` (the ``rmedia serve`` entry).

    ``db_path`` overrides ``MEDIA_DB`` for this process; settings are re-read
    so lifespan picks the override up.
    """
    import os

    import uvicorn

    if db_path is not None:
        os.environ["MEDIA_DB"] = str(db_path)
    if host is not None:
        os.environ["MEDIA_HOST"] = str(host)
    if port is not None:
        # Write it back so settings.port (which media-clip's loopback source URL
        # reads) matches the actual bind port, not just uvicorn's argument.
        os.environ["MEDIA_PORT"] = str(port)
    if db_path is not None or host is not None or port is not None:
        get_settings.cache_clear()
    settings = get_settings()
    uvicorn.run(app, host=settings.host, port=settings.port)
