"""media_api — the viewer-role router group (LANCE_MEDIA_MERGE §4.4).

Datasets + descriptor endpoints, blob/Range streaming, thumbnails/frames, doc
transcript, and atlas points — everything the future media service serves. All
table/column knowledge flows from the dataset descriptor resolved through
``backend.state.dataset_handle``; nothing in this package imports the pipeline
package or the sibling search group, and shared primitives live only in
``backend.core`` / ``backend.lancekit`` / ``backend.state`` / ``backend.deps``
/ ``backend.schemas``.

``create_media_app`` is the thin per-group factory: the two groups are mounted
together today and re-hosted separately at rask-merge time without touching
routers. No ``BaseHTTPMiddleware`` anywhere (it buffers responses and breaks
206 Range streaming); CORS ``expose_headers`` is registered app-level.
"""

from __future__ import annotations

import httpx
from fastapi import FastAPI

from backend.core.config import Settings, get_settings
from backend.core.handlers import register_handlers
from backend.core.middleware import register_middleware
from backend.state import AppState


def create_media_state(settings: Settings | None = None) -> AppState:
    """A registry-only :class:`AppState` for the media group.

    The legacy single-DB fields stay at their inert defaults; every route in
    this package resolves data through ``dataset_handle`` instead.
    """
    settings = settings or get_settings()
    return AppState(
        db_path=settings.db_path,
        names=[],
        chunks=None,
        settings=settings,
        http=httpx.Client(),
    )


def create_media_app(
    settings: Settings | None = None, state: AppState | None = None
) -> FastAPI:
    """Build the media-group FastAPI app.

    ``state`` lets the composition root share one :class:`AppState` (registry,
    points cache, HTTP pool) between router groups; omitted, a standalone one
    is created — which is what the tests and a future split deployment use.
    """
    # Routers are imported here (not at module top) so importing the package
    # for its factory never drags FastAPI route introspection into tools that
    # only need `create_media_state`.
    from backend.media_api.atlas import router as atlas_router
    from backend.media_api.datasets import router as datasets_router
    from backend.media_api.diarization import router as diarization_router
    from backend.media_api.graph import router as graph_router
    from backend.media_api.media import router as media_router
    from backend.media_api.system import router as system_router
    from backend.media_api.topics import router as topics_router
    from backend.media_api.transcripts import router as transcripts_router
    from backend.media_api.voice import router as voice_router

    settings = settings or get_settings()
    app = FastAPI(title="media api")
    app.state.resources = state if state is not None else create_media_state(settings)

    register_handlers(app)
    register_middleware(app, settings)

    for group_router in (
        datasets_router,
        media_router,
        transcripts_router,
        system_router,
        atlas_router,
        voice_router,
        diarization_router,
        topics_router,
        graph_router,
    ):
        app.include_router(group_router)
    return app
