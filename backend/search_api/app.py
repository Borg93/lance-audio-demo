"""Thin app factory for the search group (§4.4 composition contract).

``create_search_app`` wires ONLY what the search role owns: the problem+json
handlers and the search router bound to an externally-opened
:class:`~backend.state.AppState`. Middleware (CORS), probes, and the MCP mount
stay with the composition root that mounts this app next to the media group —
so the later re-host under rask ``service_kit.make_service_app`` touches
nothing here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import FastAPI

from backend.core.handlers import register_handlers
from backend.search_api.router import router

if TYPE_CHECKING:
    from backend.state import AppState


def create_search_app(state: AppState) -> FastAPI:
    """Build the search-group FastAPI app around shared, already-opened state."""
    app = FastAPI(title="search api")
    app.state.resources = state
    register_handlers(app)
    app.include_router(router)
    return app
