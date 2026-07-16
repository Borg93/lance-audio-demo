"""FastAPI dependency wrappers — the seam between the app and the routers.

Routers depend on these instead of capturing closures or touching ``app.state``.
``StateDep`` hands a router the per-app :class:`AppState`; the search group's
encoder deps live in ``backend.search_api.deps`` (each group carries its own).
Tests override via ``app.dependency_overrides``.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from backend.state import AppState


def get_state(request: Request) -> AppState:
    """The resources built in lifespan, stashed on ``app.state``."""
    return request.app.state.resources


StateDep = Annotated[AppState, Depends(get_state)]
