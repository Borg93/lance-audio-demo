"""Application lifespan — warm caches on boot, flip readiness flags.

Lance resources are opened in the app *factory* body (so a bare ``TestClient``
without a context manager still has them — see :func:`backend.app.create_app`).
This lifespan only: (1) warms the index/atlas caches off the event loop,
best-effort; (2) flips ``app.state.startup_complete``; (3) on shutdown flips
``app.state.shutting_down`` and best-effort closes any vLLM client cached on
state that happens to expose ``.close()``. We never invent a ``close()`` on the
``raudio.vllm.*`` clients — we only call it if it's there.

Plain ``lifespan`` (not a ``make_lifespan(settings)`` factory): this lifespan
needs no config — ``Settings`` already rides on ``app.state.resources`` — so
there is nothing to close over.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool

from backend.warmup import warm_caches

logger = logging.getLogger(__name__)


def _best_effort_close(obj: object) -> None:
    close = getattr(obj, "close", None)
    if callable(close):
        with suppress(Exception):
            close()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    try:
        await run_in_threadpool(warm_caches, app.state.resources)
    except Exception:  # noqa: BLE001 — warmup must never block the server coming up
        logger.warning("cache warmup failed", exc_info=True)
    app.state.startup_complete = True
    try:
        yield
    finally:
        app.state.shutting_down = True
        resources = app.state.resources
        _best_effort_close(getattr(resources, "embedder", None))
        _best_effort_close(getattr(resources, "reranker", None))
        _best_effort_close(getattr(resources, "http", None))
