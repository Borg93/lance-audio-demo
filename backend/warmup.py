"""Startup cache warmup — pull the Lance indices + the atlas /points payload into
cache so the first real request doesn't pay cold-cache latency.

Lance caches vector/scalar/FTS indices and dataset metadata lazily, on first touch
(index cache 6 GiB, metadata cache 1 GiB — see the Lance perf guide). A freshly
started backend therefore serves its first vector search and its first atlas load
slowly. This runs one tiny query per built index and precomputes the /points scan
once, moving that one-off cost off the first user. Strictly best-effort: every
step is guarded so a warmup failure can never keep the server from coming up.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from lancedb.query import MatchQuery

from backend.atlas.points import SPACES, build_points, is_projected

logger = logging.getLogger(__name__)

# Any token loads the inverted index into cache; matches aren't needed.
_FTS_PROBE = "och"


def _warm_vector(table: Any, column: str) -> None:
    """Run a 1-row search on ``column`` to load its IVF_PQ index into cache."""
    if table is None or column not in table.schema.names:
        return
    dim = getattr(table.schema.field(column).type, "list_size", None)
    if not dim or dim <= 0:
        return
    table.search([0.0] * dim, vector_column_name=column).limit(1).to_list()


def _warm_fts(table: Any, column: str) -> None:
    """Run a 1-row FTS query to load ``column``'s inverted index into cache."""
    if table is None or column not in table.schema.names:
        return
    table.search(MatchQuery(_FTS_PROBE, column), query_type="fts").limit(1).to_list()


def _warm_atlas_points(state: Any) -> int:
    """Precompute the /points payload for every built space into the state cache."""
    warmed = 0
    for space in SPACES:
        if not is_projected(state, space):
            continue
        key = (space, state.chunks_ds.version)
        if key not in state.points_cache:
            state.points_cache[key] = build_points(state, space)
        warmed += 1
    return warmed


def warm_caches(state: Any) -> None:
    """Load every built index + the atlas points payload into cache. Best-effort."""
    start = time.perf_counter()
    vector_targets = (
        (state.chunks, "text_embedding"),
        (state.chunk_frames_tbl, "frame_embedding"),
        (state.chunk_frames_tbl, "caption_embedding"),
    )
    fts_targets = (
        (state.chunks, "text"),
        (state.chunk_frames_tbl, "caption"),
    )
    for table, column in vector_targets:
        try:
            _warm_vector(table, column)
        except Exception:  # noqa: BLE001 — warmup is best-effort, never fatal
            logger.debug("vector warmup skipped (%s)", column, exc_info=True)
    for table, column in fts_targets:
        try:
            _warm_fts(table, column)
        except Exception:  # noqa: BLE001
            logger.debug("fts warmup skipped (%s)", column, exc_info=True)
    spaces = 0
    try:
        spaces = _warm_atlas_points(state)
    except Exception:  # noqa: BLE001
        logger.warning("atlas points warmup failed", exc_info=True)
    logger.info(
        "cache warmup done in %.2fs (atlas spaces warmed=%d)",
        time.perf_counter() - start,
        spaces,
    )
