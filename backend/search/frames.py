"""Frame → chunk join family: vector + FTS search over ``chunk_frames``.

Ranks ``chunk_frames`` (by an image/caption vector, or BM25 over the caption
text), dedups to one best frame per chunk, then fetches the matching ``chunks``
payload rows and re-orders them to the frame ranking. Backs visual / scene /
scene_fts search and the frame legs of ``all``. A join failure raises a domain
:class:`ValidationError` (real error logged, never interpolated); the
graceful-degradation ``except`` blocks degrade to ``[]`` when no index exists yet.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from lancedb.query import MatchQuery

from backend.core.exceptions import ValidationError
from backend.search.constants import (
    _CAPTION_COLUMN,
    _FRAME_EMBED_COLUMN,
    _PAYLOAD_COLUMNS,
    _VECTOR_MAX_NPROBES,
    _VECTOR_NPROBES,
    _VECTOR_REFINE_FACTOR,
)
from backend.search.postprocess import ChunkKey, _chunk_key

logger = logging.getLogger(__name__)


def _ranked_or_fallback(
    rank: Callable[..., list[dict[str, Any]]], *, scoped: bool
) -> list[dict[str, Any]]:
    """Run ``rank(scoped=True)``; if the scope prefilter references a column the
    frame table lacks (a metadata filter, which stays on the chunks join), retry
    unscoped. ``[]`` if even the unscoped rank fails (no vector/FTS index yet)."""
    if scoped:
        try:
            return rank(scoped=True)
        except Exception:  # noqa: BLE001 — scope hit a non-frame column → retry unscoped
            pass
    try:
        return rank(scoped=False)
    except Exception:  # noqa: BLE001 — no vector/FTS index yet → no frame hits
        return []


def _frame_search(
    chunk_frames,
    chunks,
    vec: Any,
    *,
    n: int,
    where: str | None,
    column: str = _FRAME_EMBED_COLUMN,
    scope_where: str | None = None,
) -> list[dict[str, Any]]:
    """Rank `chunk_frames` by a vector ``column``, then fetch the matching `chunks`
    rows (where the hit payload lives) and re-order to match.

    Backs both visual search (``frame_embedding`` — image similarity) and scene
    search (``caption_embedding`` — the Swedish caption's text vector); the join
    back to `chunks` is identical, only the ranked column differs.

    ``scope_where`` is an upstream result scope (doc/chunk ids) pushed down as a
    *prefilter* so a refine ranks WITHIN the scope instead of post-filtering the
    global top-n (which returned ~0). It only references columns `chunk_frames`
    has; a non-frame predicate (e.g. a metadata filter, which stays on the chunks
    join via ``where``) falls back to an unscoped rank.

    Returns ``[]`` when frames haven't been extracted/embedded yet — the frame
    pipeline is optional, so visual/scene/all search degrades to empty rather
    than erroring.
    """
    if vec is None or chunk_frames is None or column not in chunk_frames.schema.names:
        return []

    def rank(*, scoped: bool) -> list[dict[str, Any]]:
        qb = (
            chunk_frames.search(vec.tolist(), vector_column_name=column)
            .distance_type("cosine")
            .minimum_nprobes(_VECTOR_NPROBES)
            .maximum_nprobes(_VECTOR_MAX_NPROBES)
            .refine_factor(_VECTOR_REFINE_FACTOR)
            .select(["doc_id", "speech_id", "chunk_id", "_distance"])
            .limit(n)
        )
        if scoped and scope_where:
            qb = qb.where(scope_where, prefilter=True)
        return qb.to_list()

    ranked = _ranked_or_fallback(rank, scoped=bool(scope_where))
    return _frames_to_chunk_hits(chunks, ranked, where)


def _frame_fts_search(
    chunk_frames,
    chunks,
    query: str,
    *,
    n: int,
    where: str | None,
    scope_where: str | None = None,
) -> list[dict[str, Any]]:
    """BM25 keyword search over `chunk_frames.caption`, joined back to `chunks`.

    The keyword counterpart to scene-vector search: same frame→chunk join as
    :func:`_frame_search`, only the ranking is Tantivy BM25 instead of cosine.
    ``scope_where`` prefilters to an upstream scope (see :func:`_frame_search`).
    Returns ``[]`` when captions / their FTS index aren't built yet.
    """
    if not query or chunk_frames is None or _CAPTION_COLUMN not in chunk_frames.schema.names:
        return []

    def rank(*, scoped: bool) -> list[dict[str, Any]]:
        qb = (
            chunk_frames.search(MatchQuery(query, _CAPTION_COLUMN), query_type="fts")
            .select(["doc_id", "speech_id", "chunk_id", "_score"])
            .limit(n)
        )
        if scoped and scope_where:
            qb = qb.where(scope_where, prefilter=True)
        return qb.to_list()

    ranked = _ranked_or_fallback(rank, scoped=bool(scope_where))
    return _frames_to_chunk_hits(chunks, ranked, where)


def _frames_to_chunk_hits(
    chunks, ranked: list[dict[str, Any]], where: str | None
) -> list[dict[str, Any]]:
    """Dedup ranked frame rows to one best frame per chunk (ranking is best-first,
    so the first occurrence wins), then fetch the matching `chunks` payload rows in
    one scan and re-order them to the frame ranking.

    Shared by vector (visual/scene) and FTS (scene keyword) frame search — only the
    upstream ranking differs, the join is identical.
    """
    keys: list[ChunkKey] = []
    seen: set[ChunkKey] = set()
    # Carry each chunk's frame ranking signal (`_distance` for vector search,
    # `_score` for caption FTS) from the best (first) ranked frame, so the hit
    # keeps a score after the join below — the results table / `relevanceOf`
    # reads it. Without this, scene/visual hits arrive scoreless.
    score_by_key: dict[ChunkKey, dict[str, float]] = {}
    for r in ranked:
        key = _chunk_key(r)
        if key not in seen:
            seen.add(key)
            keys.append(key)
            score_by_key[key] = {k: r[k] for k in ("_distance", "_score") if k in r}
    if not keys:
        return []
    # doc_id is a sha1 hex and the ids are ints, so the filter is safe. A pure
    # metadata filter (no vector/FTS query) must go through the Lance dataset's
    # filter scan — a no-query `table.search().where(...)` returns nothing.
    key_filter = " OR ".join(
        f"(doc_id = '{d}' AND speech_id = {s} AND chunk_id = {c})" for d, s, c in keys
    )
    full_filter = f"({key_filter})" + (f" AND ({where})" if where else "")
    try:
        rows = chunks.to_lance().to_table(columns=_PAYLOAD_COLUMNS, filter=full_filter).to_pylist()
    except Exception as e:
        logger.warning("frame search join failed", exc_info=True)
        raise ValidationError("frame search join failed") from e
    by_key = {_chunk_key(r): r for r in rows}
    return [{**by_key[k], **score_by_key[k]} for k in keys if k in by_key]
