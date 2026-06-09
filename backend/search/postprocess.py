"""Finalize raw Lance rows into the uniform API hit shape.

Parses each hit's per-word alignments, attaches the chunk's representative-frame
caption (for the list/table views), and provides the RRF fusion used by the
multi-ranking modes. Extracted verbatim from the former monolithic ``service.py``.
"""

from __future__ import annotations

from typing import Any

from backend.search.constants import _CAPTION_COLUMN
from backend.search.filters import _sql_quote


def _postprocess_hits(raw: list[dict[str, Any]], chunk_frames: Any = None) -> list[dict[str, Any]]:
    """Finalize hits for the API: parse alignments + attach each frame's caption.

    ``chunk_frames`` is optional so unit tests can call this with parsed rows
    alone; ``run_search`` always passes it so list/table views get the caption.
    """
    from raudio.retrieval.search import parse_alignments_json

    for h in raw:
        h["alignments"] = parse_alignments_json(h.pop("alignments_json", None))
    _attach_captions(chunk_frames, raw)
    return raw


def _chunk_key(hit: dict[str, Any]) -> tuple[Any, int, int]:
    """The (doc_id, speech_id, chunk_id) identity shared by chunks and frames."""
    return (hit["doc_id"], int(hit["speech_id"]), int(hit["chunk_id"]))


def _attach_captions(chunk_frames, hits: list[dict[str, Any]]) -> None:
    """Set ``hit['caption']`` from each chunk's representative frame (frame_idx=0).

    Captions live on ``chunk_frames``, not ``chunks``, so this one filtered scan
    is how the list/table views learn the scene description for every mode. A
    no-op (leaves no ``caption`` key) when frames or the caption column are
    absent — captions are a nice-to-have, never a reason to fail a search.
    """
    if not hits or chunk_frames is None or _CAPTION_COLUMN not in chunk_frames.schema.names:
        return
    keys = {_chunk_key(h) for h in hits}
    # One `doc_id IN (...)` scan, not a per-hit OR-of-ANDs: DataFusion evaluates a
    # ~100-branch OR predicate row-by-row over all 145k frames (~180ms for a
    # diverse search), whereas IN is a hash-membership scan (~75ms, 2.4x faster,
    # verified identical output). We over-fetch the matched docs' frame-0 rows and
    # pick out the exact chunks in Python below. frame_idx=0 = the representative
    # caption frame.
    docs = {_sql_quote(d) for d, _, _ in keys}
    doc_list = ",".join(f"'{d}'" for d in docs)
    key_filter = f"doc_id IN ({doc_list}) AND frame_idx = 0"
    try:
        rows = (
            chunk_frames.to_lance()
            .to_table(
                columns=["doc_id", "speech_id", "chunk_id", _CAPTION_COLUMN], filter=key_filter
            )
            .to_pylist()
        )
    except Exception:  # noqa: BLE001 — caption is decorative; never fail a search over it
        return
    by_key = {_chunk_key(r): r.get(_CAPTION_COLUMN) for r in rows}
    for h in hits:
        h["caption"] = by_key.get(_chunk_key(h))


def _rrf_fuse(rankings: list[list[dict[str, Any]]], k: int = 60) -> list[dict[str, Any]]:
    """Reciprocal-rank fusion across N ranked lists keyed on (doc_id, chunk_id).

    Lance's hybrid query handles RRF natively when both FTS and vector are in
    play; we use this helper for the multi-column case (text_embedding +
    frame_embedding) where we issue two distinct vector queries and need to
    merge them ourselves.
    """
    scored: dict[tuple[Any, Any], float] = {}
    rep: dict[tuple[Any, Any], dict[str, Any]] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking):
            key = (hit.get("doc_id"), hit.get("chunk_id"))
            scored[key] = scored.get(key, 0.0) + 1.0 / (k + rank)
            # Keep the first occurrence (highest-ranked) as the canonical row.
            rep.setdefault(key, hit)
    fused = sorted(rep.values(), key=lambda h: -scored[(h["doc_id"], h["chunk_id"])])
    return fused
