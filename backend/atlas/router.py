"""Atlas endpoints — serve the precomputed EVōC 2-D projection for the map view.

``raudio feature atlas`` attaches ``atlas_x`` / ``atlas_y`` (+ optional
``atlas_cluster``) to the ``chunks`` table. These read-only routes feed the
frontend's custom Atlas tab:

* ``GET /status?space=`` — which projection spaces (text|visual) are built, and
  how many rows carry the requested one.
* ``GET /points?space=`` — compact arrays (x/y/cluster/language/namn + a doc-id
  dictionary and per-point keys) for the in-browser scatter renderer. No 2048-d
  vectors, no per-point text — small and fast to load. ``namn`` is factorized
  (low-cardinality archival metadata) for the hover popup; high-cardinality
  text/caption stay out and are lazy-fetched per chunk via ``/chunk``.
* ``GET /chunk/..`` — the full hit for one chunk (text + alignments + paths),
  fetched lazily when a point is selected, for the detail pane + playback.

All are pure ``StateDep`` reads via the native-LanceDB scan idiom over the cached
``state.chunks_ds`` handle (the same dataset ``search/service.py`` reads from).
"""

from typing import Any, Literal

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend.deps import StateDep
from raudio.features.topic_tree import topic_layer_columns

router = APIRouter(prefix="/api/atlas", tags=["atlas"])

#: Memoized /points payloads keyed on (space, dataset version). The full 145k-row
#: scan + factorize is identical for a given dataset version, so we compute it once
#: per space and serve the cached dict thereafter. The dataset version bumps on any
#: rewrite (and a backend restart reopens `chunks_ds` anyway), invalidating stale
#: entries. A plain dict suffices: writes are idempotent (same input → same output),
#: so a concurrent double-compute just overwrites with an equal value.
_POINTS_CACHE: dict[tuple[str, int], dict[str, Any]] = {}

#: The two projection spaces and their column triplets. ``text`` is the default
#: (``raudio feature atlas``, from ``text_embedding``); ``visual`` is the
#: frame-embedding map (``raudio feature atlas --space visual``). Each space's
#: X column doubling as the "is this space built?" signal — mirrors how
#: ``search/service.py`` gates semantic search on ``text_embedding`` presence.
_SPACES: dict[str, dict[str, str]] = {
    "text": {"x": "atlas_x", "y": "atlas_y", "cluster": "atlas_cluster"},
    "visual": {"x": "atlas_img_x", "y": "atlas_img_y", "cluster": "atlas_img_cluster"},
    "caption": {"x": "atlas_cap_x", "y": "atlas_cap_y", "cluster": "atlas_cap_cluster"},
}


def _space_cols(space: str) -> dict[str, str]:
    cols = _SPACES.get(space)
    if cols is None:
        raise HTTPException(
            status_code=400, detail=f"unknown space '{space}' (text|visual|caption)"
        )
    return cols


#: Full-hit columns for the per-chunk detail pane (matches the search hit shape).
_HIT_COLUMNS = [
    "doc_id",
    "audio_path",
    "speech_id",
    "chunk_id",
    "start",
    "end",
    "duration",
    "text",
    "language",
    "namn",
    "referenskod",
    "bildid",
    "extraid",
    "caption",
    "alignments_json",
]


def _is_projected(state: StateDep, space: str = "text") -> bool:
    return _space_cols(space)["x"] in state.chunks.schema.names


def _factorize(values: list[Any]) -> tuple[list[int], list[str]]:
    """Map a column of repeated values to (per-row index, distinct-label list).

    Lets us ship one small label list + a compact int per row instead of the
    full (often long, repeated) string on every point — e.g. doc ids, language.
    """
    index: dict[Any, int] = {}
    labels: list[str] = []
    codes: list[int] = []
    for v in values:
        v = "" if v is None else v
        code = index.get(v)
        if code is None:
            code = len(labels)
            index[v] = code
            labels.append(str(v))
        codes.append(code)
    return codes, labels


@router.get("/status")
def atlas_status(
    state: StateDep,
    space: Literal["text", "visual", "caption"] = Query(
        "text", description="Projection space to report rows for."
    ),
) -> dict[str, Any]:
    """Which projection spaces are built, plus the requested space's row count.

    ``spaces`` always reports both text+visual presence (so the UI can gate a
    Text/Visual toggle); ``projected``/``rows`` reflect the requested ``space``
    (back-compatible with the old single-space shape).
    """
    names = set(state.chunks.schema.names)
    spaces = {name: cols["x"] in names for name, cols in _SPACES.items()}
    cols = _space_cols(space)
    projected = cols["x"] in names
    rows = state.chunks.count_rows(filter=f"{cols['x']} IS NOT NULL") if projected else 0
    return {"projected": projected, "rows": rows, "space": space, "spaces": spaces}


def _build_points(state: StateDep, space: str) -> dict[str, Any]:
    """The expensive part of /points: full-table scan + factorize into ``out``.

    Pulled out so :func:`atlas_points` can memoize the result per (space, version).
    """
    cols = _space_cols(space)
    x_col, y_col, cluster_col = cols["x"], cols["y"], cols["cluster"]
    schema = set(state.chunks.schema.names)
    columns = ["doc_id", "speech_id", "chunk_id", x_col, y_col]
    # The broadest topic layer (data-dependent — `topic_l{N-1}`, not always l2)
    # colours the map into named regions; `doc_topic` is the per-video roll-up.
    # Both are low-cardinality (~19) so they factorize into a small label list.
    topic_cols = topic_layer_columns(list(schema))
    broad_topic_col = topic_cols[-1] if topic_cols else None
    for optional in (cluster_col, "language", "namn", broad_topic_col, "doc_topic"):
        if optional and optional in schema:
            columns.append(optional)

    # `with_row_id` ships each point's stable Lance row address (`_rowid`) so the
    # selection table can be fetched with an O(selection) `take` (see /chunks)
    # instead of a per-key filtered full-table scan.
    tbl = (
        state.chunks_ds.scanner(
            columns=columns, filter=f"{x_col} IS NOT NULL", with_row_id=True
        ).to_table()
    )

    def floats(name: str) -> list[float]:
        return np.round(tbl.column(name).to_numpy(zero_copy_only=False), 4).tolist()

    def ints(name: str) -> list[int]:
        return tbl.column(name).to_numpy(zero_copy_only=False).astype(int).tolist()

    docs_codes, docs_labels = _factorize(tbl.column("doc_id").to_pylist())
    out: dict[str, Any] = {
        "count": tbl.num_rows,
        "space": space,
        "x": floats(x_col),
        "y": floats(y_col),
        "docs": docs_labels,  # distinct doc ids
        "doc": docs_codes,  # per-point index into `docs`
        "speech_id": ints("speech_id"),
        "chunk_id": ints("chunk_id"),
        "rowid": ints("_rowid"),  # stable address for take-based selection fetch
    }
    if cluster_col in columns:
        out["cluster"] = ints(cluster_col)
    if "language" in columns:
        lang_codes, lang_labels = _factorize(tbl.column("language").to_pylist())
        out["language"] = lang_codes
        out["languages"] = lang_labels
    if "namn" in columns:
        # Factorized archival name: low-cardinality metadata for the hover popup.
        # A small label list + a per-point int — safe to ship for 145k rows
        # (unlike high-cardinality text/caption, which we lazy-fetch per chunk).
        namn_codes, namn_labels = _factorize(tbl.column("namn").to_pylist())
        out["namn"] = namn_codes
        out["namns"] = namn_labels
    if broad_topic_col and broad_topic_col in columns:
        # Broadest chunk topic — colours the map into named regions. Unclustered
        # chunks are NULL → "" label, rendered muted on the map.
        topic_codes, topic_labels = _factorize(tbl.column(broad_topic_col).to_pylist())
        out["topic"] = topic_codes
        out["topics"] = topic_labels
    if "doc_topic" in columns:
        # Per-video dominant topic; NULL → "" label when a video's chunks are all
        # noise (so not strictly 100% — most videos do get a topic).
        dt_codes, dt_labels = _factorize(tbl.column("doc_topic").to_pylist())
        out["doc_topic"] = dt_codes
        out["doc_topics"] = dt_labels

    return out


@router.get("/points")
def atlas_points(
    state: StateDep,
    space: Literal["text", "visual", "caption"] = Query(
        "text", description="Projection space to read."
    ),
) -> JSONResponse:
    """Compact arrays for the scatter renderer (coords + colour codes + keys)."""
    if not _is_projected(state, space):
        hint = (
            "raudio feature atlas" if space == "text" else f"raudio feature atlas --space {space}"
        )
        raise HTTPException(
            status_code=400,
            detail=f"{space} 2D projection not built yet — run `{hint}`",
        )

    # Memoize on (space, dataset version): the scan+factorize is identical until
    # the dataset is rewritten (version bump) or the backend restarts.
    key = (space, state.chunks_ds.version)
    out = _POINTS_CACHE.get(key)
    if out is None:
        out = _build_points(state, space)
        _POINTS_CACHE[key] = out

    return JSONResponse(out, headers={"Cache-Control": "public, max-age=300"})


def _attach_frame_captions(frames: Any, rows: list[dict[str, Any]]) -> None:
    """Attach each chunk's representative-frame (``frame_idx=0``) caption from
    ``chunk_frames`` — captions live there, not on ``chunks``. Batched so the key
    filter never grows the deep ``OR`` that overflows the parser on a big
    selection; a no-op when frames/captions are absent (captions are decorative).
    """
    from backend.search.service import _attach_captions

    if frames is None or not rows:
        return
    for start in range(0, len(rows), 150):
        _attach_captions(frames, rows[start : start + 150])


@router.get("/chunk/{doc_id}/{speech_id}/{chunk_id}")
def atlas_chunk(state: StateDep, doc_id: str, speech_id: int, chunk_id: int) -> dict[str, Any]:
    """Full hit for one chunk (detail pane + playback), looked up by key."""
    from raudio.retrieval.search import parse_alignments_json

    schema = set(state.chunks.schema.names)
    columns = [c for c in _HIT_COLUMNS if c in schema]
    safe_doc = doc_id.replace("'", "''")
    where = f"doc_id = '{safe_doc}' AND speech_id = {speech_id} AND chunk_id = {chunk_id}"
    rows = state.chunks_ds.to_table(columns=columns, filter=where).to_pylist()
    if not rows:
        raise HTTPException(status_code=404, detail="chunk not found")

    hit = rows[0]
    hit["alignments"] = parse_alignments_json(hit.pop("alignments_json", None))
    _attach_frame_captions(state.chunk_frames_tbl, [hit])
    return hit


class ChunkRowIds(BaseModel):
    """A batch of stable Lance row addresses (``_rowid``) for selected points.

    The frontend reads these from /points (one per scatter point) and sends back
    exactly the selected subset — far cheaper than re-deriving rows from keys.
    """

    rowids: list[int]


@router.post("/chunks")
def atlas_chunks(state: StateDep, body: ChunkRowIds) -> list[dict[str, Any]]:
    """Full hits for a lasso/box/legend selection, addressed by ``_rowid``.

    A flat ``_rowid IN (...)`` predicate lets Lance fetch exactly the selected
    rows by address (~one random-access take, not a filtered full scan), so a
    1000-point selection resolves in a few ms instead of seconds. Addresses are
    stable for the served table version; stale ids simply don't match. Capped at
    1000 (the table render budget); the full selection still drives map dimming.
    """
    from raudio.retrieval.search import parse_alignments_json

    rowids = body.rowids[:1000]
    if not rowids:
        return []
    schema = set(state.chunks.schema.names)
    columns = [c for c in _HIT_COLUMNS if c in schema]
    csv = ",".join(str(int(r)) for r in rowids)
    rows = state.chunks_ds.to_table(columns=columns, filter=f"_rowid in ({csv})").to_pylist()
    for hit in rows:
        hit["alignments"] = parse_alignments_json(hit.pop("alignments_json", None))
    _attach_frame_captions(state.chunk_frames_tbl, rows)
    return rows
