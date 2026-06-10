"""Atlas endpoints — serve the precomputed EVōC 2-D projection for the map view.

``raudio feature atlas`` attaches ``atlas_x`` / ``atlas_y`` (+ optional
``atlas_cluster``) to the ``chunks`` table. These read-only routes feed the
frontend's custom Atlas tab:

* ``GET /status?space=`` — which projection spaces (text|visual) are built, and
  how many rows carry the requested one.
* ``GET /points?space=`` — one Apache Arrow IPC **stream** (binary, parse-free):
  x/y coords, per-point keys, and a handful of categorical columns shipped as
  Arrow ``DICTIONARY<int32, utf8>`` (the dictionary indices are the per-point
  colour codes, the dictionary values are the labels — factorization for free).
  No 2048-d vectors, no per-point text — small and fast to load. ``namn`` is a
  dictionary column (low-cardinality archival metadata) for the hover popup;
  high-cardinality text/caption stay out and are lazy-fetched per chunk via
  ``/chunk``.
* ``GET /chunk/..`` — the full hit for one chunk (text + alignments + paths),
  fetched lazily when a point is selected, for the detail pane + playback.

All are pure ``StateDep`` reads via the native-LanceDB scan idiom over the cached
``state.chunks_ds`` handle (the same dataset ``search/service.py`` reads from). The
Arrow ``/points`` serialization lives in :mod:`backend.atlas.points`.
"""

from typing import Annotated, Any

from fastapi import APIRouter, Query, Response

from backend.atlas.points import (
    _POINTS_CACHE,
    _SPACES,
    _build_points,
    _is_projected,
    _space_cols,
)
from backend.core.exceptions import NotFoundError, ValidationError
from backend.deps import StateDep
from backend.schemas.atlas import AtlasSpace, AtlasStatusResponse, ChunkRowIds

router = APIRouter(prefix="/api/atlas", tags=["atlas"])

#: Media type for the /points Arrow IPC stream response.
_ARROW_STREAM_MEDIA_TYPE = "application/vnd.apache.arrow.stream"


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


@router.get("/status")
def atlas_status(
    state: StateDep,
    space: Annotated[AtlasSpace, Query(description="Projection space to report rows for.")] = (
        AtlasSpace.text
    ),
) -> AtlasStatusResponse:
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
    return AtlasStatusResponse(projected=projected, rows=rows, space=space, spaces=spaces)


@router.get("/points")
def atlas_points(
    state: StateDep,
    space: Annotated[AtlasSpace, Query(description="Projection space to read.")] = AtlasSpace.text,
) -> Response:
    """One Apache Arrow IPC stream for the scatter renderer (coords + codes + keys)."""
    if not _is_projected(state, space):
        hint = (
            "raudio feature atlas" if space == "text" else f"raudio feature atlas --space {space}"
        )
        raise ValidationError(f"{space} 2D projection not built yet — run `{hint}`")

    # Memoize on (space, dataset version): the scan+encode is identical until the
    # dataset is rewritten (version bump) or the backend restarts.
    key = (space, state.chunks_ds.version)
    body = _POINTS_CACHE.get(key)
    if body is None:
        body = _build_points(state, space)
        _POINTS_CACHE[key] = body

    return Response(
        content=body,
        media_type=_ARROW_STREAM_MEDIA_TYPE,
        headers={"Cache-Control": "public, max-age=300"},
    )


def _attach_frame_captions(frames: Any, rows: list[dict[str, Any]]) -> None:
    """Attach each chunk's representative-frame (``frame_idx=0``) caption from
    ``chunk_frames`` — captions live there, not on ``chunks``. Batched to keep the
    number of ``chunk_frames`` scans low per selection; a no-op when frames or
    captions are absent (captions are decorative).
    """
    from backend.search.service import _attach_captions

    if frames is None or not rows:
        return
    # _attach_captions now filters with `doc_id IN (...)` (not the deep per-hit
    # OR that used to overflow the parser), so we batch large — a 1000-point
    # lasso goes from ~7 scans to ~2 (≈430ms→360ms, benchmarked).
    for start in range(0, len(rows), 500):
        _attach_captions(frames, rows[start : start + 500])


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
        raise NotFoundError("chunk not found")

    hit = rows[0]
    hit["alignments"] = parse_alignments_json(hit.pop("alignments_json", None))
    _attach_frame_captions(state.chunk_frames_tbl, [hit])
    return hit


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
