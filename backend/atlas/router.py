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
``state.chunks_ds`` handle (the same dataset ``search/service.py`` reads from).
"""

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import BaseModel

from backend.deps import StateDep
from raudio.features.topic_tree import topic_layer_columns

router = APIRouter(prefix="/api/atlas", tags=["atlas"])

#: Media type for the /points Arrow IPC stream response.
_ARROW_STREAM_MEDIA_TYPE = "application/vnd.apache.arrow.stream"

#: Memoized /points payloads keyed on (space, dataset version). The full 145k-row
#: scan + dictionary-encode is identical for a given dataset version, so we build
#: the Arrow IPC stream once per space and serve the cached *bytes* thereafter. The
#: dataset version bumps on any rewrite (and a backend restart reopens `chunks_ds`
#: anyway), invalidating stale entries. A plain dict suffices: writes are idempotent
#: (same input → same bytes), so a concurrent double-compute overwrites with equal
#: bytes.
_POINTS_CACHE: dict[tuple[str, int], bytes] = {}

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


def _dictionary(column: pa.ChunkedArray) -> pa.Array:
    """Encode a string column as Arrow ``DICTIONARY<int32, utf8>`` — the codes/
    labels split done natively (indices = per-point colour codes, values =
    labels), replacing the hand-rolled ``_factorize``.

    NULLs are filled to the empty-string label ``""`` first (the frontend
    renders that muted), so the dictionary carries no nulls — matching the old
    contract. ``int32`` indices keep the codes in the typed-array range the JS
    scatter expects.
    """
    filled = pc.fill_null(column.cast(pa.string()), "")
    encoded = filled.combine_chunks().dictionary_encode()
    return encoded.cast(pa.dictionary(pa.int32(), pa.string()))


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


def _doc_file_stems(doc_dict: pa.Array, doc_ids: list[str], audio_paths: list[str]) -> list[str]:
    """Filename stem per distinct doc, aligned with the ``doc`` dictionary order.

    The map view colours/labels by video using the readable audio stem (e.g.
    ``T0000234_00001``) rather than the hashed doc id. The result is one value
    per dictionary entry, in dictionary order, so the frontend can index it by
    the same code it uses for ``doc``. Shipped in schema metadata (not a column:
    it has one entry per distinct doc, not one per point).
    """
    stem_by_doc: dict[str, str] = {}
    for d, a in zip(doc_ids, audio_paths, strict=True):
        if d not in stem_by_doc:
            stem_by_doc[d] = Path(a).stem if a else d
    return [stem_by_doc.get(d, d) for d in doc_dict.to_pylist()]


def _build_points(state: StateDep, space: str) -> bytes:
    """The expensive part of /points: full-table scan → one Arrow IPC stream.

    Builds a single ``pyarrow.Table`` (float16 coords, ~3 sig-digit precision —
    fine for a ~2000px scatter, and it halves both the wire payload and the GPU
    vertex buffer; int32/int64 keys, and a handful of ``DICTIONARY<int32, utf8>``
    colour columns) and serializes it to Arrow IPC **stream** bytes. Pulled out
    so :func:`atlas_points` can memoize the bytes per (space, version).
    """
    cols = _space_cols(space)
    x_col, y_col, cluster_col = cols["x"], cols["y"], cols["cluster"]
    schema = set(state.chunks.schema.names)
    columns = ["doc_id", "audio_path", "speech_id", "chunk_id", x_col, y_col]
    # The broadest topic layer (data-dependent — `topic_l{N-1}`, not always l2)
    # colours the map into named regions; `doc_topic` is the per-video roll-up.
    # Both are low-cardinality (~19), so the dictionary label list stays tiny.
    topic_cols = topic_layer_columns(list(schema))
    broad_topic_col = topic_cols[-1] if topic_cols else None
    for optional in (cluster_col, "language", "namn", broad_topic_col, "doc_topic"):
        if optional and optional in schema:
            columns.append(optional)

    # `with_row_id` ships each point's stable Lance row address (`_rowid`) so the
    # selection table can be fetched with an O(selection) `take` (see /chunks)
    # instead of a per-key filtered full-table scan.
    tbl = state.chunks_ds.scanner(
        columns=columns, filter=f"{x_col} IS NOT NULL", with_row_id=True
    ).to_table()

    def halves(name: str) -> pa.Array:
        # Ship coords as float16 — ~3 sig-digit precision, sub-pixel on the
        # ~2000px scatter — halving the /points payload AND the GPU vertex
        # buffer. The frontend reads the raw f16 bits for the GPU and decodes
        # them to f32 for CPU hover/lasso math (api.ts:f16ToF32).
        arr = tbl.column(name).to_numpy(zero_copy_only=False).astype(np.float16)
        return pa.array(arr, type=pa.float16())

    def ints(name: str, dtype: pa.DataType) -> pa.Array:
        return tbl.column(name).combine_chunks().cast(dtype)

    doc_dict = _dictionary(tbl.column("doc_id"))
    doc_files = _doc_file_stems(
        doc_dict.dictionary,
        tbl.column("doc_id").to_pylist(),
        tbl.column("audio_path").to_pylist(),
    )

    arrays: list[pa.Array] = [
        halves(x_col),
        halves(y_col),
        ints("speech_id", pa.int32()),
        ints("chunk_id", pa.int32()),
        ints("_rowid", pa.int64()),  # stable address for take-based selection fetch
        doc_dict,  # DICTIONARY<int32, utf8>: indices = `doc`, values = `docs`
    ]
    names = ["x", "y", "speech_id", "chunk_id", "rowid", "doc"]

    if cluster_col in columns:
        arrays.append(ints(cluster_col, pa.int32()))
        names.append("cluster")
    if "language" in columns:
        arrays.append(_dictionary(tbl.column("language")))
        names.append("language")
    if "namn" in columns:
        # Archival name dictionary: low-cardinality metadata for the hover popup.
        # A small label list + a per-point int32 — safe to ship for 145k rows
        # (unlike high-cardinality text/caption, which we lazy-fetch per chunk).
        arrays.append(_dictionary(tbl.column("namn")))
        names.append("namn")
    if broad_topic_col and broad_topic_col in columns:
        # Broadest chunk topic — colours the map into named regions. Unclustered
        # chunks are NULL → "" label, rendered muted on the map.
        arrays.append(_dictionary(tbl.column(broad_topic_col)))
        names.append("topic")
    if "doc_topic" in columns:
        # Per-video dominant topic; NULL → "" label when a video's chunks are all
        # noise (so not strictly 100% — most videos do get a topic).
        arrays.append(_dictionary(tbl.column("doc_topic")))
        names.append("doc_topic")

    # count + space + docFiles ride along in the schema metadata. `docFiles` has
    # one entry per distinct doc (not one per point), so it can't be a column in
    # this per-point table — it's JSON-encoded here, aligned with the `doc`
    # dictionary order so the frontend can index it by the same code.
    out_schema = pa.schema(
        [pa.field(n, a.type) for n, a in zip(names, arrays, strict=True)],
        metadata={
            b"count": str(tbl.num_rows).encode(),
            b"space": space.encode(),
            b"docFiles": json.dumps(doc_files).encode(),
        },
    )
    out = pa.table(arrays, schema=out_schema)

    sink = pa.BufferOutputStream()
    with pa.ipc.RecordBatchStreamWriter(sink, out.schema) as writer:
        writer.write_table(out)
    return sink.getvalue().to_pybytes()


@router.get("/points")
def atlas_points(
    state: StateDep,
    space: Literal["text", "visual", "caption"] = Query(
        "text", description="Projection space to read."
    ),
) -> Response:
    """One Apache Arrow IPC stream for the scatter renderer (coords + codes + keys)."""
    if not _is_projected(state, space):
        hint = (
            "raudio feature atlas" if space == "text" else f"raudio feature atlas --space {space}"
        )
        raise HTTPException(
            status_code=400,
            detail=f"{space} 2D projection not built yet — run `{hint}`",
        )

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
