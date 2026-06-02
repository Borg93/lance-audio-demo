"""Atlas endpoints — serve the precomputed EVōC 2-D projection for the map view.

``raudio feature atlas`` attaches ``atlas_x`` / ``atlas_y`` (+ optional
``atlas_cluster``) to the ``chunks`` table. These read-only routes feed the
frontend's custom Atlas tab:

* ``GET /status``  — is the projection built, and how many rows carry it.
* ``GET /points``  — compact arrays (x/y/cluster/language + a doc-id dictionary
  and per-point keys) for the in-browser scatter renderer. No 2048-d vectors,
  no per-point text — small and fast to load.
* ``GET /chunk/..`` — the full hit for one chunk (text + alignments + paths),
  fetched lazily when a point is selected, for the detail pane + playback.
* ``GET /parquet`` — the same projection as a single parquet (kept for export /
  notebook use; the live UI uses ``/points`` instead).

All are pure ``StateDep`` reads via the native-LanceDB scan idiom
(``chunks.to_lance().to_table(...)``, the same one ``search/service.py`` uses).
"""

import io
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend.deps import StateDep

router = APIRouter(prefix="/api/atlas", tags=["atlas"])

#: The 2-D projection X column. Its presence on ``chunks`` is the signal that
#: ``raudio feature atlas`` has been run (mirrors how ``search/service.py``
#: gates semantic search on ``text_embedding`` being present).
_ATLAS_X = "atlas_x"

#: Columns exported by ``/parquet`` (kept for export). Embedding columns absent.
_EXPORT_COLUMNS = [
    "doc_id", "speech_id", "chunk_id", "atlas_x", "atlas_y", "atlas_cluster",
    "text", "language", "namn", "referenskod", "start", "end", "duration",
]

#: Full-hit columns for the per-chunk detail pane (matches the search hit shape).
_HIT_COLUMNS = [
    "doc_id", "audio_path", "speech_id", "chunk_id", "start", "end", "duration",
    "text", "language", "namn", "referenskod", "bildid", "extraid", "caption",
    "alignments_json",
]


def _is_projected(state: StateDep) -> bool:
    return _ATLAS_X in state.chunks.schema.names


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
def atlas_status(state: StateDep) -> dict[str, Any]:
    """Whether the 2-D projection exists, and how many rows carry it."""
    if not _is_projected(state):
        return {"projected": False, "rows": 0}
    rows = state.chunks.count_rows(filter=f"{_ATLAS_X} IS NOT NULL")
    return {"projected": True, "rows": rows}


@router.get("/points")
def atlas_points(state: StateDep) -> JSONResponse:
    """Compact arrays for the scatter renderer (coords + colour codes + keys)."""
    if not _is_projected(state):
        raise HTTPException(
            status_code=400,
            detail="2D projection not built yet — run `raudio feature atlas`",
        )

    schema = set(state.chunks.schema.names)
    columns = ["doc_id", "speech_id", "chunk_id", "atlas_x", "atlas_y"]
    for optional in ("atlas_cluster", "language"):
        if optional in schema:
            columns.append(optional)

    tbl = state.chunks.to_lance().to_table(
        columns=columns, filter=f"{_ATLAS_X} IS NOT NULL"
    )

    def floats(name: str) -> list[float]:
        return np.round(tbl.column(name).to_numpy(zero_copy_only=False), 4).tolist()

    def ints(name: str) -> list[int]:
        return tbl.column(name).to_numpy(zero_copy_only=False).astype(int).tolist()

    docs_codes, docs_labels = _factorize(tbl.column("doc_id").to_pylist())
    out: dict[str, Any] = {
        "count": tbl.num_rows,
        "x": floats("atlas_x"),
        "y": floats("atlas_y"),
        "docs": docs_labels,   # distinct doc ids
        "doc": docs_codes,     # per-point index into `docs`
        "speech_id": ints("speech_id"),
        "chunk_id": ints("chunk_id"),
    }
    if "atlas_cluster" in columns:
        out["cluster"] = ints("atlas_cluster")
    if "language" in columns:
        lang_codes, lang_labels = _factorize(tbl.column("language").to_pylist())
        out["language"] = lang_codes
        out["languages"] = lang_labels

    return JSONResponse(out, headers={"Cache-Control": "public, max-age=300"})


@router.get("/chunk/{doc_id}/{speech_id}/{chunk_id}")
def atlas_chunk(state: StateDep, doc_id: str, speech_id: int, chunk_id: int) -> dict[str, Any]:
    """Full hit for one chunk (detail pane + playback), looked up by key."""
    from raudio.retrieval.search import parse_alignments_json

    schema = set(state.chunks.schema.names)
    columns = [c for c in _HIT_COLUMNS if c in schema]
    safe_doc = doc_id.replace("'", "''")
    where = f"doc_id = '{safe_doc}' AND speech_id = {speech_id} AND chunk_id = {chunk_id}"
    rows = state.chunks.to_lance().to_table(columns=columns, filter=where).to_pylist()
    if not rows:
        raise HTTPException(status_code=404, detail="chunk not found")

    hit = rows[0]
    hit["alignments"] = parse_alignments_json(hit.pop("alignments_json", None))
    return hit


class ChunkKeys(BaseModel):
    """A batch of (doc_id, speech_id, chunk_id) chunk keys."""

    keys: list[tuple[str, int, int]]


@router.post("/chunks")
def atlas_chunks(state: StateDep, body: ChunkKeys) -> list[dict[str, Any]]:
    """Full hits for a set of chunk keys — the lasso/box selection's results table.

    Capped at 1000 keys; reuses the same Lance scan + alignment parsing as the
    single-chunk lookup, with an OR filter over the keys (the idiom
    ``search/service.py:_frame_search`` uses for its key join).
    """
    from raudio.retrieval.search import parse_alignments_json

    keys = body.keys[:1000]
    if not keys:
        return []
    schema = set(state.chunks.schema.names)
    columns = [c for c in _HIT_COLUMNS if c in schema]
    clauses = []
    for doc_id, speech_id, chunk_id in keys:
        safe_doc = doc_id.replace("'", "''")
        clauses.append(
            f"(doc_id = '{safe_doc}' AND speech_id = {int(speech_id)} AND chunk_id = {int(chunk_id)})"
        )
    where = " OR ".join(clauses)
    rows = state.chunks.to_lance().to_table(columns=columns, filter=where).to_pylist()
    for hit in rows:
        hit["alignments"] = parse_alignments_json(hit.pop("alignments_json", None))
    return rows


@router.get("/parquet")
def atlas_parquet(state: StateDep) -> Response:
    """The projected chunks as a single parquet (export / notebook use)."""
    if not _is_projected(state):
        raise HTTPException(
            status_code=400,
            detail="2D projection not built yet — run `raudio feature atlas`",
        )
    available = set(state.chunks.schema.names)
    columns = [c for c in _EXPORT_COLUMNS if c in available]
    table = state.chunks.to_lance().to_table(columns=columns, filter=f"{_ATLAS_X} IS NOT NULL")
    buffer = io.BytesIO()
    pq.write_table(table, buffer, compression="zstd")
    return Response(
        content=buffer.getvalue(),
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": "inline; filename=atlas.parquet",
            "Cache-Control": "public, max-age=300",
        },
    )
