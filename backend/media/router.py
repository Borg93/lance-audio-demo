"""Media endpoints — thumbnail, per-chunk frame, and Range-streamed media.

All three read Lance Blob V2 columns via :mod:`backend.media.blobs`. ``media``
honours HTTP Range so the browser can seek video without downloading the whole
file; ``thumbnail`` / ``chunk_frame`` return small inline JPEGs.
"""

import logging
import math
from typing import Annotated

import lance
from fastapi import APIRouter, Query, Request, Response
from fastapi.responses import FileResponse, StreamingResponse

from backend.core.exceptions import NotFoundError, ValidationError
from backend.deps import StateDep
from backend.media.blobs import (
    doc_blob_size,
    parse_range,
    rowid_for_doc_id,
    stream_blob_range,
    valid_doc_id,
)
from backend.media.clips import MAX_CLIP_S, build_clip

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["media"])


def _doc_mime(ds: lance.LanceDataset, doc_id: str, column: str, default: str) -> str:
    """MIME type stored for a document, or ``default`` when the cell is absent/null.

    ``doc_id`` is already whitelisted by :func:`valid_doc_id` at every call site,
    so inlining it into the filter literal is safe.
    """
    t = ds.to_table(columns=[column], filter=f"doc_id = '{doc_id}'", limit=1)
    if t.num_rows > 0 and t.column(column)[0].is_valid:
        return t.column(column)[0].as_py()
    return default


@router.get("/thumbnail/{doc_id}")
def thumbnail(doc_id: str, state: StateDep) -> Response:
    valid_doc_id(doc_id)
    if state.docs_ds is None:
        raise NotFoundError("documents table missing")
    rowid = rowid_for_doc_id(state.docs_ds, doc_id)
    if rowid is None:
        raise NotFoundError("doc_id not found")
    mime = _doc_mime(state.docs_ds, doc_id, "thumbnail_mime", "image/jpeg")
    blob = state.docs_ds.take_blobs("thumbnail", ids=[rowid])[0]
    with blob as f:
        data = f.read()
    if not data:
        raise NotFoundError("no thumbnail for doc_id")
    return Response(
        content=data, media_type=mime, headers={"Cache-Control": "public, max-age=86400"}
    )


@router.get("/chunk-frame/{doc_id}/{speech_id}/{chunk_id}")
def chunk_frame(
    doc_id: str,
    speech_id: int,
    chunk_id: int,
    state: StateDep,
    frame_idx: Annotated[int, Query(ge=0)] = 0,
) -> Response:
    """A chunk's frame from the `chunk_frames` table; 404 until
    `extract-chunk-frames` has run. ``frame_idx`` selects which frame when a
    chunk was sampled multiple times (0 = the representative frame)."""
    valid_doc_id(doc_id)
    if state.chunk_frames_ds is None:
        raise NotFoundError("frame not extracted yet")

    # frame_idx is matched in Python, NOT in the SQL predicate. With a scalar
    # (BTREE) index on chunk_frames.doc_id, the combination of `with_row_id=True`
    # + a multi-clause filter that *also* constrains frame_idx trips a Lance
    # planner bug that silently returns 0 rows (the row is there — verified via a
    # filter without the frame_idx clause). Keying on (doc_id, speech_id,
    # chunk_id) in SQL and selecting frame_idx here sidesteps it; a chunk has only
    # a handful of frames, so the extra rows are negligible.
    keyed = state.chunk_frames_ds.to_table(
        columns=["frame_mime", "frame_idx"],
        filter=(
            f"doc_id = '{doc_id}' AND speech_id = {int(speech_id)} AND chunk_id = {int(chunk_id)}"
        ),
        with_row_id=True,
    )
    row = next(
        (
            i
            for i in range(keyed.num_rows)
            if keyed.column("frame_idx")[i].as_py() == int(frame_idx)
        ),
        None,
    )
    if row is None:
        raise NotFoundError("frame not extracted yet")
    rowid = keyed.column("_rowid")[row].as_py()
    mime_cell = keyed.column("frame_mime")[row]
    mime = mime_cell.as_py() if mime_cell.is_valid else "image/jpeg"
    try:
        blob = state.chunk_frames_ds.take_blobs("frame_blob", ids=[rowid])[0]
    except Exception as e:
        logger.warning("frame blob read failed", exc_info=True)
        raise NotFoundError("no frame for chunk") from e
    with blob as f:
        data = f.read()
    if not data:
        raise NotFoundError("frame body empty")
    return Response(
        content=data, media_type=mime, headers={"Cache-Control": "public, max-age=86400"}
    )


@router.get("/media-clip/{doc_id}")
def media_clip(
    doc_id: str,
    state: StateDep,
    lo: Annotated[float, Query(ge=0)],
    hi: Annotated[float, Query(gt=0)],
) -> FileResponse:
    """A windowed excerpt of the media with MP3 audio — built for MCP-app
    iframes in hosts whose webview lacks the AAC codec (VS Code). The clip is
    transcoded on first request (ffmpeg over the Range-streaming ``/media``
    endpoint) and disk-cached; FileResponse serves it with Range support so
    in-clip seeking works. Timestamps are clip-local: second 0 == ``lo``.
    """
    valid_doc_id(doc_id)
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        raise ValidationError("need finite seconds with hi > lo")
    if hi - lo > MAX_CLIP_S:
        raise ValidationError(f"clip window capped at {MAX_CLIP_S:.0f}s")
    if state.docs_ds is None or rowid_for_doc_id(state.docs_ds, doc_id) is None:
        raise NotFoundError("doc_id not found")
    source = f"http://127.0.0.1:{state.settings.port}/api/media/{doc_id}"
    path = build_clip(source, doc_id, lo, hi)
    return FileResponse(path, media_type="video/mp4", headers={"Cache-Control": "no-store"})


@router.get("/media/{doc_id}")
def media(doc_id: str, request: Request, state: StateDep) -> Response:
    valid_doc_id(doc_id)
    if state.docs_ds is None:
        raise NotFoundError("documents table missing")
    rowid = rowid_for_doc_id(state.docs_ds, doc_id)
    if rowid is None:
        raise NotFoundError("doc_id not found")

    mime = _doc_mime(state.docs_ds, doc_id, "media_mime", "application/octet-stream")
    total = doc_blob_size(state.docs_ds, "media_blob", rowid)
    range_hdr = request.headers.get("range")
    if range_hdr:
        rng = parse_range(range_hdr, total)
        if rng is None:
            return Response(status_code=416, headers={"Content-Range": f"bytes */{total}"})
        start, end = rng
        return StreamingResponse(
            stream_blob_range(state.docs_ds, "media_blob", rowid, start=start, end=end),
            status_code=206,
            media_type=mime,
            headers={
                "Content-Length": str(end - start + 1),
                "Content-Range": f"bytes {start}-{end}/{total}",
                "Accept-Ranges": "bytes",
                "Cache-Control": "no-store",
            },
        )

    return StreamingResponse(
        stream_blob_range(state.docs_ds, "media_blob", rowid, start=0, end=total - 1),
        media_type=mime,
        headers={
            "Content-Length": str(total),
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-store",
        },
    )
