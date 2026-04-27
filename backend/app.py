"""FastAPI backend for the Lance-backed transcript viewer.

Every endpoint reads from Lance directly — no disk walks, no sidecar JSON. The
``media_blob`` column is Lance Blob V2 External (URI); ``thumbnail`` is Blob V2
Inline (bytes); ``alignments_json`` is Lance JSONB. All blob reads go through
``ds.take_blobs(...)`` which returns lazy, seekable ``BlobFile`` handles, so
HTTP Range requests map directly to ``BlobFile.seek(start) + read(length)``.

Run:
    raudio serve --db ./transcripts.lance --port 3000
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import lance
import lancedb
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse


logger = logging.getLogger(__name__)

_DOC_ID_RE = re.compile(r"^[a-f0-9]{16}$")

# 1 MiB streaming chunk — big enough to amortize seek cost, small enough to
# keep memory bounded when many clients stream concurrently.
_STREAM_CHUNK = 1 << 20


def _valid_doc_id(doc_id: str) -> None:
    if not _DOC_ID_RE.match(doc_id):
        raise HTTPException(status_code=400, detail="invalid doc_id")


def _stream_blob_range(ds: "lance.LanceDataset", column: str, index: int, start: int, end: int):
    """Yield chunks of ``[start, end]`` (inclusive) from a Lance blob column."""
    blob = ds.take_blobs(column, indices=[index])[0]
    # ``BlobFile`` is a file-like object with seek/read; we open it with `with`
    # per the Lance examples to release the handle when done.
    with blob as f:
        f.seek(start)
        remaining = end - start + 1
        while remaining > 0:
            chunk = f.read(min(_STREAM_CHUNK, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


def _index_for_doc_id(ds: "lance.LanceDataset", doc_id: str) -> int | None:
    """Resolve a ``doc_id`` to its row index in the ``documents`` dataset."""
    t = ds.to_table(columns=["doc_id"], filter=f"doc_id = '{doc_id}'", with_row_id=True)
    if t.num_rows == 0:
        return None
    # ``_rowid`` here is actually the row index for stable positional reads.
    # We fetch the full row id list and use the first (and only) match.
    return int(t.column("_rowid")[0].as_py())


def _parse_range(header: str, total: int) -> tuple[int, int] | None:
    """Parse a single ``bytes=start-end`` range header, clamped to ``total``."""
    m = re.match(r"^\s*bytes=(\d*)-(\d*)\s*$", header)
    if not m:
        return None
    s, e = m.group(1), m.group(2)
    if s == "" and e == "":
        return None
    if s == "":
        length = int(e)
        start = max(0, total - length)
        end = total - 1
    else:
        start = int(s)
        end = int(e) if e else total - 1
    if start > end or start >= total:
        return None
    return start, min(end, total - 1)


def _doc_blob_size(ds: "lance.LanceDataset", column: str, index: int) -> int:
    """Probe the size of a blob without reading its contents."""
    blob = ds.take_blobs(column, indices=[index])[0]
    with blob as f:
        try:
            return f.size()  # type: ignore[attr-defined]
        except AttributeError:
            f.seek(0, 2)
            return f.tell()


def create_app(db_path: str | Path) -> FastAPI:
    """Build the API-only FastAPI app."""
    app = FastAPI(title="raudio api")

    db = lancedb.connect(str(db_path))
    names = db.table_names()
    logger.info("opened Lance DB %s — tables: %s", db_path, names)

    if "chunks" not in names:
        raise RuntimeError(f"'chunks' table missing in {db_path}")
    chunks = db.open_table("chunks")

    # We access `documents` as a raw Lance dataset because lancedb.Table
    # doesn't expose take_blobs.
    docs_ds = None
    if "documents" in names:
        docs_ds = lance.dataset(str(Path(db_path) / "documents.lance"))

    # ── /api/search ───────────────────────────────────────────────────────
    @app.get("/api/search")
    def search(
        q: str = "",
        n: int = 20,
        language: str | None = None,
        namn: str | None = None,
        referenskod: str | None = None,
        extraid: str | None = None,
        fuzziness: int = 0,
        phrase: bool = False,
    ) -> list[dict[str, Any]]:
        q = q.strip()
        if not q:
            return []
        # Build optional SQL-style WHERE filters against the metadata columns.
        clauses: list[str] = []
        if language:
            clauses.append(f"language = '{language.replace(chr(39), chr(39)*2)}'")
        if namn:
            clauses.append(f"namn LIKE '%{namn.replace(chr(39), chr(39)*2)}%'")
        if referenskod:
            clauses.append(f"referenskod LIKE '%{referenskod.replace(chr(39), chr(39)*2)}%'")
        if extraid:
            clauses.append(f"extraid = '{extraid.replace(chr(39), chr(39)*2)}'")
        where = " AND ".join(clauses) if clauses else None

        # Pick the structured FTS query. Lance's plain-string FTS is a thin
        # wrapper that supports fewer features than the Match*/Phrase APIs:
        # we want explicit fuzziness control + phrase support.
        from lancedb.query import MatchQuery, PhraseQuery
        fz = max(0, min(2, fuzziness))
        if phrase:
            fts_query = PhraseQuery(q, "text")
        else:
            fts_query = MatchQuery(q, "text", fuzziness=fz)

        try:
            qb = (
                chunks.search(fts_query)
                .select([
                    "_score", "doc_id", "audio_path", "speech_id", "chunk_id",
                    "start", "end", "duration", "text", "language",
                    "namn", "referenskod", "bildid", "extraid",
                    "alignments_json",
                ])
                .limit(min(n, 100))
            )
            if where:
                qb = qb.where(where, prefilter=False)
            raw = qb.to_list()
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"search failed: {e}")
        # alignments_json comes back as a JSON string — parse for the client.
        out = []
        for h in raw:
            aj = h.pop("alignments_json", None) or "[]"
            try:
                h["alignments"] = json.loads(aj) if isinstance(aj, str) else aj
            except json.JSONDecodeError:
                h["alignments"] = []
            out.append(h)
        return out

    # ── /api/documents (gallery) ──────────────────────────────────────────
    @app.get("/api/documents")
    def documents(page: int = 1, per_page: int = 24) -> dict[str, Any]:
        if docs_ds is None:
            return {"total": 0, "page": 1, "docs": []}
        total = docs_ds.count_rows()
        offset = max(0, (page - 1) * per_page)
        tbl = docs_ds.to_table(
            columns=[
                "doc_id", "audio_path", "duration", "referenskod", "namn",
                "bildid", "extraid",
            ],
            limit=per_page,
            offset=offset,
        )
        return {"total": total, "page": page, "docs": tbl.to_pylist()}

    # ── /api/thumbnail/:doc_id ────────────────────────────────────────────
    @app.get("/api/thumbnail/{doc_id}")
    def thumbnail(doc_id: str) -> Response:
        _valid_doc_id(doc_id)
        if docs_ds is None:
            raise HTTPException(status_code=404, detail="documents table missing")
        idx = _index_for_doc_id(docs_ds, doc_id)
        if idx is None:
            raise HTTPException(status_code=404, detail="doc_id not found")
        mime_row = docs_ds.to_table(
            columns=["thumbnail_mime"],
            filter=f"doc_id = '{doc_id}'",
            limit=1,
        )
        mime = "image/jpeg"
        if mime_row.num_rows > 0 and mime_row.column("thumbnail_mime")[0].is_valid:
            mime = mime_row.column("thumbnail_mime")[0].as_py()
        blob = docs_ds.take_blobs("thumbnail", indices=[idx])[0]
        with blob as f:
            data = f.read()
        if not data:
            raise HTTPException(status_code=404, detail="no thumbnail for doc_id")
        return Response(
            content=data,
            media_type=mime,
            headers={"Cache-Control": "public, max-age=86400"},
        )

    # ── /api/media/:doc_id  (HTTP Range → BlobFile.seek/read) ─────────────
    @app.get("/api/media/{doc_id}")
    def media(doc_id: str, request: Request) -> Response:
        _valid_doc_id(doc_id)
        if docs_ds is None:
            raise HTTPException(status_code=404, detail="documents table missing")
        idx = _index_for_doc_id(docs_ds, doc_id)
        if idx is None:
            raise HTTPException(status_code=404, detail="doc_id not found")

        mime_row = docs_ds.to_table(
            columns=["media_mime"],
            filter=f"doc_id = '{doc_id}'",
            limit=1,
        )
        mime = "application/octet-stream"
        if mime_row.num_rows > 0 and mime_row.column("media_mime")[0].is_valid:
            mime = mime_row.column("media_mime")[0].as_py()

        total = _doc_blob_size(docs_ds, "media_blob", idx)
        range_hdr = request.headers.get("range")
        if range_hdr:
            rng = _parse_range(range_hdr, total)
            if rng is None:
                return Response(
                    status_code=416,
                    headers={"Content-Range": f"bytes */{total}"},
                )
            start, end = rng
            length = end - start + 1
            return StreamingResponse(
                _stream_blob_range(docs_ds, "media_blob", idx, start, end),
                status_code=206,
                media_type=mime,
                headers={
                    "Content-Length": str(length),
                    "Content-Range": f"bytes {start}-{end}/{total}",
                    "Accept-Ranges": "bytes",
                    "Cache-Control": "no-store",
                },
            )

        return StreamingResponse(
            _stream_blob_range(docs_ds, "media_blob", idx, 0, total - 1),
            media_type=mime,
            headers={
                "Content-Length": str(total),
                "Accept-Ranges": "bytes",
                "Cache-Control": "no-store",
            },
        )

    # Backend is API-only — the Bun frontend serves static assets and proxies
    # /api/* to this service. Enable permissive CORS for the Bun dev server.
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["Content-Range", "Content-Length", "Accept-Ranges"],
    )

    return app


def run(
    db_path: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    """Start the API with uvicorn."""
    import uvicorn

    app = create_app(db_path)
    uvicorn.run(app, host=host, port=port, log_level="info")
