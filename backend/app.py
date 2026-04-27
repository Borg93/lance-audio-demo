"""FastAPI backend for the Lance-backed transcript viewer.

Every endpoint reads from Lance directly — no disk walks, no sidecar JSON. The
``media_blob`` column is Lance Blob V2 External (URI); ``thumbnail`` and
``frame_blob`` are Lance Blob V2 Inline (bytes); ``alignments_json`` is Lance
JSONB. All blob reads go through ``ds.take_blobs(...)`` which returns lazy,
seekable ``BlobFile`` handles, so HTTP Range requests map directly to
``BlobFile.seek(start) + read(length)``.

Search modes (Phase 1+2 of the multimodal plan):

* ``mode=fts``      — Tantivy BM25 over ``chunks.text`` (default for GET).
* ``mode=semantic`` — vector search over ``chunks.text_embedding``.
* ``mode=visual``   — vector search over ``chunks.frame_embedding``
                      (image query → search frames; text query also works since
                      Qwen3-VL puts text and image in the same space).
* ``mode=hybrid``   — Lance native hybrid (FTS + text vector + RRF rerank).
* ``mode=all``      — fuse all three signals via RRF.

The ``rerank=true`` flag swaps the default RRFReranker for our cross-encoder
``QwenVLReranker`` (Qwen3-VL-Reranker-8B). Adds latency, big quality bump.

Run:
    raudio serve --db ./transcripts.lance --port 8000
"""

from __future__ import annotations

import base64
import io
import logging
import re
from pathlib import Path
from typing import Any, Literal, get_args


# Search modes accepted by /api/search. Constants used as the single source
# of truth for both type-checking (Literal) and runtime validation.
SearchMode = Literal["fts", "semantic", "visual", "hybrid", "all"]
_VALID_MODES: frozenset[str] = frozenset(get_args(SearchMode))

import lance
import lancedb
from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
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
    with blob as f:
        f.seek(start)
        remaining = end - start + 1
        while remaining > 0:
            chunk = f.read(min(_STREAM_CHUNK, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


def _index_for_row(ds: "lance.LanceDataset", filter_expr: str) -> int | None:
    """Resolve a SQL filter expression to a single ``_rowid``.

    Returns ``None`` if the filter matches no rows. Callers are responsible
    for sanitizing values that go into ``filter_expr`` (e.g. ``doc_id`` is
    validated up-front by :func:`_valid_doc_id`).
    """
    t = ds.to_table(columns=["doc_id"], filter=filter_expr, with_row_id=True)
    if t.num_rows == 0:
        return None
    return int(t.column("_rowid")[0].as_py())


def _index_for_doc_id(ds: "lance.LanceDataset", doc_id: str) -> int | None:
    return _index_for_row(ds, f"doc_id = '{doc_id}'")


def _index_for_chunk(
    ds: "lance.LanceDataset", doc_id: str, speech_id: int, chunk_id: int
) -> int | None:
    return _index_for_row(
        ds,
        f"doc_id = '{doc_id}' AND speech_id = {int(speech_id)} AND chunk_id = {int(chunk_id)}",
    )


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


# ─────────────────────────────────────────────────────────────────────
# Multimodal search helpers
# ─────────────────────────────────────────────────────────────────────


_HIT_COLUMNS = [
    "_score", "doc_id", "audio_path", "speech_id", "chunk_id",
    "start", "end", "duration", "text", "language",
    "namn", "referenskod", "bildid", "extraid",
    "alignments_json",
]


def _postprocess_hits(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Parse alignments_json JSONB → Python list/dict for each hit."""
    from raudio.search import parse_alignments_json
    for h in raw:
        h["alignments"] = parse_alignments_json(h.pop("alignments_json", None))
    return raw


def _build_where_clause(
    *,
    language: str | None,
    namn: str | None,
    referenskod: str | None,
    extraid: str | None,
) -> str | None:
    """Compose the SQL WHERE clause for metadata filters."""
    clauses: list[str] = []
    if language:
        clauses.append(f"language = '{language.replace(chr(39), chr(39)*2)}'")
    if namn:
        clauses.append(f"namn LIKE '%{namn.replace(chr(39), chr(39)*2)}%'")
    if referenskod:
        clauses.append(f"referenskod LIKE '%{referenskod.replace(chr(39), chr(39)*2)}%'")
    if extraid:
        clauses.append(f"extraid = '{extraid.replace(chr(39), chr(39)*2)}'")
    return " AND ".join(clauses) if clauses else None


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


# ─────────────────────────────────────────────────────────────────────
# App factory
# ─────────────────────────────────────────────────────────────────────


def create_app(db_path: str | Path) -> FastAPI:
    """Build the API-only FastAPI app."""
    app = FastAPI(title="raudio api")
    db_path = Path(db_path)

    db = lancedb.connect(str(db_path))
    names = db.table_names()
    logger.info("opened Lance DB %s — tables: %s", db_path, names)

    if "chunks" not in names:
        raise RuntimeError(f"'chunks' table missing in {db_path}")
    chunks = db.open_table("chunks")

    # We also need a raw lance.LanceDataset for `chunks` to do take_blobs()
    # on `frame_blob` (lancedb.Table doesn't expose blob fetch).
    chunks_ds = lance.dataset(str(db_path / "chunks.lance"))

    # `documents` is optional (only present after `ingest --audio-root …`).
    docs_ds = None
    if "documents" in names:
        docs_ds = lance.dataset(str(db_path / "documents.lance"))

    # `chunk_frames` is the post-Phase-2 home for per-chunk video frames
    # (separate from `chunks` because Lance 4.0 merge_insert crashes on the
    # wide chunks schema). Optional — only present after `extract-chunk-frames`.
    chunk_frames_ds = None
    chunk_frames_tbl = None
    chunk_frames_path = db_path / "chunk_frames.lance"
    if chunk_frames_path.exists():
        chunk_frames_ds = lance.dataset(str(chunk_frames_path))
        if "chunk_frames" in names:
            chunk_frames_tbl = db.open_table("chunk_frames")
        logger.info(
            "opened chunk_frames (%d row(s); has_embeddings=%s)",
            chunk_frames_ds.count_rows(),
            "frame_embedding" in chunk_frames_ds.schema.names,
        )

    # Lazy-init container for the embedding client. We don't connect at
    # startup so the API stays usable for FTS-only workflows even when
    # vLLM isn't running. First semantic-mode call will try to connect
    # and fail with a structured 503 if vLLM is unreachable.
    state: dict[str, Any] = {"client": None}

    def _get_client():
        if state["client"] is not None:
            return state["client"]
        try:
            from raudio.embeddings import make_client
            state["client"] = make_client(backend="vllm")
            return state["client"]
        except Exception as e:  # noqa: BLE001
            logger.exception("failed to initialize embedding client")
            raise HTTPException(
                status_code=503,
                detail=f"embedding service unavailable: {e}",
            )

    # ── /api/search GET (text query, query-string only) ───────────────────
    @app.get("/api/search")
    def search_get(
        q: str = "",
        n: int = 20,
        mode: str = "fts",
        rerank: bool = False,
        language: str | None = None,
        namn: str | None = None,
        referenskod: str | None = None,
        extraid: str | None = None,
        fuzziness: int = 0,
        phrase: bool = False,
        weight: float | None = None,
    ) -> list[dict[str, Any]]:
        spec = SearchSpec(
            q=q.strip(), n=n, mode=mode, rerank=rerank,
            language=language, namn=namn, referenskod=referenskod, extraid=extraid,
            fuzziness=fuzziness, phrase=phrase, weight=weight,
        )
        if not spec.q:
            return []
        return _run_search(chunks, chunks_ds, _get_client, spec, image_bytes=None)

    # ── /api/search POST (multipart, accepts an uploaded image) ──────────
    @app.post("/api/search")
    async def search_post(
        image: UploadFile | None = File(None),
        q: str = Form(""),
        n: int = Form(20),
        mode: str = Form("hybrid"),
        rerank: bool = Form(False),
        weight: float | None = Form(None),
        language: str | None = Form(None),
        namn: str | None = Form(None),
        referenskod: str | None = Form(None),
        extraid: str | None = Form(None),
    ) -> list[dict[str, Any]]:
        spec = SearchSpec(
            q=q.strip(), n=n, mode=mode, rerank=rerank,
            language=language, namn=namn, referenskod=referenskod, extraid=extraid,
            fuzziness=0, phrase=False, weight=weight,
        )
        image_bytes: bytes | None = None
        if image is not None:
            image_bytes = await image.read()
        if not spec.q and not image_bytes:
            return []
        return _run_search(chunks, chunks_ds, _get_client, spec, image_bytes=image_bytes)

    # ── /api/health ───────────────────────────────────────────────────────
    @app.get("/api/health")
    def health() -> dict[str, Any]:
        """Frontend status badge: pings vLLM embed/rerank, reports DB facts."""
        from raudio.embeddings import DEFAULT_EMBED_URL, DEFAULT_RERANK_URL

        def _ping(url: str) -> dict[str, Any]:
            import httpx

            try:
                r = httpx.get(f"{url}/health", timeout=1.5)
                return {"ok": r.status_code == 200, "url": url}
            except Exception as e:  # noqa: BLE001
                return {"ok": False, "url": url, "error": str(e).split("\n")[0][:120]}

        return {
            "db": {
                "path": str(db_path),
                "tables": names,
                "chunks": chunks.count_rows(),
                "documents": docs_ds.count_rows() if docs_ds is not None else 0,
            },
            "embed": _ping(DEFAULT_EMBED_URL),
            "rerank": _ping(DEFAULT_RERANK_URL),
        }

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

    # ── /api/chunk-frame/:doc_id/:speech_id/:chunk_id ────────────────────
    @app.get("/api/chunk-frame/{doc_id}/{speech_id}/{chunk_id}")
    def chunk_frame(doc_id: str, speech_id: int, chunk_id: int) -> Response:
        """Per-chunk representative frame, captured at chunk.start.

        Reads from the new `chunk_frames` table if present; falls back to
        the legacy `chunks.frame_blob` column for older datasets.
        """
        _valid_doc_id(doc_id)

        if chunk_frames_ds is not None:
            # Filter chunk_frames directly by composite key. Row IDs aren't
            # stable across appends here, so we look up via `with_row_id=True`
            # then `take_blobs(ids=[…])`.
            keyed = chunk_frames_ds.to_table(
                columns=["frame_mime"],
                filter=(
                    f"doc_id = '{doc_id}' AND speech_id = {int(speech_id)} "
                    f"AND chunk_id = {int(chunk_id)}"
                ),
                with_row_id=True,
                limit=1,
            )
            if keyed.num_rows == 0:
                raise HTTPException(status_code=404, detail="frame not extracted yet")
            rowid = keyed.column("_rowid")[0].as_py()
            mime = "image/jpeg"
            if keyed.column("frame_mime")[0].is_valid:
                mime = keyed.column("frame_mime")[0].as_py()
            try:
                blob = chunk_frames_ds.take_blobs("frame_blob", ids=[rowid])[0]
            except Exception as e:  # noqa: BLE001
                raise HTTPException(status_code=404, detail=f"no frame: {e}")
            with blob as f:
                data = f.read()
            if not data:
                raise HTTPException(status_code=404, detail="frame body empty")
            return Response(
                content=data,
                media_type=mime,
                headers={"Cache-Control": "public, max-age=86400"},
            )

        # Legacy fallback: read from chunks.frame_blob (pre-Phase-2-v2 datasets).
        idx = _index_for_chunk(chunks_ds, doc_id, speech_id, chunk_id)
        if idx is None:
            raise HTTPException(status_code=404, detail="chunk not found")
        try:
            blob = chunks_ds.take_blobs("frame_blob", indices=[idx])[0]
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=404, detail=f"no frame: {e}")
        with blob as f:
            data = f.read()
        if not data:
            raise HTTPException(status_code=404, detail="frame not extracted yet")
        mime_row = chunks_ds.to_table(
            columns=["frame_mime"],
            filter=(
                f"doc_id = '{doc_id}' AND speech_id = {int(speech_id)} "
                f"AND chunk_id = {int(chunk_id)}"
            ),
            limit=1,
        )
        mime = "image/jpeg"
        if mime_row.num_rows > 0 and mime_row.column("frame_mime")[0].is_valid:
            mime = mime_row.column("frame_mime")[0].as_py()
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
    # /api/* to this service. POST is needed for image uploads.
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["Content-Range", "Content-Length", "Accept-Ranges"],
    )

    return app


# ─────────────────────────────────────────────────────────────────────
# Search core (mode-aware)
# ─────────────────────────────────────────────────────────────────────


class SearchSpec:
    """Normalized search request used by both GET and POST handlers."""
    __slots__ = (
        "q", "n", "mode", "rerank", "language", "namn", "referenskod",
        "extraid", "fuzziness", "phrase", "weight",
    )

    def __init__(
        self, *, q: str, n: int, mode: str, rerank: bool,
        language: str | None, namn: str | None,
        referenskod: str | None, extraid: str | None,
        fuzziness: int, phrase: bool, weight: float | None = None,
    ) -> None:
        if mode not in _VALID_MODES:
            raise HTTPException(
                status_code=400,
                detail=f"unknown mode {mode!r}; expected one of {sorted(_VALID_MODES)}",
            )
        self.q = q
        self.n = max(1, min(n, 100))
        self.mode = mode
        self.rerank = rerank
        self.language = language
        self.namn = namn
        self.referenskod = referenskod
        self.extraid = extraid
        self.fuzziness = max(0, min(2, fuzziness))
        self.phrase = phrase
        # weight ∈ [0, 1]: bias toward FTS (0) or vector (1). None = neutral
        # RRF (parameter-free reciprocal-rank fusion).
        self.weight = (
            None if weight is None else max(0.0, min(1.0, float(weight)))
        )


def _run_search(
    chunks,                       # lancedb.Table
    chunks_ds: "lance.LanceDataset",
    get_client,                   # callable returning EmbeddingClient or raising HTTPException(503)
    spec: SearchSpec,
    *,
    image_bytes: bytes | None,
) -> list[dict[str, Any]]:
    """Mode-aware search router.

    All paths return the same hit shape (alignments_json parsed into
    `alignments`). The frontend renders one card type for everything.
    """
    where = _build_where_clause(
        language=spec.language, namn=spec.namn,
        referenskod=spec.referenskod, extraid=spec.extraid,
    )

    # ── FTS-only (today's path, unchanged behaviour) ──────────────
    if spec.mode == "fts":
        from lancedb.query import MatchQuery, PhraseQuery

        if spec.phrase:
            fts_query = PhraseQuery(spec.q, "text")
        else:
            fts_query = MatchQuery(spec.q, "text", fuzziness=spec.fuzziness)
        try:
            qb = chunks.search(fts_query).select(_HIT_COLUMNS).limit(spec.n)
            if where:
                qb = qb.where(where, prefilter=False)
            raw = qb.to_list()
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"search failed: {e}")
        return _postprocess_hits(raw)

    # All remaining modes need the embedding client.
    client = get_client()

    # Build query vector(s). Convert connection / network errors into a
    # structured 503 so the frontend shows a meaningful message instead
    # of "Internal Server Error".
    text_vec = None
    image_vec = None
    try:
        if spec.q:
            text_vec = client.embed_text([spec.q])[0]
        if image_bytes:
            image_vec = client.embed_image([image_bytes])[0]
    except Exception as e:  # noqa: BLE001
        # httpx.ConnectError / httpx.HTTPError / etc. all collapse to one
        # 503 here — the user-actionable message is "vLLM isn't up".
        msg = type(e).__name__
        detail = str(e).splitlines()[0] if str(e) else ""
        raise HTTPException(
            status_code=503,
            detail=f"embedding service unavailable ({msg}): {detail}",
        )

    # ── single-column vector modes ────────────────────────────────
    if spec.mode == "semantic":
        vec = text_vec if text_vec is not None else image_vec
        return _postprocess_hits(
            _vector_search(chunks, vec, "text_embedding", spec.n, where)
        )
    if spec.mode == "visual":
        vec = image_vec if image_vec is not None else text_vec
        return _postprocess_hits(
            _vector_search(chunks, vec, "frame_embedding", spec.n, where)
        )

    # ── hybrid (Lance native: FTS + text vector + RRF/rerank) ─────
    if spec.mode == "hybrid":
        if text_vec is None:
            raise HTTPException(status_code=400, detail="hybrid requires text query")
        try:
            from lancedb.rerankers import LinearCombinationReranker, RRFReranker
            from raudio.embeddings import get_qwen_vl_reranker

            # Reranker priority:
            #   1) Cross-encoder rerank (Qwen3-VL-Reranker) when explicitly opted in
            #   2) LinearCombination with the user's weight when one is supplied
            #   3) RRF (parameter-free) — Lance's default for hybrid queries
            if spec.rerank:
                reranker = get_qwen_vl_reranker(client)
            elif spec.weight is not None:
                # weight ∈ [0, 1]: 0 = pure FTS, 1 = pure vector, 0.5 = balanced
                reranker = LinearCombinationReranker(weight=spec.weight)
            else:
                reranker = RRFReranker()
            qb = (
                chunks.query()
                .full_text_search(spec.q)
                .nearest_to(text_vec.tolist())
                .rerank(reranker)
                .select(_HIT_COLUMNS)
                .limit(spec.n)
            )
            if where:
                qb = qb.where(where, prefilter=False)
            raw = qb.to_list()
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"hybrid search failed: {e}")
        return _postprocess_hits(raw)

    # ── all: fuse text-FTS + text-vector + frame-vector via RRF ───
    if spec.mode == "all":
        rankings: list[list[dict[str, Any]]] = []
        from lancedb.query import MatchQuery

        # FTS branch (only if we have text)
        if spec.q:
            try:
                fts_hits = (
                    chunks.search(MatchQuery(spec.q, "text", fuzziness=spec.fuzziness))
                    .select(_HIT_COLUMNS)
                    .limit(spec.n * 3)
                )
                if where:
                    fts_hits = fts_hits.where(where, prefilter=False)
                rankings.append(fts_hits.to_list())
            except Exception:  # noqa: BLE001
                pass

        # Text vector branch
        if text_vec is not None:
            rankings.append(
                _vector_search(chunks, text_vec, "text_embedding", spec.n * 3, where)
            )
        # Frame vector branch (uses text_vec or image_vec — same shared space)
        vec_for_frames = image_vec if image_vec is not None else text_vec
        if vec_for_frames is not None:
            rankings.append(
                _vector_search(chunks, vec_for_frames, "frame_embedding", spec.n * 3, where)
            )

        fused = _rrf_fuse(rankings)[: spec.n]
        # Optional cross-encoder rerank on fused top-K
        if spec.rerank and spec.q and fused:
            scores = client.rerank(spec.q, [h["text"] for h in fused])
            fused = [
                h for _, h in sorted(
                    zip(scores, fused), key=lambda p: -p[0]
                )
            ]
        return _postprocess_hits(fused)

    # Unreachable — SearchSpec.__init__ rejects unknown modes up-front.
    raise AssertionError(f"unhandled mode: {spec.mode!r}")


def _vector_search(
    chunks, vec, column: str, n: int, where: str | None,
) -> list[dict[str, Any]]:
    """Run a vector cosine search on ``column``; returns raw list of dicts."""
    if vec is None:
        return []
    try:
        qb = (
            chunks.query()
            .nearest_to(vec.tolist())
            .column(column)
            .distance_type("cosine")
            .select(_HIT_COLUMNS)
            .limit(n)
        )
        if where:
            qb = qb.where(where, prefilter=False)
        return qb.to_list()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"vector search failed: {e}")


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
