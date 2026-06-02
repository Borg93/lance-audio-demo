"""Framework-free search business logic — the mode-aware retrieval core.

``run_search`` routes a :class:`~backend.search.spec.SearchSpec` across the five
modes (fts / semantic / visual / hybrid / all) and returns a uniform hit shape.
It takes the two vLLM client getters as plain callables, so this module never
imports the FastAPI app or app state — only :class:`HTTPException` for error
mapping. The HTTP routers wire it to the request via dependency injection.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException

from backend.search.spec import SearchSpec

if TYPE_CHECKING:
    from raudio.vllm.embedding import VLLMEmbeddingClient
    from raudio.vllm.reranker import VLLMReranker

_HIT_COLUMNS = [
    "_score",
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
    "alignments_json",
]

# Hit columns without the FTS-only BM25 `_score`. Vector and hybrid searches
# surface `_distance` / `_relevance_score` instead, so selecting `_score` there
# would fail.
_PAYLOAD_COLUMNS = [c for c in _HIT_COLUMNS if c != "_score"]

# IVF_PQ recall knobs. Lance's default probes too few partitions for good recall;
# ~√(num_partitions) partitions plus a refine pass that re-scores the top
# candidates with full-precision vectors restores it at a small latency cost
# (see docs/INVESTIGATION.md §A3). Ignored when the column has no IVF index.
_VECTOR_NPROBES = 20
_VECTOR_REFINE_FACTOR = 3


def _postprocess_hits(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Parse alignments_json JSONB → Python list/dict for each hit."""
    from raudio.retrieval.search import parse_alignments_json

    for h in raw:
        h["alignments"] = parse_alignments_json(h.pop("alignments_json", None))
    return raw


def _sql_quote(value: str) -> str:
    """Escape single quotes for inlining a value in a SQL string literal."""
    return value.replace("'", "''")


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
        clauses.append(f"language = '{_sql_quote(language)}'")
    if namn:
        clauses.append(f"namn LIKE '%{_sql_quote(namn)}%'")
    if referenskod:
        clauses.append(f"referenskod LIKE '%{_sql_quote(referenskod)}%'")
    if extraid:
        clauses.append(f"extraid = '{_sql_quote(extraid)}'")
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


def run_search(
    chunks,  # lancedb sync Table; typed loosely — the query builder isn't on the abstract Table stub
    chunk_frames,  # lancedb Table for frame vectors, or None if frames not extracted
    get_embedder: Callable[
        [], VLLMEmbeddingClient
    ],  # raises HTTPException(503) if the embed server is unreachable
    get_reranker: Callable[
        [], VLLMReranker
    ],  # raises HTTPException(503) if the rerank server is unreachable
    spec: SearchSpec,
    *,
    image_bytes: bytes | None,
) -> list[dict[str, Any]]:
    """Mode-aware search router.

    All paths return the same hit shape (alignments_json parsed into
    `alignments`). The frontend renders one card type for everything.
    """
    where = _build_where_clause(
        language=spec.language,
        namn=spec.namn,
        referenskod=spec.referenskod,
        extraid=spec.extraid,
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
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"search failed: {e}") from e
        return _postprocess_hits(raw)

    # All remaining modes need the embedding client.
    client = get_embedder()

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
    except Exception as e:
        # httpx.ConnectError / httpx.HTTPError / etc. all collapse to one
        # 503 here — the user-actionable message is "vLLM isn't up".
        msg = type(e).__name__
        detail = str(e).splitlines()[0] if str(e) else ""
        raise HTTPException(
            status_code=503,
            detail=f"embedding service unavailable ({msg}): {detail}",
        ) from e

    # ── single-column vector modes ────────────────────────────────
    if spec.mode == "semantic":
        if "text_embedding" not in chunks.schema.names:
            raise HTTPException(
                status_code=400,
                detail="text embeddings not built yet — run `raudio embed-chunks`",
            )
        vec = text_vec if text_vec is not None else image_vec
        return _postprocess_hits(_vector_search(chunks, vec, "text_embedding", spec.n, where))
    if spec.mode == "visual":
        vec = image_vec if image_vec is not None else text_vec
        return _postprocess_hits(_frame_search(chunk_frames, chunks, vec, spec.n, where))

    # ── hybrid (Lance native: FTS + text vector + RRF/rerank) ─────
    if spec.mode == "hybrid":
        if text_vec is None:
            raise HTTPException(status_code=400, detail="hybrid requires text query")
        if "text_embedding" not in chunks.schema.names:
            raise HTTPException(
                status_code=400,
                detail="text embeddings not built yet — run `raudio embed-chunks`",
            )
        try:
            from lancedb.rerankers import LinearCombinationReranker, RRFReranker

            from raudio.vllm.reranker import QwenVLReranker

            # Reranker priority:
            #   1) Cross-encoder rerank (Qwen3-VL-Reranker) when explicitly opted in
            #   2) LinearCombination with the user's weight when one is supplied
            #   3) RRF (parameter-free) — Lance's default for hybrid queries
            if spec.rerank:
                reranker = QwenVLReranker(get_reranker())
            elif spec.weight is not None:
                # weight ∈ [0, 1]: 0 = pure FTS, 1 = pure vector, 0.5 = balanced
                reranker = LinearCombinationReranker(weight=spec.weight)
            else:
                reranker = RRFReranker()
            qb = (
                chunks.search(query_type="hybrid")
                .vector(text_vec.tolist())
                .text(spec.q)
                .rerank(reranker)
                .nprobes(_VECTOR_NPROBES)
                .refine_factor(_VECTOR_REFINE_FACTOR)
                .select(_PAYLOAD_COLUMNS)
                .limit(spec.n)
            )
            if where:
                qb = qb.where(where, prefilter=False)
            raw = qb.to_list()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"hybrid search failed: {e}") from e
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
            rankings.append(_vector_search(chunks, text_vec, "text_embedding", spec.n * 3, where))
        # Frame vector branch — searches the chunk_frames table (same shared
        # 2048-d space), joined back to chunks. Empty until frames are embedded.
        vec_for_frames = image_vec if image_vec is not None else text_vec
        if vec_for_frames is not None:
            rankings.append(_frame_search(chunk_frames, chunks, vec_for_frames, spec.n * 3, where))

        fused = _rrf_fuse(rankings)[: spec.n]
        # Optional cross-encoder rerank on fused top-K
        if spec.rerank and spec.q and fused:
            scores = get_reranker().rerank(spec.q, [h["text"] for h in fused])
            fused = [h for _, h in sorted(zip(scores, fused, strict=False), key=lambda p: -p[0])]
        return _postprocess_hits(fused)

    # Unreachable — SearchSpec validation rejects unknown modes up-front.
    raise AssertionError(f"unhandled mode: {spec.mode!r}")


def _vector_search(
    table,
    vec: Any,
    column: str,
    n: int,
    where: str | None,
) -> list[dict[str, Any]]:
    """Run a cosine vector search on ``column``; returns raw list of dicts.

    Returns ``[]`` when the embedding column doesn't exist yet (embeddings are
    attached post-ingest by ``embed-chunks``), so fusion modes degrade to the
    other rankings instead of erroring.
    """
    if vec is None or column not in table.schema.names:
        return []
    try:
        qb = (
            table.search(vec.tolist(), vector_column_name=column)
            .distance_type("cosine")
            .nprobes(_VECTOR_NPROBES)
            .refine_factor(_VECTOR_REFINE_FACTOR)
            .select([*_PAYLOAD_COLUMNS, "_distance"])
            .limit(n)
        )
        if where:
            qb = qb.where(where, prefilter=False)
        return qb.to_list()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"vector search failed: {e}") from e


def _frame_search(
    chunk_frames,
    chunks,
    vec: Any,
    n: int,
    where: str | None,
) -> list[dict[str, Any]]:
    """Frame-vector search: rank by `chunk_frames.frame_embedding`, then fetch the
    matching `chunks` rows (where the hit payload lives) and re-order to match.

    Returns ``[]`` when frames haven't been extracted/embedded yet — the frame
    pipeline is optional, so visual/all search degrades to empty rather than erroring.
    """
    if vec is None or chunk_frames is None or "frame_embedding" not in chunk_frames.schema.names:
        return []
    try:
        ranked = (
            chunk_frames.search(vec.tolist(), vector_column_name="frame_embedding")
            .distance_type("cosine")
            .nprobes(_VECTOR_NPROBES)
            .refine_factor(_VECTOR_REFINE_FACTOR)
            .select(["doc_id", "speech_id", "chunk_id", "_distance"])
            .limit(n)
            .to_list()
        )
    except Exception:  # noqa: BLE001 — no frame_embedding column/index yet → no frame hits
        return []
    # A chunk may have several frames; the ranking is ascending by distance, so
    # keeping the first occurrence per chunk key collapses to the best frame and
    # yields one hit per chunk. (No-op when each chunk has a single frame.)
    keys: list[tuple[Any, int, int]] = []
    seen: set[tuple[Any, int, int]] = set()
    for r in ranked:
        key = (r["doc_id"], int(r["speech_id"]), int(r["chunk_id"]))
        if key not in seen:
            seen.add(key)
            keys.append(key)
    if not keys:
        return []

    # Fetch the chunk rows for those keys in one scan, then re-order to the frame
    # ranking. doc_id is a sha1 hex and the ids are ints, so the filter is safe.
    key_filter = " OR ".join(
        f"(doc_id = '{d}' AND speech_id = {s} AND chunk_id = {c})" for d, s, c in keys
    )
    full_filter = f"({key_filter})" + (f" AND ({where})" if where else "")
    # A pure metadata filter (no vector/FTS query) must go through the Lance
    # dataset's filter scan — a no-query `table.search().where(...)` is a
    # degenerate empty search that returns nothing.
    try:
        rows = chunks.to_lance().to_table(columns=_PAYLOAD_COLUMNS, filter=full_filter).to_pylist()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"frame search join failed: {e}") from e
    by_key = {(r["doc_id"], int(r["speech_id"]), int(r["chunk_id"])): r for r in rows}
    return [by_key[k] for k in keys if k in by_key]
