"""Decoupled search business logic — the mode-aware retrieval core.

``run_search`` routes a :class:`~backend.search.spec.SearchSpec` across the seven
modes (fts / semantic / visual / scene / scene_fts / hybrid / all) and returns a
uniform hit shape. It takes the two vLLM client getters as plain callables, so
this module never imports the FastAPI app or app state — only domain exceptions
from :mod:`backend.core.exceptions` for error mapping. The HTTP routers wire it to
the request via dependency injection.

The retrieval mechanics are split into cohesive sibling modules
(``constants`` / ``filters`` / ``postprocess`` / ``rerank`` / ``vector`` /
``frames``); this module owns ONLY the mode dispatch and re-exports the
test-facing names (so test imports of ``_build_where_clause``, ``_rrf_fuse``,
``_frame_search``, ``_postprocess_hits``, ``_vector_search`` keep resolving
from here). Cross-package consumers import public names from the sibling
modules directly (e.g. ``backend.search.postprocess.attach_captions``).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from backend.core.exceptions import ServiceUnavailableError, ValidationError
from backend.search.constants import (
    _CAPTION_EMBED_COLUMN,
    _HIT_COLUMNS,
    _PAYLOAD_COLUMNS,
    _VECTOR_MAX_NPROBES,
    _VECTOR_NPROBES,
    _VECTOR_REFINE_FACTOR,
)

# Re-export block: tests import a few private helpers by name from this module;
# they live in the split sibling modules now, so we import and list them in
# __all__ to keep those imports resolving. (`_frame_fts_search` and
# `_rerank_by_text` below are ordinary body-used imports, not re-exports.)
from backend.search.filters import _build_where_clause
from backend.search.frames import _frame_fts_search, _frame_search
from backend.search.postprocess import _postprocess_hits, _rrf_fuse
from backend.search.rerank import _rerank_by_text
from backend.search.spec import SearchSpec
from backend.search.vector import _vector_search
from raudio.features.topic_tree import topic_layer_columns

if TYPE_CHECKING:
    from raudio.vllm.embedding import VLLMEmbeddingClient
    from raudio.vllm.reranker import VLLMReranker

logger = logging.getLogger(__name__)

# `run_search` is the public entrypoint; the underscore names are test-facing
# re-exports (test imports depend on them — do not remove).
__all__ = [
    "_build_where_clause",
    "_frame_search",
    "_postprocess_hits",
    "_rrf_fuse",
    "_vector_search",
    "run_search",
]


def run_search(
    chunks,  # lancedb sync Table; typed loosely — the query builder isn't on the abstract Table stub
    chunk_frames,  # lancedb Table for frame vectors, or None if frames not extracted
    get_embedder: Callable[
        [], VLLMEmbeddingClient
    ],  # raises ServiceUnavailableError (HTTP 503) if the embed server is unreachable
    get_reranker: Callable[
        [], VLLMReranker
    ],  # raises ServiceUnavailableError (HTTP 503) if the rerank server is unreachable
    spec: SearchSpec,
    *,
    image_bytes: bytes | None,
) -> list[dict[str, Any]]:
    """Mode-aware search router.

    All paths return the same hit shape (alignments_json parsed into
    `alignments`). The frontend renders one card type for everything.
    """
    # Reuse the tree builder's source of truth so the filter columns can't drift
    # from the topic layers the hierarchy/atlas are built on.
    topic_columns = topic_layer_columns(list(chunks.schema.names)) if spec.topic else None
    where = _build_where_clause(
        language=spec.language,
        namn=spec.namn,
        referenskod=spec.referenskod,
        extraid=spec.extraid,
        topic=spec.topic,
        topic_columns=topic_columns,
        raw=spec.where,
    )

    # Text fed to the cross-encoder reranker = the user's full text intent
    # (keyword + meaning question). The reranker is text-only: it never sees the
    # image or the vectors, only this string vs each candidate's transcript.
    rerank_query = " ".join(p for p in (spec.q, spec.q_vec) if p)

    # ── Filter-only browse (no query text or image) — list rows matching the
    # WHERE clause, e.g. a topic facet clicked on the Tree page. Nothing to
    # rank, so it's a plain Lance scan; no filter means nothing to list. ──
    if not spec.q and not spec.q_vec and image_bytes is None:
        if not where:
            return []
        try:
            raw = (
                chunks.to_lance()
                .to_table(columns=_PAYLOAD_COLUMNS, filter=where, limit=spec.n)
                .to_pylist()
            )
        except Exception as e:
            logger.warning("browse scan failed", exc_info=True)
            raise ValidationError("browse failed") from e
        return _postprocess_hits(raw, chunk_frames)

    # ── FTS ───────────────────────────────────────────────────────
    if spec.mode == "fts":
        from lancedb.query import MatchQuery, PhraseQuery

        if spec.phrase:
            fts_query = PhraseQuery(spec.q, "text")
        else:
            fts_query = MatchQuery(spec.q, "text", fuzziness=spec.fuzziness)
        try:
            qb = chunks.search(fts_query).select(_HIT_COLUMNS).limit(spec.n)
            if where:
                qb = qb.where(where, prefilter=spec.prefilter)
            raw = qb.to_list()
        except Exception as e:
            logger.warning("fts search failed", exc_info=True)
            raise ValidationError("search failed") from e
        if spec.rerank:
            raw = _rerank_by_text(get_reranker, rerank_query, raw, spec.rerank_n, spec.n)
        return _postprocess_hits(raw, chunk_frames)

    # ── Scene keyword (BM25 over chunk_frames.caption) — also embedding-free ──
    if spec.mode == "scene_fts":
        hits = _frame_fts_search(
            chunk_frames, chunks, spec.q, spec.n, where, scope_where=spec.where
        )
        if spec.rerank:
            hits = _rerank_by_text(get_reranker, rerank_query, hits, spec.rerank_n, spec.n)
        return _postprocess_hits(hits, chunk_frames)

    # All remaining modes need the embedding client.
    client = get_embedder()

    # Build query vector(s). Convert connection / network errors into a
    # structured 503 so the frontend shows a meaningful message instead
    # of "Internal Server Error".
    # The vector leg may use a distinct query string (spec.q_vec); the FTS leg
    # always uses spec.q. Empty q_vec falls back to q.
    vec_text = spec.q_vec or spec.q
    text_vec = None
    image_vec = None
    try:
        if vec_text:
            text_vec = client.embed_text([vec_text])[0]
        if image_bytes:
            image_vec = client.embed_image([image_bytes])[0]
    except Exception as e:
        # httpx.ConnectError / httpx.HTTPError / etc. all collapse to one
        # 503 here — the user-actionable message is "vLLM isn't up".
        logger.warning("embedding request failed", exc_info=True)
        raise ServiceUnavailableError("embedding service unavailable") from e

    # ── single-column vector modes ────────────────────────────────
    if spec.mode == "semantic":
        if "text_embedding" not in chunks.schema.names:
            raise ValidationError(
                "text embeddings not built yet — run `raudio feature text_embedding` (make embed-chunks)"
            )
        vec = text_vec if text_vec is not None else image_vec
        hits = _vector_search(
            chunks, vec, "text_embedding", spec.n, where, prefilter=spec.prefilter
        )
        if spec.rerank:
            hits = _rerank_by_text(get_reranker, rerank_query, hits, spec.rerank_n, spec.n)
        return _postprocess_hits(hits, chunk_frames)
    if spec.mode == "visual":
        # Image-only: the text reranker has no query text to score, so results
        # keep their frame-similarity order regardless of the rerank toggle.
        vec = image_vec if image_vec is not None else text_vec
        return _postprocess_hits(
            _frame_search(chunk_frames, chunks, vec, spec.n, where, scope_where=spec.where),
            chunk_frames,
        )
    if spec.mode == "scene":
        # Rank frames by how well their Swedish caption matches the query, in the
        # shared text-embedding space. Falls back to the image vector if that's
        # all we got. Degrades to [] when caption_embedding hasn't been built.
        vec = text_vec if text_vec is not None else image_vec
        if vec is None:
            raise ValidationError("scene search requires a query")
        hits = _frame_search(
            chunk_frames,
            chunks,
            vec,
            spec.n,
            where,
            column=_CAPTION_EMBED_COLUMN,
            scope_where=spec.where,
        )
        if spec.rerank:
            hits = _rerank_by_text(get_reranker, rerank_query, hits, spec.rerank_n, spec.n)
        return _postprocess_hits(hits, chunk_frames)

    # ── hybrid (Lance native FTS + text vector, fused by RRF/Linear) ─────
    if spec.mode == "hybrid":
        if text_vec is None:
            raise ValidationError("hybrid requires text query")
        if "text_embedding" not in chunks.schema.names:
            raise ValidationError(
                "text embeddings not built yet — run `raudio feature text_embedding` (make embed-chunks)"
            )
        try:
            from lancedb.query import MatchQuery, PhraseQuery
            from lancedb.rerankers import LinearCombinationReranker, RRFReranker

            # Fusion: LinearCombination with the user's weight (the Balance
            # slider) when supplied, else parameter-free RRF. The cross-encoder
            # rerank, if enabled, is applied to the head afterwards (below) so
            # the rerank window stays independent of the result count.
            if spec.weight is not None:
                # weight ∈ [0, 1]: 0 = pure FTS, 1 = pure vector, 0.5 = balanced
                fusion = LinearCombinationReranker(weight=spec.weight)
            else:
                fusion = RRFReranker()
            # FTS leg honours phrase/fuzziness like the dedicated 'fts' branch.
            if spec.phrase:
                fts_query = PhraseQuery(spec.q, "text")
            else:
                fts_query = MatchQuery(spec.q, "text", fuzziness=spec.fuzziness)
            # Name the vector column explicitly: `chunks` also carries a
            # `frame_embedding` column (for the image-atlas), so Lance's hybrid
            # query can't auto-pick which vector column to search.
            qb = (
                chunks.search(query_type="hybrid", vector_column_name="text_embedding")
                .vector(text_vec.tolist())
                .text(fts_query)
                .rerank(fusion)
                .minimum_nprobes(_VECTOR_NPROBES)
                .maximum_nprobes(_VECTOR_MAX_NPROBES)
                .refine_factor(_VECTOR_REFINE_FACTOR)
                .select(_PAYLOAD_COLUMNS)
                .limit(spec.n)
            )
            if where:
                qb = qb.where(where, prefilter=spec.prefilter)
            raw = qb.to_list()
        except Exception as e:
            logger.warning("hybrid search failed", exc_info=True)
            raise ValidationError("hybrid search failed") from e
        if spec.rerank:
            raw = _rerank_by_text(get_reranker, rerank_query, raw, spec.rerank_n, spec.n)
        return _postprocess_hits(raw, chunk_frames)

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
                    fts_hits = fts_hits.where(where, prefilter=spec.prefilter)
                rankings.append(fts_hits.to_list())
            except Exception:  # noqa: BLE001
                pass

        # Text vector branch
        if text_vec is not None:
            rankings.append(
                _vector_search(
                    chunks, text_vec, "text_embedding", spec.n * 3, where, prefilter=spec.prefilter
                )
            )
        # Frame vector branch — searches the chunk_frames table (same shared
        # 2048-d space), joined back to chunks. Empty until frames are embedded.
        vec_for_frames = image_vec if image_vec is not None else text_vec
        if vec_for_frames is not None:
            rankings.append(
                _frame_search(
                    chunk_frames, chunks, vec_for_frames, spec.n * 3, where, scope_where=spec.where
                )
            )
        # Caption (scene) vector branch — frames whose Swedish caption matches the
        # query text. Text-only (captions live in the text-embedding space); empty
        # until caption_embedding is built.
        if text_vec is not None:
            rankings.append(
                _frame_search(
                    chunk_frames,
                    chunks,
                    text_vec,
                    spec.n * 3,
                    where,
                    column=_CAPTION_EMBED_COLUMN,
                    scope_where=spec.where,
                )
            )

        fused = _rrf_fuse(rankings)
        # Optional cross-encoder rerank on the fused head (rerank_n), then trim
        # to spec.n. Without rerank we just take the fused top-n.
        if spec.rerank:
            fused = _rerank_by_text(get_reranker, rerank_query, fused, spec.rerank_n, spec.n)
        else:
            fused = fused[: spec.n]
        return _postprocess_hits(fused, chunk_frames)

    # Unreachable — SearchSpec validation rejects unknown modes up-front.
    raise AssertionError(f"unhandled mode: {spec.mode!r}")
