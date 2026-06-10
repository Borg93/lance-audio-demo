"""Decoupled search business logic — the mode-aware retrieval core.

``run_search`` routes a :class:`~backend.search.spec.SearchSpec` across the seven
modes (fts / semantic / visual / scene / scene_fts / hybrid / all) and returns a
uniform hit shape. Each mode is one handler in :data:`_MODE_HANDLERS` (dict
dispatch — adding a mode = adding an entry, not editing a chain); handlers share
a :class:`_SearchContext` and own their rerank policy, while ``run_search`` owns
the WHERE composition, the embedding step, and the uniform postprocess tail.

It takes the two vLLM client getters as plain callables, so this module never
imports the FastAPI app or app state — only domain exceptions from
:mod:`backend.core.exceptions` for error mapping. The HTTP routers wire it to
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

import httpx
from pydantic import BaseModel, ConfigDict

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
# __all__ to keep those imports resolving.
from backend.search.filters import _build_where_clause
from backend.search.frames import _frame_fts_search, _frame_search
from backend.search.postprocess import _postprocess_hits, _rrf_fuse
from backend.search.rerank import _rerank_by_text
from backend.search.spec import SearchMode, SearchSpec
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


class _SearchContext(BaseModel):
    """Everything a mode handler needs, resolved once by ``run_search``.

    The Lance handles aren't Pydantic-validatable, hence
    ``arbitrary_types_allowed``. ``text_vec``/``image_vec`` stay ``None`` for
    the embedding-free modes (fts / scene_fts).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    chunks: Any  # lancedb sync Table
    chunk_frames: Any  # lancedb Table for frame vectors, or None
    get_reranker: Any  # Callable[[], VLLMReranker]
    spec: SearchSpec
    where: str | None
    # Text fed to the cross-encoder reranker = the user's full text intent
    # (keyword + meaning question). The reranker is text-only.
    rerank_query: str
    text_vec: Any | None = None
    image_vec: Any | None = None


def _maybe_rerank(ctx: _SearchContext, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply the optional cross-encoder head rerank per the spec toggle.

    The client getter maps *connection* failures to 503; a failure during the
    rerank call itself (server restarting, mid-request drop) would otherwise
    surface as a raw 500, so it gets the same mapping here.
    """
    if not ctx.spec.rerank:
        return hits
    try:
        return _rerank_by_text(
            ctx.get_reranker, ctx.rerank_query, hits, rerank_n=ctx.spec.rerank_n, n=ctx.spec.n
        )
    except httpx.HTTPError as e:
        logger.warning("rerank call failed", exc_info=True)
        raise ServiceUnavailableError("rerank service unavailable") from e


def _search_fts(ctx: _SearchContext) -> list[dict[str, Any]]:
    """BM25 over ``chunks.text`` — phrase or fuzzy match per the spec."""
    from lancedb.query import MatchQuery, PhraseQuery

    spec = ctx.spec
    if spec.phrase:
        fts_query = PhraseQuery(spec.q, "text")
    else:
        fts_query = MatchQuery(spec.q, "text", fuzziness=spec.fuzziness)
    try:
        qb = ctx.chunks.search(fts_query).select(_HIT_COLUMNS).limit(spec.n)
        if ctx.where:
            qb = qb.where(ctx.where, prefilter=spec.prefilter)
        raw = qb.to_list()
    except Exception as e:
        logger.warning("fts search failed", exc_info=True)
        raise ValidationError("search failed") from e
    return _maybe_rerank(ctx, raw)


def _search_scene_fts(ctx: _SearchContext) -> list[dict[str, Any]]:
    """BM25 over ``chunk_frames.caption``, joined back to chunks."""
    hits = _frame_fts_search(
        ctx.chunk_frames,
        ctx.chunks,
        ctx.spec.q,
        n=ctx.spec.n,
        where=ctx.where,
        scope_where=ctx.spec.where,
    )
    return _maybe_rerank(ctx, hits)


def _search_semantic(ctx: _SearchContext) -> list[dict[str, Any]]:
    """Cosine over ``chunks.text_embedding``."""
    if "text_embedding" not in ctx.chunks.schema.names:
        raise ValidationError(
            "text embeddings not built yet — run `raudio feature text_embedding` (make embed-chunks)"
        )
    vec = ctx.text_vec if ctx.text_vec is not None else ctx.image_vec
    hits = _vector_search(
        ctx.chunks,
        vec,
        "text_embedding",
        n=ctx.spec.n,
        where=ctx.where,
        prefilter=ctx.spec.prefilter,
    )
    return _maybe_rerank(ctx, hits)


def _search_visual(ctx: _SearchContext) -> list[dict[str, Any]]:
    """Frame-image similarity (``chunk_frames.frame_embedding``) → chunk join.

    Image-only: the text reranker has no query text to score, so results keep
    their frame-similarity order regardless of the rerank toggle.
    """
    vec = ctx.image_vec if ctx.image_vec is not None else ctx.text_vec
    return _frame_search(
        ctx.chunk_frames,
        ctx.chunks,
        vec,
        n=ctx.spec.n,
        where=ctx.where,
        scope_where=ctx.spec.where,
    )


def _search_scene(ctx: _SearchContext) -> list[dict[str, Any]]:
    """Rank frames by caption similarity in the shared text-embedding space.

    Falls back to the image vector if that's all we got; degrades to ``[]``
    when ``caption_embedding`` hasn't been built.
    """
    vec = ctx.text_vec if ctx.text_vec is not None else ctx.image_vec
    if vec is None:
        raise ValidationError("scene search requires a query")
    hits = _frame_search(
        ctx.chunk_frames,
        ctx.chunks,
        vec,
        n=ctx.spec.n,
        where=ctx.where,
        column=_CAPTION_EMBED_COLUMN,
        scope_where=ctx.spec.where,
    )
    return _maybe_rerank(ctx, hits)


def _search_hybrid(ctx: _SearchContext) -> list[dict[str, Any]]:
    """Lance-native FTS + text-vector fusion (RRF, or Linear when weighted)."""
    spec = ctx.spec
    if ctx.text_vec is None:
        raise ValidationError("hybrid requires text query")
    if "text_embedding" not in ctx.chunks.schema.names:
        raise ValidationError(
            "text embeddings not built yet — run `raudio feature text_embedding` (make embed-chunks)"
        )
    try:
        from lancedb.query import MatchQuery, PhraseQuery
        from lancedb.rerankers import LinearCombinationReranker, RRFReranker

        # Fusion: LinearCombination with the user's weight (the Balance slider)
        # when supplied, else parameter-free RRF. The cross-encoder rerank, if
        # enabled, is applied to the head afterwards so the rerank window stays
        # independent of the result count.
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
            ctx.chunks.search(query_type="hybrid", vector_column_name="text_embedding")
            .vector(ctx.text_vec.tolist())
            .text(fts_query)
            .rerank(fusion)
            .minimum_nprobes(_VECTOR_NPROBES)
            .maximum_nprobes(_VECTOR_MAX_NPROBES)
            .refine_factor(_VECTOR_REFINE_FACTOR)
            .select(_PAYLOAD_COLUMNS)
            .limit(spec.n)
        )
        if ctx.where:
            qb = qb.where(ctx.where, prefilter=spec.prefilter)
        raw = qb.to_list()
    except Exception as e:
        logger.warning("hybrid search failed", exc_info=True)
        raise ValidationError("hybrid search failed") from e
    return _maybe_rerank(ctx, raw)


def _search_all(ctx: _SearchContext) -> list[dict[str, Any]]:
    """Fuse text-FTS + text-vector + frame-vector + caption-vector via RRF."""
    spec = ctx.spec
    rankings: list[list[dict[str, Any]]] = []
    from lancedb.query import MatchQuery

    # FTS branch (only if we have text)
    if spec.q:
        try:
            fts_hits = (
                ctx.chunks.search(MatchQuery(spec.q, "text", fuzziness=spec.fuzziness))
                .select(_HIT_COLUMNS)
                .limit(spec.n * 3)
            )
            if ctx.where:
                fts_hits = fts_hits.where(ctx.where, prefilter=spec.prefilter)
            rankings.append(fts_hits.to_list())
        except Exception:  # noqa: BLE001
            pass

    # Text vector branch
    if ctx.text_vec is not None:
        rankings.append(
            _vector_search(
                ctx.chunks,
                ctx.text_vec,
                "text_embedding",
                n=spec.n * 3,
                where=ctx.where,
                prefilter=spec.prefilter,
            )
        )
    # Frame vector branch — searches the chunk_frames table (same shared
    # 2048-d space), joined back to chunks. Empty until frames are embedded.
    vec_for_frames = ctx.image_vec if ctx.image_vec is not None else ctx.text_vec
    if vec_for_frames is not None:
        rankings.append(
            _frame_search(
                ctx.chunk_frames,
                ctx.chunks,
                vec_for_frames,
                n=spec.n * 3,
                where=ctx.where,
                scope_where=spec.where,
            )
        )
    # Caption (scene) vector branch — frames whose Swedish caption matches the
    # query text. Text-only (captions live in the text-embedding space); empty
    # until caption_embedding is built.
    if ctx.text_vec is not None:
        rankings.append(
            _frame_search(
                ctx.chunk_frames,
                ctx.chunks,
                ctx.text_vec,
                n=spec.n * 3,
                where=ctx.where,
                column=_CAPTION_EMBED_COLUMN,
                scope_where=spec.where,
            )
        )

    fused = _rrf_fuse(rankings)
    # Optional cross-encoder rerank on the fused head (rerank_n), then trim to
    # spec.n. Without rerank we just take the fused top-n.
    if spec.rerank:
        return _rerank_by_text(
            ctx.get_reranker, ctx.rerank_query, fused, rerank_n=spec.rerank_n, n=spec.n
        )
    return fused[: spec.n]


_MODE_HANDLERS: dict[SearchMode, Callable[[_SearchContext], list[dict[str, Any]]]] = {
    SearchMode.FTS: _search_fts,
    SearchMode.SCENE_FTS: _search_scene_fts,
    SearchMode.SEMANTIC: _search_semantic,
    SearchMode.VISUAL: _search_visual,
    SearchMode.SCENE: _search_scene,
    SearchMode.HYBRID: _search_hybrid,
    SearchMode.ALL: _search_all,
}

#: Modes that need a query vector — only these pay the embedding round-trip.
_EMBEDDING_MODES = frozenset(
    {SearchMode.SEMANTIC, SearchMode.VISUAL, SearchMode.SCENE, SearchMode.HYBRID, SearchMode.ALL}
)


def run_search(
    chunks,  # lancedb sync Table; typed loosely — the query builder isn't on the abstract Table stub
    chunk_frames,  # lancedb Table for frame vectors, or None if frames not extracted
    *,
    get_embedder: Callable[[], VLLMEmbeddingClient],
    get_reranker: Callable[[], VLLMReranker],
    spec: SearchSpec,
    image_bytes: bytes | None,
) -> list[dict[str, Any]]:
    """Mode-aware search router.

    All paths return the same hit shape (alignments_json parsed into
    ``alignments``). The frontend renders one card type for everything. The
    client getters raise :class:`ServiceUnavailableError` (HTTP 503) when their
    vLLM server is unreachable.
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

    # Build query vector(s) for the modes that rank by them. Connection /
    # network errors collapse into a structured 503 so the frontend shows a
    # meaningful message instead of "Internal Server Error". The vector leg may
    # use a distinct query string (spec.q_vec); the FTS leg always uses spec.q.
    text_vec = None
    image_vec = None
    if spec.mode in _EMBEDDING_MODES:
        client = get_embedder()
        vec_text = spec.q_vec or spec.q  # empty q_vec falls back to q
        try:
            if vec_text:
                text_vec = client.embed_text([vec_text])[0]
            if image_bytes:
                image_vec = client.embed_image([image_bytes])[0]
        except Exception as e:
            logger.warning("embedding request failed", exc_info=True)
            raise ServiceUnavailableError("embedding service unavailable") from e

    ctx = _SearchContext(
        chunks=chunks,
        chunk_frames=chunk_frames,
        get_reranker=get_reranker,
        spec=spec,
        where=where,
        rerank_query=" ".join(p for p in (spec.q, spec.q_vec) if p),
        text_vec=text_vec,
        image_vec=image_vec,
    )
    handler = _MODE_HANDLERS.get(spec.mode)
    if handler is None:
        # Unreachable — SearchSpec validation rejects unknown modes up-front.
        raise AssertionError(f"unhandled mode: {spec.mode!r}")
    return _postprocess_hits(handler(ctx), chunk_frames)
