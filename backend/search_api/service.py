"""Decoupled search business logic — the mode-aware retrieval core.

Ported from ``backend.search.service``, descriptor-driven: ``run_search`` takes
a :class:`~backend.lancekit.registry.DatasetHandle`, resolves it to a
:class:`~backend.search_api.target.SearchTarget` once, and routes a
:class:`~backend.search_api.spec.SearchSpec` across the seven modes
(fts / semantic / visual / scene / scene_fts / hybrid / all), returning a
uniform hit shape. Each mode is one handler in :data:`_MODE_HANDLERS` (dict
dispatch — adding a mode = adding an entry, not editing a chain); handlers
share a :class:`SearchContext` and own their rerank policy, while
``run_search`` owns the WHERE composition, the embedding step, and the uniform
postprocess tail.

Mode availability follows the descriptor: modes whose vector binding is
undeclared degrade to ``[]`` (visual/scene/scene_fts — optional enrichments)
or 400 (semantic/hybrid — the caller asked for a space that doesn't exist).
Vector legs whose binding lives on the row table search it directly; legs on a
frame table rank there and join back by the identity key fields.

It takes the two vLLM client getters as plain callables, so this module never
imports the FastAPI app or app state — only domain exceptions from
:mod:`backend.core.exceptions` for error mapping. The HTTP router wires it to
the request via dependency injection.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import httpx
from pydantic import BaseModel, ConfigDict

from backend.core.exceptions import ServiceUnavailableError, ValidationError
from backend.search_api.constants import (
    VECTOR_MAX_NPROBES,
    VECTOR_NPROBES,
    VECTOR_REFINE_FACTOR,
)
from backend.search_api.filters import build_where_clause
from backend.search_api.frames import frame_fts_search, frame_search
from backend.search_api.postprocess import postprocess_hits, rrf_fuse
from backend.search_api.rerank import rerank_by_text
from backend.search_api.spec import SearchMode, SearchSpec
from backend.search_api.target import SearchTarget, resolve_target
from backend.search_api.vector import vector_search

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from backend.lancekit.descriptor import FtsBinding, VectorBinding
    from backend.lancekit.registry import DatasetHandle
    from backend.search_api.encoders.embedding import EmbeddingClient
    from backend.search_api.encoders.reranker import VLLMReranker

logger = logging.getLogger(__name__)

__all__ = ["run_search"]


class SearchContext(BaseModel):
    """Everything a mode handler needs, resolved once by ``run_search``.

    The Lance handles (inside ``target``) aren't Pydantic-validatable, hence
    ``arbitrary_types_allowed``. ``text_vec``/``image_vec`` stay ``None`` for
    the embedding-free modes (fts / scene_fts).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    target: SearchTarget
    get_reranker: Any  # Callable[[], VLLMReranker]
    spec: SearchSpec
    where: str | None
    # Text fed to the cross-encoder reranker = the user's full text intent
    # (keyword + meaning question). The reranker is text-only.
    rerank_query: str
    text_vec: Any | None = None
    image_vec: Any | None = None


def _maybe_rerank(ctx: SearchContext, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply the optional cross-encoder head rerank per the spec toggle.

    The client getter maps *connection* failures to 503; a failure during the
    rerank call itself (server restarting, mid-request drop) would otherwise
    surface as a raw 500, so it gets the same mapping here.
    """
    if not ctx.spec.rerank:
        return hits
    try:
        return rerank_by_text(
            ctx.get_reranker,
            ctx.rerank_query,
            hits,
            body_column=ctx.target.body_column,
            rerank_n=ctx.spec.rerank_n,
            n=ctx.spec.n,
        )
    except httpx.HTTPError as e:
        logger.warning("rerank call failed", exc_info=True)
        raise ServiceUnavailableError("rerank service unavailable") from e


def _fts_binding(target: SearchTarget) -> FtsBinding:
    """The FTS binding, required to live on the row table (hits ARE row rows)."""
    fts = target.fts
    if fts is None or fts.table != target.row_table_name:
        raise ValidationError("keyword search is not configured for this dataset")
    return fts


def _query_vec(ctx: SearchContext, binding: VectorBinding) -> Any | None:
    """Pick the query vector a binding prefers, falling back to the other one.

    The bi-encoder maps text and images into one space, so a text query can
    rank an image column (and vice versa) when only one vector exists.
    """
    if binding.query_encoder == "image":
        return ctx.image_vec if ctx.image_vec is not None else ctx.text_vec
    return ctx.text_vec if ctx.text_vec is not None else ctx.image_vec


def _vector_leg(
    ctx: SearchContext, binding: VectorBinding, vec: Any, *, n: int
) -> list[dict[str, Any]]:
    """One ranked vector leg: direct on the row table, or frame-ranked + joined."""
    target = ctx.target
    if binding.table == target.row_table_name:
        return vector_search(
            target.row_tbl,
            vec,
            binding.column,
            payload_columns=target.payload_columns,
            n=n,
            where=ctx.where,
            prefilter=ctx.spec.prefilter,
        )
    return frame_search(
        target.table_for(binding.table),
        target,
        vec,
        column=binding.column,
        n=n,
        where=ctx.where,
        scope_where=ctx.spec.where,
    )


def _search_fts(ctx: SearchContext) -> list[dict[str, Any]]:
    """BM25 over the declared FTS column — phrase or fuzzy match per the spec."""
    from lancedb.query import MatchQuery, PhraseQuery

    spec = ctx.spec
    fts = _fts_binding(ctx.target)
    if spec.phrase:
        fts_query = PhraseQuery(spec.q, fts.column)
    else:
        fts_query = MatchQuery(spec.q, fts.column, fuzziness=spec.fuzziness)
    try:
        qb = (
            ctx.target.row_tbl.search(fts_query)
            .select(["_score", *ctx.target.payload_columns])
            .limit(spec.n)
        )
        if ctx.where:
            qb = qb.where(ctx.where, prefilter=spec.prefilter)
        raw = qb.to_list()
    except Exception as e:
        logger.warning("fts search failed", exc_info=True)
        raise ValidationError("search failed") from e
    return _maybe_rerank(ctx, raw)


def _search_scene_fts(ctx: SearchContext) -> list[dict[str, Any]]:
    """BM25 over the scene binding's caption column, joined back to the row table."""
    scene = ctx.target.binding(SearchMode.SCENE)
    if scene is None or not scene.caption_source:
        return []
    hits = frame_fts_search(
        ctx.target.table_for(scene.table),
        ctx.target,
        ctx.spec.q,
        column=scene.caption_source,
        n=ctx.spec.n,
        where=ctx.where,
        scope_where=ctx.spec.where,
    )
    return _maybe_rerank(ctx, hits)


def _search_semantic(ctx: SearchContext) -> list[dict[str, Any]]:
    """Cosine over the semantic binding's vector column."""
    binding = ctx.target.binding(SearchMode.SEMANTIC)
    if binding is None:
        raise ValidationError("text embeddings are not built for this dataset")
    hits = _vector_leg(ctx, binding, _query_vec(ctx, binding), n=ctx.spec.n)
    return _maybe_rerank(ctx, hits)


def _search_visual(ctx: SearchContext) -> list[dict[str, Any]]:
    """Frame-image similarity → row-table join.

    Image-only ranking: the text reranker has no query text to score, so
    results keep their frame-similarity order regardless of the rerank toggle.
    Degrades to ``[]`` when the dataset declares no visual binding.
    """
    binding = ctx.target.binding(SearchMode.VISUAL)
    if binding is None:
        return []
    return _vector_leg(ctx, binding, _query_vec(ctx, binding), n=ctx.spec.n)


def _search_scene(ctx: SearchContext) -> list[dict[str, Any]]:
    """Rank frames by caption similarity in the shared text-embedding space.

    Falls back to the image vector if that's all we got; degrades to ``[]``
    when the dataset declares no scene binding.
    """
    binding = ctx.target.binding(SearchMode.SCENE)
    if binding is None:
        return []
    vec = _query_vec(ctx, binding)
    if vec is None:
        raise ValidationError("scene search requires a query")
    hits = _vector_leg(ctx, binding, vec, n=ctx.spec.n)
    return _maybe_rerank(ctx, hits)


def _search_hybrid(ctx: SearchContext) -> list[dict[str, Any]]:
    """Lance-native FTS + text-vector fusion (RRF, or Linear when weighted)."""
    spec = ctx.spec
    target = ctx.target
    if ctx.text_vec is None:
        raise ValidationError("hybrid requires text query")
    fts = _fts_binding(target)
    binding = target.binding(SearchMode.SEMANTIC)
    # Lance's hybrid query runs both legs on ONE table, so the semantic binding
    # must live on the row table alongside the FTS column.
    if binding is None or binding.table != target.row_table_name:
        raise ValidationError("text embeddings are not built for this dataset")
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
            fts_query = PhraseQuery(spec.q, fts.column)
        else:
            fts_query = MatchQuery(spec.q, fts.column, fuzziness=spec.fuzziness)
        # Name the vector column explicitly: the row table may carry several
        # vector columns, so Lance's hybrid query can't auto-pick which to search.
        qb = (
            target.row_tbl.search(query_type="hybrid", vector_column_name=binding.column)
            .vector(ctx.text_vec.tolist())
            .text(fts_query)
            .rerank(fusion)
            .minimum_nprobes(VECTOR_NPROBES)
            .maximum_nprobes(VECTOR_MAX_NPROBES)
            .refine_factor(VECTOR_REFINE_FACTOR)
            .select(target.payload_columns)
            .limit(spec.n)
        )
        if ctx.where:
            qb = qb.where(ctx.where, prefilter=spec.prefilter)
        raw = qb.to_list()
    except Exception as e:
        logger.warning("hybrid search failed", exc_info=True)
        raise ValidationError("hybrid search failed") from e
    return _maybe_rerank(ctx, raw)


def _search_all(ctx: SearchContext) -> list[dict[str, Any]]:
    """Fuse the FTS + semantic + visual + scene legs via RRF (whatever exists)."""
    spec = ctx.spec
    target = ctx.target
    rankings: list[list[dict[str, Any]]] = []
    from lancedb.query import MatchQuery

    # FTS leg (only if we have text and a row-table FTS binding).
    fts = target.fts
    if spec.q and fts is not None and fts.table == target.row_table_name:
        try:
            qb = (
                target.row_tbl.search(MatchQuery(spec.q, fts.column, fuzziness=spec.fuzziness))
                .select(["_score", *target.payload_columns])
                .limit(spec.n * 3)
            )
            if ctx.where:
                qb = qb.where(ctx.where, prefilter=spec.prefilter)
            rankings.append(qb.to_list())
        except Exception:  # noqa: BLE001 — a missing FTS index just drops this leg
            pass

    # Semantic (text-vector) leg.
    semantic = target.binding(SearchMode.SEMANTIC)
    if ctx.text_vec is not None and semantic is not None:
        rankings.append(_vector_leg(ctx, semantic, ctx.text_vec, n=spec.n * 3))

    # Visual (frame-vector) leg — image query preferred, text fallback (shared space).
    visual = target.binding(SearchMode.VISUAL)
    if visual is not None:
        vec_for_frames = ctx.image_vec if ctx.image_vec is not None else ctx.text_vec
        if vec_for_frames is not None:
            rankings.append(_vector_leg(ctx, visual, vec_for_frames, n=spec.n * 3))

    # Scene (caption-vector) leg — text-only (captions live in the text space).
    scene = target.binding(SearchMode.SCENE)
    if ctx.text_vec is not None and scene is not None:
        rankings.append(_vector_leg(ctx, scene, ctx.text_vec, n=spec.n * 3))

    fused = rrf_fuse(rankings, key_fields=target.key_fields)
    # Optional cross-encoder rerank on the fused head (rerank_n), then trim to
    # spec.n. Without rerank we just take the fused top-n.
    return _maybe_rerank(ctx, fused)[: spec.n]


_MODE_HANDLERS: dict[SearchMode, Callable[[SearchContext], list[dict[str, Any]]]] = {
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
    handle: DatasetHandle,
    *,
    get_embedder: Callable[[], EmbeddingClient],
    get_reranker: Callable[[], VLLMReranker],
    spec: SearchSpec,
    filters: Mapping[str, str] | None = None,
    image_bytes: bytes | None,
) -> list[dict[str, Any]]:
    """Mode-aware search router over one dataset.

    ``filters`` maps descriptor-declared filterable field names to values (the
    router extracts them from the request). All paths return the same hit shape
    (``alignments`` parsed, ``caption`` attached when declared). The client
    getters raise :class:`ServiceUnavailableError` (HTTP 503) when their vLLM
    server is unreachable.
    """
    target = resolve_target(handle)
    where = build_where_clause(
        filters=filters or {},
        filterable=target.filterable,
        topic_columns=target.topic_columns,
        raw=spec.where,
    )

    # ── Filter-only browse (no query text or image) — list rows matching the
    # WHERE clause, e.g. a topic facet clicked on the Tree page. Nothing to
    # rank, so it's a plain Lance scan; no filter means nothing to list. ──
    if not spec.q and not spec.q_vec and image_bytes is None:
        if not where:
            return []
        try:
            raw = target.row_ds.to_table(
                columns=target.payload_columns, filter=where, limit=spec.n
            ).to_pylist()
        except Exception as e:
            logger.warning("browse scan failed", exc_info=True)
            raise ValidationError("browse failed") from e
        return postprocess_hits(raw, target)

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

    ctx = SearchContext(
        target=target,
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
    return postprocess_hits(handler(ctx), target)
