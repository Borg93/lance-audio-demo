"""Normalized search request — pure validation, no Lance/HTTP/embedding deps.

Ported from ``backend.search.spec`` with the corpus-specific filter fields
removed: metadata filters are now read dynamically from the request against the
descriptor's ``search.filterable`` list (see :mod:`backend.search_api.filters`),
so this model carries only the schema-agnostic knobs. ``mode`` is a ``StrEnum``
so FastAPI validates it at the route boundary (422 on an unknown mode); the
numeric fields clamp rather than reject, so the GET/POST handlers construct
``SearchSpec(...)`` directly without a factory.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, field_validator

from backend.lancekit.descriptor import Search


class SearchMode(StrEnum):
    FTS = "fts"
    SEMANTIC = "semantic"
    VISUAL = "visual"
    SCENE = "scene"  # text → cosine over the scene binding's caption-embedding column
    SCENE_FTS = "scene_fts"  # keyword → BM25 over the scene binding's caption_source
    HYBRID = "hybrid"
    ALL = "all"


def available_modes(search: Search) -> list[SearchMode]:
    """The mode set a dataset's search bindings support.

    ``fts`` + ``hybrid`` + ``all`` are always in the set (``all`` fuses whatever
    legs exist); ``semantic`` / ``visual`` / ``scene`` appear iff
    ``search.vectors`` declares a binding under that mode name, and ``scene_fts``
    iff the scene binding names a ``caption_source`` text column.
    """
    modes = [SearchMode.FTS]
    for mode in (SearchMode.SEMANTIC, SearchMode.VISUAL, SearchMode.SCENE):
        if mode.value in search.vectors:
            modes.append(mode)
    scene = search.vectors.get(SearchMode.SCENE.value)
    if scene is not None and scene.caption_source:
        modes.append(SearchMode.SCENE_FTS)
    modes.extend((SearchMode.HYBRID, SearchMode.ALL))
    return modes


class SearchSpec(BaseModel):
    """Normalized search request shared by the GET and POST handlers.

    ``n`` / ``fuzziness`` / ``weight`` are *clamped* (not rejected) so the
    frontend's "load more" can keep raising ``n`` past 100 without erroring.
    ``extra="ignore"`` keeps the wire shape stable: the descriptor-declared
    filter params (and ``dataset``) ride in the same query string but are
    extracted separately at the router.
    """

    model_config = ConfigDict(extra="ignore")

    q: str = ""
    n: int = 20
    mode: SearchMode = SearchMode.FTS
    rerank: bool = False
    # How many of the top results the cross-encoder reranker re-scores (the
    # rerank "head"); the rest keep first-stage order. Only used when
    # ``rerank`` is True. Clamped, never rejected.
    rerank_n: int = 20
    fuzziness: int = 0
    phrase: bool = False
    # weight ∈ [0, 1]: bias toward FTS (0) or vector (1). None = neutral RRF.
    weight: float | None = None
    # Optional separate text for the vector leg of hybrid/semantic/all; falls
    # back to ``q`` when empty. The FTS leg always uses ``q``.
    q_vec: str = ""
    # Raw user-typed SQL WHERE expression, ANDed with the structured metadata
    # filters (e.g. "duration > 60").
    where: str | None = None
    # True => filter applies BEFORE vector/FTS search (prefilter); False => after.
    prefilter: bool = True
    # OPTIONAL dataset selector (None → the default DB). Lives on the model
    # because FastAPI embeds a query-param model whenever a sibling Query param
    # is declared next to it — this keeps the GET wire shape flat. The router
    # resolves it; the service layer never reads it.
    dataset: str | None = None

    @field_validator("q", "q_vec")
    @classmethod
    def _strip(cls, v: str) -> str:
        # Boundary normalization: a whitespace-only query must read as empty so
        # the handlers' empty-input short-circuit (and q_vec's fall-back-to-q
        # contract) hold.
        return v.strip()

    @field_validator("n")
    @classmethod
    def _clamp_n(cls, v: int) -> int:
        return max(1, min(v, 200))

    @field_validator("rerank_n")
    @classmethod
    def _clamp_rerank_n(cls, v: int) -> int:
        return max(1, min(v, 200))

    @field_validator("fuzziness")
    @classmethod
    def _clamp_fuzziness(cls, v: int) -> int:
        return max(0, min(2, v))

    @field_validator("weight")
    @classmethod
    def _clamp_weight(cls, v: float | None) -> float | None:
        return None if v is None else max(0.0, min(1.0, float(v)))


class PostSearchSpec(SearchSpec):
    """The POST (multipart, image upload) variant: defaults to hybrid mode.

    Same fields and clamping as :class:`SearchSpec` — only the default ``mode``
    differs, matching the image-search UX (an upload usually wants fusion).
    """

    mode: SearchMode = SearchMode.HYBRID
