"""Normalized search request — pure validation, no Lance/HTTP/embedding deps.

Importable in isolation (unit-testable). ``mode`` is a ``StrEnum`` so FastAPI
validates it at the route boundary (422 on an unknown mode); the numeric fields
clamp rather than reject, so the GET/POST handlers construct ``SearchSpec(...)``
directly without a factory.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, field_validator


class SearchMode(StrEnum):
    FTS = "fts"
    SEMANTIC = "semantic"
    VISUAL = "visual"
    SCENE = "scene"  # text → cosine over chunk_frames.caption_embedding (Swedish captions)
    SCENE_FTS = "scene_fts"  # keyword → BM25 over chunk_frames.caption
    HYBRID = "hybrid"
    ALL = "all"


class SearchSpec(BaseModel):
    """Normalized search request shared by the GET and POST handlers.

    ``n`` / ``fuzziness`` / ``weight`` are *clamped* (not rejected) so the
    frontend's "load more" can keep raising ``n`` past 100 without erroring.
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
    language: str | None = None
    namn: str | None = None
    referenskod: str | None = None
    extraid: str | None = None
    # Topic facet (Tree page): a topic name matched exactly against any topic_l*
    # layer column on chunks (the layers are nested, so one name filters the
    # chunks tagged with it at whatever layer the user clicked).
    topic: str | None = None
    fuzziness: int = 0
    phrase: bool = False
    # weight ∈ [0, 1]: bias toward FTS (0) or vector (1). None = neutral RRF.
    weight: float | None = None
    # Optional separate text for the vector leg of hybrid/semantic/all; falls
    # back to ``q`` when empty. The FTS leg always uses ``q``.
    q_vec: str = ""
    # Raw user-typed SQL WHERE expression, ANDed with the structured metadata
    # filters (e.g. "duration > 60 AND namn LIKE '%alkohol%'").
    where: str | None = None
    # True => filter applies BEFORE vector/FTS search (prefilter); False => after.
    prefilter: bool = True

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
