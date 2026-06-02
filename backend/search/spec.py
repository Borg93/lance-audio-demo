"""Normalized search request — pure validation, no Lance/HTTP/embedding deps.

Importable in isolation (unit-testable). ``mode`` is a ``StrEnum`` so FastAPI
validates it at the route boundary (422 on an unknown mode); the numeric fields
clamp rather than reject, so the GET/POST handlers construct ``SearchSpec(...)``
directly without a factory.
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, field_validator


class SearchMode(StrEnum):
    FTS = "fts"
    SEMANTIC = "semantic"
    VISUAL = "visual"
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
    language: str | None = None
    namn: str | None = None
    referenskod: str | None = None
    extraid: str | None = None
    fuzziness: int = 0
    phrase: bool = False
    # weight ∈ [0, 1]: bias toward FTS (0) or vector (1). None = neutral RRF.
    weight: float | None = None

    @field_validator("n")
    @classmethod
    def _clamp_n(cls, v: int) -> int:
        return max(1, min(v, 100))

    @field_validator("fuzziness")
    @classmethod
    def _clamp_fuzziness(cls, v: int) -> int:
        return max(0, min(2, v))

    @field_validator("weight")
    @classmethod
    def _clamp_weight(cls, v: float | None) -> float | None:
        return None if v is None else max(0.0, min(1.0, float(v)))
