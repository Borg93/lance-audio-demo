"""Probe response models for ``/livez`` and ``/readyz``."""

from typing import Literal

from pydantic import BaseModel, Field


class Liveness(BaseModel):
    status: Literal["ok"] = "ok"


class Readiness(BaseModel):
    status: Literal["starting", "ready", "shutting_down"]
    # Optional per-component status map (e.g. {"db": "healthy"}); only the
    # ready/shutting-down responses populate it. Defaults empty so the gating
    # responses ``Readiness(status="starting")`` stay constructible.
    components: dict[str, str] = Field(default_factory=dict)
