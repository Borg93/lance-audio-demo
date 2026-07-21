"""Batch labeling jobs — enqueue a producer over a read-plane SELECTION (the 3rd mode).

The interactive modes (manual, AI-assist) write locally then merge_insert one version.
Bulk / auto-labeling instead runs a producer over a CHUNK-LEVEL selection (a lasso /
scope / corpus formed in the read plane) as a SILVER DERIVER — async, corpus-scale,
replace-protects-humans. That deriver is lance-ns's engine (lance-ray + the catalog
mover); OUR job is the thin SEAM that submits it and reports status.

Routes to a job runner (``MEDIA_JOBS_URL`` — a lance-ns RayJob submit endpoint) when set,
else a deterministic in-repo MOCK so the submit/poll round-trip is wired + testable
(drop-in for the real submitter, exactly like the assist + catalog transports). No job
runs in-repo — the mock only proves submission + status polling.
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

from fastapi import APIRouter
from pydantic import BaseModel, Field

from backend.core.exceptions import ServiceUnavailableError
from backend.deps import StateDep

if TYPE_CHECKING:
    from backend.state import AppState

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


class JobScope(BaseModel):
    """The chunk-level target set — mirrors the frontend ChunkSelection."""

    level: str  # "chunks" | "scope" | "corpus"
    keys: list[str] = Field(default_factory=list)  # level=chunks: descriptor hitKeys
    where: str | None = None  # level=scope: a SQL WHERE (scope.ts ScopeClause)


class JobRequest(BaseModel):
    producer: str  # grounding-dino | insid3 | vlm-judge | embed-propagate | …
    op: str  # predict | propagate | judge
    scope: JobScope
    dataset: str | None = None
    prompt: str | None = None
    # PROPAGATE (INSID3 / embed-propagate): the few-shot REFERENCE — annotation ids of the
    # exemplar masks. The deriver reads their masks + DINOv3 features and propagates them
    # over `scope`. Empty for predict/judge. This is the seam that carries INSID3's
    # "1–few exemplars → apply to all from small data" through to the lance-ns deriver.
    exemplars: list[str] = Field(default_factory=list)


class JobResult(BaseModel):
    job_id: str
    status: str  # queued | running | succeeded | failed
    backend: str  # "mock" | "remote"


def _scope_size(scope: JobScope) -> int:
    """Item count for a chunk selection; -1 for scope/corpus (server-side count)."""
    return len(scope.keys) if scope.level == "chunks" else -1


def _job_id(body: JobRequest) -> str:
    """Deterministic id from the request — a re-submit of the same op+scope+exemplars is
    idempotent (no wall-clock / randomness, so runs are reproducible). Exemplars are part
    of the identity: propagating a DIFFERENT few-shot reference is a different job."""
    exemplars = ",".join(sorted(body.exemplars))
    basis = (
        f"{body.producer}:{body.op}:{body.scope.level}:{_scope_size(body.scope)}"
        f":{body.dataset}:{exemplars}"
    )
    return "job-" + hashlib.sha1(basis.encode(), usedforsecurity=False).hexdigest()[:12]


@router.post("/apply")
def apply(state: StateDep, body: JobRequest) -> JobResult:
    """Submit a batch labeling deriver over a read-plane selection."""
    n = _scope_size(body.scope)
    logger.info(
        "job apply %s:%s over %s (%s items, %d exemplars)",
        body.producer, body.op, body.scope.level, n, len(body.exemplars),
    )
    url = state.settings.jobs_url
    if url:
        return _remote(state, url, body)
    return JobResult(job_id=_job_id(body), status="queued", backend="mock")


@router.get("/{job_id}")
def status(state: StateDep, job_id: str) -> JobResult:
    """Poll a submitted job. The mock reports 'queued' deterministically; a real runner
    (``MEDIA_JOBS_URL``) reflects the deriver's live state."""
    url = state.settings.jobs_url
    if url:
        return _remote_status(state, url, job_id)
    return JobResult(job_id=job_id, status="queued", backend="mock")


def _remote(state: AppState, url: str, body: JobRequest) -> JobResult:
    """Proxy the submit to the lance-ns RayJob endpoint (WIRED, not exercised in-repo)."""
    http = state.http
    if http is None:
        raise ServiceUnavailableError("jobs: HTTP client unavailable")
    resp = http.post(url, json=body.model_dump(), timeout=30.0)
    resp.raise_for_status()
    data = resp.json()
    return JobResult(job_id=str(data["job_id"]), status=data.get("status", "queued"), backend="remote")


def _remote_status(state: AppState, url: str, job_id: str) -> JobResult:
    http = state.http
    if http is None:
        raise ServiceUnavailableError("jobs: HTTP client unavailable")
    resp = http.get(f"{url.rstrip('/')}/{job_id}", timeout=15.0)
    resp.raise_for_status()
    data = resp.json()
    return JobResult(job_id=job_id, status=data.get("status", "queued"), backend="remote")
