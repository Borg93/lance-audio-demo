"""Batch labeling jobs — the mock submitter's contract.

The endpoint routes to a lance-ns RayJob runner when MEDIA_JOBS_URL is set, else this
deterministic mock so the submit/poll round-trip is testable in-repo. No deriver runs
here — these pin the enqueue seam (id determinism, status, scope sizing).
"""

from __future__ import annotations

from backend.media_api.jobs import JobRequest, JobScope, _job_id, _scope_size


def _req(
    producer: str = "grounding-dino",
    op: str = "predict",
    scope: JobScope | None = None,
) -> JobRequest:
    return JobRequest(
        producer=producer,
        op=op,
        scope=scope or JobScope(level="chunks", keys=["d/0/1", "d/0/2"]),
    )


def test_job_id_is_deterministic_for_the_same_request() -> None:
    # A re-submit of the same op+scope is idempotent (no wall-clock / randomness).
    assert _job_id(_req()) == _job_id(_req())
    assert _job_id(_req()).startswith("job-")


def test_job_id_varies_with_producer_op_and_scope() -> None:
    base = _job_id(_req())
    assert _job_id(_req(producer="insid3")) != base
    assert _job_id(_req(op="propagate")) != base
    assert _job_id(_req(scope=JobScope(level="corpus"))) != base


def test_scope_size_counts_chunks_and_flags_scope_corpus() -> None:
    assert _scope_size(JobScope(level="chunks", keys=["a", "b", "c"])) == 3
    assert _scope_size(JobScope(level="scope", where="doc_id = 'x'")) == -1
    assert _scope_size(JobScope(level="corpus")) == -1
