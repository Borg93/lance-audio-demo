"""``ratch.core.jobs`` — the runner→Ray Jobs seam (mirrors lance-ns ray_submit).

Pins the contract against a fake ``JobSubmitter``: deterministic uuid5 submission
ids, the ``python -m runners.<name>.worker`` entrypoint, the runtime_env contents
(working_dir + runner pip env + MEDIA_*/AWS_* forwarding, nothing else), the
``kind`` metadata label, re-attach vs delete-and-resubmit semantics, and the
``ray_enabled=False`` in-process fallback.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from ratch.core.jobs import (
    JobsSettings,
    RunnerJob,
    RunnerJobError,
    run_runner,
    submission_id_for,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def fast_settings(**overrides: Any) -> JobsSettings:
    defaults: dict[str, Any] = {
        "ray_enabled": True,
        "ray_poll_interval_seconds": 0.0,
        "ray_job_timeout_seconds": 5.0,
    }
    return JobsSettings(**{**defaults, **overrides})


class FakeJobInfo:
    def __init__(self, message: str | None):
        self.message = message


class FakeSubmitter:
    """Scripted, id-honest double of the ``JobSubmitter`` protocol.

    ``existing=True`` primes a pre-existing job under the same submission id: the
    first submit raises (Ray's duplicate-id contract) and status polls drain
    ``status_script`` (last entry repeats). Polling an id that was never
    submitted nor primed raises, mirroring the real client — so a seam that
    polls the wrong id cannot pass.
    """

    def __init__(
        self,
        *,
        status_script: tuple[str, ...] = ("SUCCEEDED",),
        existing: bool = False,
        failure_message: str | None = None,
    ):
        self.submitted: list[dict[str, Any]] = []
        self.deleted: list[str] = []
        self.polled: list[str] = []
        self._existing = existing
        self._script = list(status_script)
        self._failure_message = failure_message
        self._known: set[str] = set()

    def submit_job(
        self,
        *,
        entrypoint: str,
        submission_id: str,
        runtime_env: dict[str, Any],
        metadata: dict[str, str],
    ) -> str:
        self._known.add(submission_id)
        if self._existing:
            self._existing = False
            raise RuntimeError(f"Submission {submission_id} already exists")
        self.submitted.append(
            {
                "entrypoint": entrypoint,
                "submission_id": submission_id,
                "runtime_env": runtime_env,
                "metadata": metadata,
            }
        )
        return submission_id

    def get_job_status(self, submission_id: str, /) -> str:
        if submission_id not in self._known:
            raise RuntimeError(f"Job {submission_id} does not exist")
        self.polled.append(submission_id)
        return self._script.pop(0) if len(self._script) > 1 else self._script[0]

    def get_job_info(self, submission_id: str, /) -> FakeJobInfo:
        if submission_id not in self._known:
            raise RuntimeError(f"Job {submission_id} does not exist")
        return FakeJobInfo(self._failure_message)

    def delete_job(self, submission_id: str, /) -> bool:
        self.deleted.append(submission_id)
        return True


class TestSubmissionId:
    def test_deterministic_per_runner_and_token(self) -> None:
        job = RunnerJob(runner="topics", token="/data/db.lance")
        assert submission_id_for(job) == submission_id_for(job)
        assert submission_id_for(job).startswith("ratch-topics-")

    def test_varies_by_token_and_runner(self) -> None:
        base = submission_id_for(RunnerJob(runner="topics", token="a"))
        assert base != submission_id_for(RunnerJob(runner="topics", token="b"))
        assert base != submission_id_for(RunnerJob(runner="kg", token="a"))


class TestSubmitPath:
    def test_entrypoint_metadata_and_id(self) -> None:
        fake = FakeSubmitter()
        job = RunnerJob(runner="topics", entrypoint_args=("--db", "/data/db.lance"), token="t")
        run_runner(job, settings=fast_settings(), submitter=fake)

        [submitted] = fake.submitted
        assert submitted["entrypoint"] == "python -m runners.topics.worker --db /data/db.lance"
        assert submitted["metadata"] == {"kind": "topics"}
        assert submitted["submission_id"] == submission_id_for(job)
        assert fake.polled == [submission_id_for(job)]  # the seam awaits the id it submitted

    def test_runtime_env_carries_repo_runner_env_and_filtered_vars(self, monkeypatch) -> None:
        monkeypatch.setenv("MEDIA_S3_ENDPOINT", "http://rustfs:9100")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "shh")
        monkeypatch.setenv("UNRELATED_VAR", "nope")
        fake = FakeSubmitter()
        run_runner(RunnerJob(runner="topics"), settings=fast_settings(), submitter=fake)

        runtime_env = fake.submitted[0]["runtime_env"]
        assert runtime_env["working_dir"] == str(REPO_ROOT)
        assert any("toponymy" in dep for dep in runtime_env["pip"])
        assert runtime_env["env_vars"]["MEDIA_S3_ENDPOINT"] == "http://rustfs:9100"
        assert runtime_env["env_vars"]["AWS_SECRET_ACCESS_KEY"] == "shh"
        assert "UNRELATED_VAR" not in runtime_env["env_vars"]
        assert "PATH" not in runtime_env["env_vars"]

    def test_reattaches_to_a_running_job(self) -> None:
        fake = FakeSubmitter(existing=True, status_script=("RUNNING", "SUCCEEDED"))
        run_runner(RunnerJob(runner="topics", token="t"), settings=fast_settings(), submitter=fake)
        assert fake.submitted == []  # attached, never raced a second job
        assert fake.deleted == []

    @pytest.mark.parametrize("prior", ["FAILED", "STOPPED", "SUCCEEDED"])
    def test_terminal_prior_job_is_deleted_and_resubmitted(self, prior: str) -> None:
        # Deliberate deviation from lance-ns (SUCCEEDED included): ratch's trigger
        # is a human asking for a (re)build, not an event redelivery.
        fake = FakeSubmitter(existing=True, status_script=(prior, "SUCCEEDED"))
        job = RunnerJob(runner="topics", token="t")
        run_runner(job, settings=fast_settings(), submitter=fake)
        assert fake.deleted == [submission_id_for(job)]
        assert len(fake.submitted) == 1

    def test_failed_job_raises_with_the_jobs_own_message(self) -> None:
        fake = FakeSubmitter(status_script=("RUNNING", "FAILED"), failure_message="OOM on worker")
        with pytest.raises(RunnerJobError, match=r"FAILED.*OOM on worker"):
            run_runner(RunnerJob(runner="topics"), settings=fast_settings(), submitter=fake)

    def test_timeout_raises(self) -> None:
        fake = FakeSubmitter(status_script=("RUNNING",))
        settings = fast_settings(ray_job_timeout_seconds=0.05)
        with pytest.raises(RunnerJobError, match="did not finish"):
            run_runner(RunnerJob(runner="topics"), settings=settings, submitter=fake)


class TestInProcessFallback:
    def test_runs_worker_main_with_argv(self, monkeypatch) -> None:
        seen: dict[str, list[str]] = {}

        def main() -> int:
            seen["argv"] = list(sys.argv)
            return 0

        worker = ModuleType("runners.fake.worker")
        worker.main = main  # ty: ignore[unresolved-attribute]
        monkeypatch.setitem(sys.modules, "runners.fake.worker", worker)

        fake = FakeSubmitter()
        argv_before = list(sys.argv)
        run_runner(
            RunnerJob(runner="fake", entrypoint_args=("--db", "x")),
            settings=JobsSettings(ray_enabled=False),
            submitter=fake,
        )
        assert seen["argv"] == ["fake-worker", "--db", "x"]
        assert sys.argv == argv_before  # restored
        assert fake.submitted == []  # the cluster path was never touched

    def test_nonzero_exit_raises(self, monkeypatch) -> None:
        worker = ModuleType("runners.fake.worker")
        worker.main = lambda: 3  # ty: ignore[unresolved-attribute]
        monkeypatch.setitem(sys.modules, "runners.fake.worker", worker)
        with pytest.raises(RunnerJobError, match="exited with 3"):
            run_runner(RunnerJob(runner="fake"), settings=JobsSettings(ray_enabled=False))

    def test_missing_worker_module_is_actionable(self) -> None:
        with pytest.raises(RunnerJobError, match="RATCH_RAY_ENABLED=1"):
            run_runner(RunnerJob(runner="no_such_runner"), settings=JobsSettings(ray_enabled=False))

    def test_missing_model_deps_inside_main_are_actionable(self, monkeypatch) -> None:
        # The topics worker imports toponymy lazily inside main(); in ratch's env
        # that import fails and must surface as the same "wrong env" guidance.
        monkeypatch.setenv("OPENAI_API_KEY", "unset-after")  # worker.main mutates these
        monkeypatch.setenv("OPENAI_BASE_URL", "unset-after")
        job = RunnerJob(runner="topics", entrypoint_args=("--db", "/nonexistent"))
        with pytest.raises(RunnerJobError, match="sealed env"):
            run_runner(job, settings=JobsSettings(ray_enabled=False))
