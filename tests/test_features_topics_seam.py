"""``ratch feature topics`` → the jobs seam — the argv/token contract.

Replaces the deleted subprocess-client tests: pins what ``_run_topics`` hands
``run_runner`` (resolved absolute ``--db``, ``--llm-url`` only when set, the
canonical token) and that the tree derivation still runs after the job.
"""

from __future__ import annotations

from pathlib import Path

from ratch.features.columns import FEATURES, FeatureRunOptions


def _capture_seam(monkeypatch) -> dict[str, object]:
    """Stub the lazily-imported seam + tree build at their defining modules."""
    import ratch.core.jobs as jobs
    import ratch.features.topic_tree as topic_tree

    seen: dict[str, object] = {}
    monkeypatch.setattr(jobs, "run_runner", lambda job: seen.__setitem__("job", job))
    monkeypatch.setattr(
        topic_tree, "build_topic_tree", lambda db: seen.__setitem__("tree_db", db) or 7
    )
    return seen


class TestTopicsFeatureSeam:
    def test_submits_resolved_db_and_builds_the_tree(self, monkeypatch, tmp_path: Path) -> None:
        seen = _capture_seam(monkeypatch)
        relative = Path(tmp_path.name)  # a relative spelling of tmp_path
        monkeypatch.chdir(tmp_path.parent)

        rows = FEATURES["topics"].run(relative, FeatureRunOptions(), None)

        job = seen["job"]
        resolved = str(tmp_path.resolve())
        assert job.runner == "topics"  # ty: ignore[unresolved-attribute]
        assert job.entrypoint_args == ("--db", resolved)  # ty: ignore[unresolved-attribute]
        assert job.token == resolved  # ty: ignore[unresolved-attribute]
        assert seen["tree_db"] == relative  # tree runs after the job, pure compute
        assert rows == 7  # the feature returns the tree's count

    def test_llm_url_forwarded_only_when_set(self, monkeypatch, tmp_path: Path) -> None:
        seen = _capture_seam(monkeypatch)
        FEATURES["topics"].run(tmp_path, FeatureRunOptions(url="http://gemma:8003/v1"), None)
        args = seen["job"].entrypoint_args  # ty: ignore[unresolved-attribute]
        assert args[-2:] == ("--llm-url", "http://gemma:8003/v1")

    def test_topic_tree_feature_is_pure_compute(self, monkeypatch, tmp_path: Path) -> None:
        seen = _capture_seam(monkeypatch)
        rows = FEATURES["topic_tree"].run(tmp_path, FeatureRunOptions(), None)
        assert rows == 7
        assert "job" not in seen  # no runner job — the tree derives from existing columns
