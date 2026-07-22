"""``ratch maintain`` + ``ratch tag`` — version GC with milestone exemption.

The annotations table commits versions per Save and nothing else prunes
(docs/LANCEDB_SDK_AUDIT.md probe 4): these tests pin the adopted contract —
tagged versions and the latest always survive a prune, everything else goes,
and a pruned version surfaces as a clean 404 through the read plane's checkout.
"""

from __future__ import annotations

from pathlib import Path

import lance
import pyarrow as pa
import pytest
from typer.testing import CliRunner

from annotator.annotations.versions import checkout
from common.core.exceptions import NotFoundError
from ratch.cli._app import app

runner = CliRunner()


@pytest.fixture
def versioned_db(tmp_path: Path) -> Path:
    """A db dir with one 5-version table (v1..v5)."""
    db = tmp_path / "db"
    db.mkdir()
    uri = str(db / "annotations.lance")
    lance.write_dataset(pa.table({"id": ["a"]}), uri)
    for i in range(4):
        lance.write_dataset(pa.table({"id": [f"r{i}"]}), uri, mode="append")
    return db


def _cli(db: Path, *args: str) -> str:
    result = runner.invoke(app, ["--db", str(db), "--table", "annotations", *args])
    assert result.exit_code == 0, result.output
    return result.output


class TestTag:
    def test_tags_latest_by_default_and_lists(self, versioned_db: Path) -> None:
        out = _cli(versioned_db, "tag", "batch-1-reviewed")
        assert "tagged v5" in out
        assert "batch-1-reviewed\tv5" in _cli(versioned_db, "tag", "--list")

    def test_tags_explicit_version_and_deletes(self, versioned_db: Path) -> None:
        _cli(versioned_db, "tag", "old", "--version", "2")
        assert "old\tv2" in _cli(versioned_db, "tag", "--list")
        assert "deleted" in _cli(versioned_db, "tag", "old", "--delete")
        assert "old" not in _cli(versioned_db, "tag", "--list")


class TestMaintain:
    def test_prunes_all_but_tagged_and_latest(self, versioned_db: Path) -> None:
        uri = str(versioned_db / "annotations.lance")
        lance.dataset(uri).tags.create("milestone", 2)
        out = _cli(
            versioned_db, "maintain", "--older-than-days", "0", "--delete-unverified"
        )
        assert "versions 5 -> 2" in out
        survivors = [v["version"] for v in lance.dataset(uri).versions()]
        assert survivors == [2, 5]

    def test_pruned_version_is_a_clean_404_via_checkout(self, versioned_db: Path) -> None:
        uri = str(versioned_db / "annotations.lance")
        _cli(versioned_db, "maintain", "--older-than-days", "0", "--delete-unverified")
        ds = lance.dataset(uri)
        with pytest.raises(NotFoundError):
            checkout(ds, 3)

    def test_default_window_keeps_fresh_versions(self, versioned_db: Path) -> None:
        out = _cli(versioned_db, "maintain")
        assert "versions 5 -> 5" in out
