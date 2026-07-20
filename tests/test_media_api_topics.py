"""Topics endpoint (media_api port) over a synthetic single-row JSONB table.

New coverage for ``backend.media_api.topics`` (the old topics router had no
endpoint tests): ``built: false`` when the capability is undeclared or the
table is absent/empty, the single-JSONB-row contract (hierarchy decoded,
layers/n_chunks surfaced), the per-request re-read, and the vendored
``backend.lancekit.topics_meta`` helpers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lance
import pyarrow as pa
import pytest
from backend.core.config import Settings
from backend.core.handlers import register_handlers
from backend.lancekit.topics_meta import NOISE_LABEL, topic_layer_columns
from backend.media_api import topics
from backend.state import AppState
from fastapi import FastAPI
from fastapi.testclient import TestClient

_DECLARED = {
    "identity": {"key_fields": ["doc_id"]},
    "capabilities": {"topics": "topics.hierarchy"},
}

_TREE = {
    "name": "Alla ämnen",
    "children": [
        {"name": "Politik", "value": 3},
        {"name": NOISE_LABEL, "value": 2},
    ],
}


def _write_topics(db: Path, tree: dict[str, Any], *, layers: int = 2, n_chunks: int = 5) -> None:
    table = pa.table(
        {
            "hierarchy": pa.array([json.dumps(tree, ensure_ascii=False)], type=pa.json_()),
            "layers": pa.array([layers], pa.int32()),
            "n_chunks": pa.array([n_chunks], pa.int64()),
        }
    )
    lance.write_dataset(table, str(db / "topics.lance"))


def _make_app(root: Path, declared: dict) -> FastAPI:
    desc_dir = root / "descriptors"
    desc_dir.mkdir(exist_ok=True)
    (desc_dir / "corpus.json").write_text(json.dumps(declared))
    db = root / "corpus.lance"
    db.mkdir(exist_ok=True)
    settings = Settings(MEDIA_DB=db, MEDIA_DB_ROOT=root, MEDIA_DESCRIPTOR_DIR=desc_dir)
    app = FastAPI()
    register_handlers(app)
    app.include_router(topics.router)
    app.state.resources = AppState(db_path=db, names=[], chunks=None, settings=settings)
    return app


@pytest.fixture
def client_unbuilt(tmp_path):
    return TestClient(_make_app(tmp_path, _DECLARED))


@pytest.fixture
def client_built(tmp_path):
    db = tmp_path / "corpus.lance"
    db.mkdir()
    _write_topics(db, _TREE)
    return TestClient(_make_app(tmp_path, _DECLARED))


class TestBuiltContract:
    def test_absent_table_is_built_false(self, client_unbuilt) -> None:
        r = client_unbuilt.get("/api/topics")
        assert r.status_code == 200
        assert r.json() == {
            "built": False,
            "layers": 0,
            "n_chunks": 0,
            "hierarchy": None,
            "noise_label": NOISE_LABEL,
        }

    def test_undeclared_capability_is_built_false(self, tmp_path) -> None:
        """A topics table on disk but no declared capability → the empty contract."""
        db = tmp_path / "corpus.lance"
        db.mkdir()
        _write_topics(db, _TREE)
        client = TestClient(_make_app(tmp_path, {**_DECLARED, "capabilities": {}}))
        assert client.get("/api/topics").json()["built"] is False

    def test_built_serves_the_single_row(self, client_built) -> None:
        body = client_built.get("/api/topics").json()
        assert body["built"] is True
        assert body["layers"] == 2
        assert body["n_chunks"] == 5
        assert body["hierarchy"] == _TREE  # JSONB decoded to the nested tree
        assert body["noise_label"] == NOISE_LABEL

    def test_rebuild_is_picked_up_without_restart(self, client_unbuilt, tmp_path) -> None:
        """The one-row table is re-read per request, so a topics rebuild lands live."""
        assert client_unbuilt.get("/api/topics").json()["built"] is False
        _write_topics(tmp_path / "corpus.lance", _TREE)
        assert client_unbuilt.get("/api/topics").json()["built"] is True

    def test_hierarchy_column_name_comes_from_descriptor(self, tmp_path) -> None:
        """The JSONB column is whatever the capability declares, not a constant."""
        db = tmp_path / "corpus.lance"
        db.mkdir()
        table = pa.table(
            {
                "tree_json": pa.array([json.dumps(_TREE)], type=pa.json_()),
                "layers": pa.array([2], pa.int32()),
                "n_chunks": pa.array([5], pa.int64()),
            }
        )
        lance.write_dataset(table, str(db / "topic_tree.lance"))
        declared = {**_DECLARED, "capabilities": {"topics": "topic_tree.tree_json"}}
        client = TestClient(_make_app(tmp_path, declared))
        body = client.get("/api/topics").json()
        assert body["built"] is True
        assert body["hierarchy"] == _TREE


class TestDatasetParam:
    def test_unknown_dataset_is_404(self, client_built) -> None:
        assert client_built.get("/api/topics", params={"dataset": "nope"}).status_code == 404

    def test_explicit_default_dataset_matches(self, client_built) -> None:
        implicit = client_built.get("/api/topics").json()
        explicit = client_built.get("/api/topics", params={"dataset": "corpus"}).json()
        assert explicit == implicit


class TestTopicsMeta:
    """The vendored ``backend.lancekit.topics_meta`` helpers."""

    def test_layer_columns_sorted_numerically(self) -> None:
        names = ["topic_l10", "text", "topic_l0", "topic_l2", "doc_topic"]
        assert topic_layer_columns(names) == ["topic_l0", "topic_l2", "topic_l10"]

    def test_no_layer_columns(self) -> None:
        assert topic_layer_columns(["text", "doc_id"]) == []

    def test_noise_label_is_stable(self) -> None:
        # The frontend reads this off the response instead of re-hardcoding it.
        assert NOISE_LABEL == "(övrigt)"
