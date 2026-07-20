"""Annotation Arrow-IPC serving — the annotator's read wire.

The endpoint itself is proven end-to-end (live curl → 200 Arrow stream; playwright
renders the rows on the PixiJS canvas); these pin the serialization contract the
frontend `tableFromIPC` depends on.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import lance
import pyarrow as pa
from backend.media_api.annotate import _EMPTY_SCHEMA, _build_delta, _ipc_stream

if TYPE_CHECKING:
    from pathlib import Path


def _read_ipc(raw: bytes) -> pa.Table:
    with pa.ipc.open_stream(pa.BufferReader(raw)) as reader:
        return reader.read_all()


def test_ipc_stream_roundtrips() -> None:
    tbl = pa.table(
        {
            "id": ["x"],
            "x": pa.array([1.0], pa.float32()),
            "shape_type": ["rectangle"],
        }
    )
    back = _read_ipc(_ipc_stream(tbl))
    assert back.num_rows == 1
    assert back.column("id")[0].as_py() == "x"


def test_empty_schema_is_a_parseable_empty_stream() -> None:
    # A dataset with no annotations table degrades to this — the client must still
    # parse it and render 0, so it carries the columns ArrowDataPlugin reads.
    back = _read_ipc(_ipc_stream(_EMPTY_SCHEMA.empty_table()))
    assert back.num_rows == 0
    names = set(back.schema.names)
    # geometry the PixiJS ArrowDataPlugin reads
    assert {"x", "y", "width", "height", "polygon", "shape_type", "status", "mask"} <= names
    # active-learning columns (the review queue ranks by these)
    assert {"confidence", "uncertainty", "source", "model_version"} <= names


def _ann_table() -> pa.Table:
    """Three annotations: a prediction with a polygon, an accepted one, another prediction."""
    return pa.table(
        {
            "id": ["a", "b", "c"],
            "status": ["prediction", "accepted", "prediction"],
            "label": ["text-line", "figure", "text-line"],
            "text": ["foo", "", "bar"],
            "x": pa.array([1.0, 2.0, 3.0], pa.float32()),
            "polygon": pa.array([[0.0, 0.0, 1.0, 1.0], [], [2.0, 2.0]], pa.list_(pa.float32())),
        }
    )


def test_build_delta_patches_only_editable_fields_and_carries_geometry() -> None:
    current = _ann_table()
    delta = _build_delta(current, {"a": {"status": "accepted"}, "c": {"label": "heading"}})
    # only the two edited rows are in the delta, matched by id
    assert delta.num_rows == 2
    by_id = {r["id"]: r for r in delta.to_pylist()}
    assert by_id["a"]["status"] == "accepted"  # patched
    assert by_id["a"]["polygon"] == [0.0, 0.0, 1.0, 1.0]  # geometry carried forward
    assert by_id["a"]["label"] == "text-line"  # untouched field carried forward
    assert by_id["c"]["label"] == "heading"  # patched
    assert by_id["c"]["x"] == 3.0  # geometry carried forward


def test_merge_insert_save_is_one_atomic_version(tmp_path: Path) -> None:
    uri = str(tmp_path / "annotations.lance")
    lance.write_dataset(_ann_table(), uri)
    ds = lance.dataset(uri)
    v0 = ds.version

    delta = _build_delta(ds.to_table(), {"a": {"status": "accepted"}, "b": {"status": "rejected"}})
    ds.merge_insert("id").when_matched_update_all().execute(delta)

    after = lance.dataset(uri)
    got = {r["id"]: r for r in after.to_table().to_pylist()}
    assert got["a"]["status"] == "accepted"  # persisted
    assert got["b"]["status"] == "rejected"  # persisted
    assert got["c"]["status"] == "prediction"  # untouched row unchanged
    assert got["a"]["polygon"] == [0.0, 0.0, 1.0, 1.0]  # geometry survived the round-trip
    assert after.version == v0 + 1  # exactly one new version
