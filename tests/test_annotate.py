"""Annotation Arrow-IPC serving — the annotator's read wire.

The endpoint itself is proven end-to-end (live curl → 200 Arrow stream; playwright
renders the rows on the PixiJS canvas); these pin the serialization contract the
frontend `tableFromIPC` depends on.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import lance
import pyarrow as pa
from backend.media_api.annotate import (
    _EMPTY_SCHEMA,
    NewAnnotation,
    _build_delta,
    _ipc_stream,
    _new_rows,
)

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


def _full_schema() -> pa.Schema:
    """The annotations contract + the chunk identity columns a real table carries."""
    return pa.schema(
        [("doc_id", pa.string()), ("speech_id", pa.int64()), ("chunk_id", pa.int64()), *_EMPTY_SCHEMA]
    )


def test_new_rows_stamps_identity_and_carries_geometry() -> None:
    ident = {"doc_id": "d1", "speech_id": 0, "chunk_id": 19}
    tbl = _new_rows(
        [NewAnnotation(id="n1", shape_type="polygon", polygon=[1.0, 2.0, 3.0, 4.0], label="line")],
        ident,
        _full_schema(),
    )
    r = tbl.to_pylist()[0]
    assert (r["doc_id"], r["speech_id"], r["chunk_id"]) == ("d1", 0, 19)  # identity stamped
    assert r["id"] == "n1" and r["shape_type"] == "polygon" and r["polygon"] == [1.0, 2.0, 3.0, 4.0]
    assert r["status"] == "accepted" and r["source"] == "human"  # human-drawn defaults
    assert r["confidence"] is None  # unspecified column → null via schema


def test_save_edit_insert_delete_round_trip(tmp_path: Path) -> None:
    schema = _full_schema()
    ident = {"doc_id": "d1", "speech_id": 0, "chunk_id": 19}
    uri = str(tmp_path / "annotations.lance")
    lance.write_dataset(
        pa.Table.from_pylist(
            [{**ident, "id": "a", "status": "prediction"}, {**ident, "id": "b", "status": "accepted"}],
            schema=schema,
        ),
        uri,
    )
    ds = lance.dataset(uri)
    v0 = ds.version

    # edit `a` + insert new shape `c` — ONE merge_insert (update + insert)
    delta = pa.concat_tables(
        [
            _build_delta(ds.to_table(), {"a": {"status": "accepted"}}),
            _new_rows([NewAnnotation(id="c", shape_type="rectangle", x=9.0)], ident, schema),
        ]
    )
    ds.merge_insert("id").when_matched_update_all().when_not_matched_insert_all().execute(delta)
    after = lance.dataset(uri)
    got = {r["id"]: r for r in after.to_table().to_pylist()}
    assert got["a"]["status"] == "accepted"  # edit applied
    assert got["c"]["shape_type"] == "rectangle" and got["c"]["x"] == 9.0  # insert applied
    assert got["c"]["doc_id"] == "d1"  # new shape carries the unit identity
    assert after.version == v0 + 1  # edit + insert = one atomic version

    # delete `b`
    after.delete("id IN ('b')")
    final = lance.dataset(uri)
    assert {r["id"] for r in final.to_table().to_pylist()} == {"a", "c"}
    assert final.version == v0 + 2


def test_save_emits_spec_2_0_2_openlineage(tmp_path: Path) -> None:
    from backend.lancekit.lineage_emit import build_save_event

    uri = str(tmp_path / "annotations.lance")
    lance.write_dataset(
        pa.Table.from_pylist(
            [{"doc_id": "d1", "speech_id": 0, "chunk_id": 19, "id": "a", "status": "accepted"}],
            schema=_full_schema(),
        ),
        uri,
    )
    ev = build_save_event(
        ds=lance.dataset(uri), table_uri=uri, table_name="annotations", unit_key="d1/0/19"
    )
    # spec-2-0-2 RunEvent shape
    assert ev["eventType"] == "COMPLETE"
    assert ev["schemaURL"].endswith("2-0-2/OpenLineage.json#/$defs/RunEvent")
    assert ev["producer"].endswith("rmedia")  # our emitter, drop-in with lance-ns constants
    assert ev["job"]["name"] == "annotate.merge_insert"
    assert ev["run"]["runId"]  # deterministic uuid5
    # media unit in, annotations table out
    assert ev["inputs"] == [{"namespace": "media", "name": "d1/0/19"}]
    out = ev["outputs"][0]
    assert out["namespace"] == "media" and out["name"] == "annotations"
    facets = out["facets"]
    assert {"schema", "outputStatistics", "columnLineage", "dataSource"} <= facets.keys()
    # the schema facet carries the annotation columns; columnLineage is non-empty
    names = {f["name"] for f in facets["schema"]["fields"]}
    assert {"id", "status", "polygon", "x"} <= names
    assert facets["columnLineage"]
