"""P1.5: every create path applies the lance-media invariants via create_dataset."""

from __future__ import annotations

import json

import lance
import pyarrow as pa
import pytest

from ratch.core.dataset import (
    DESCRIPTOR_METADATA_KEY,
    append_rows,
    create_dataset,
    overwrite_dataset,
    read_descriptor,
)

SCHEMA = pa.schema([pa.field("doc_id", pa.string()), pa.field("n", pa.int32())])
DESCRIPTOR = {"identity": {"key_fields": ["doc_id"]}, "display": {"title": ["doc_id"]}}


@pytest.fixture
def created(tmp_path):
    path = tmp_path / "t.lance"
    create_dataset(path, SCHEMA, descriptor=DESCRIPTOR)
    return path


def test_create_sets_storage_version_2_2(created) -> None:
    assert lance.dataset(str(created)).data_storage_version == "2.2"


def test_create_stamps_descriptor_metadata(created) -> None:
    metadata = lance.dataset(str(created)).schema.metadata
    assert json.loads(metadata[DESCRIPTOR_METADATA_KEY].decode()) == DESCRIPTOR
    assert read_descriptor(created) == DESCRIPTOR


def test_stable_row_ids_survive_appends(created) -> None:
    append_rows(created, pa.table({"doc_id": ["a"], "n": [1]}, schema=SCHEMA))
    append_rows(created, pa.table({"doc_id": ["b"], "n": [2]}, schema=SCHEMA))
    ds = lance.dataset(str(created))
    rowids = ds.to_table(columns=["doc_id"], with_row_id=True)
    assert rowids.num_rows == 2
    # Stable row ids are a manifest feature flag; deleting row 0 must not
    # renumber row 1 (positional addressing would).
    ds.delete("doc_id = 'a'")
    after = lance.dataset(str(created)).to_table(columns=["doc_id"], with_row_id=True)
    b_rowid_before = rowids.filter(pa.compute.equal(rowids["doc_id"], "b"))["_rowid"][0].as_py()
    assert after["_rowid"][0].as_py() == b_rowid_before


def test_overwrite_reapplies_invariants(created) -> None:
    overwrite_dataset(created, pa.table({"doc_id": ["c"], "n": [3]}, schema=SCHEMA), descriptor=DESCRIPTOR)
    ds = lance.dataset(str(created))
    assert ds.data_storage_version == "2.2"
    assert read_descriptor(created) == DESCRIPTOR
    assert ds.count_rows() == 1
