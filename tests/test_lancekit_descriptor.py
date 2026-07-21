"""P2.2: descriptor cross-checks hold against LIVE schemas, not just themselves.

Unit half: a synthetic Lance table proves each invariant trips (missing
identity field, non-blob media column, wrong vector dim). Integration half:
the shipped ``config/descriptors/transcripts_v2.json`` validates against the
real corpus when it is present (repo convention: skip otherwise).
"""

from __future__ import annotations

from pathlib import Path

import lance
import pyarrow as pa
import pytest
from lance import blob_array, blob_field

from common.lancekit.descriptor import (
    Declared,
    load_dataset_descriptor,
    validate_descriptor,
)
from common.lancekit.introspect import discover_tables

DIM = 8


@pytest.fixture
def tiny_db(tmp_path):
    chunks = pa.table(
        {
            "doc_id": pa.array(["a" * 16]),
            "chunk_id": pa.array([0], pa.int32()),
            "text": pa.array(["hej"]),
            "text_embedding": pa.array([[0.0] * DIM], pa.list_(pa.float32(), DIM)),
        }
    )
    lance.write_dataset(chunks, str(tmp_path / "chunks.lance"))
    docs_schema = pa.schema(
        [pa.field("doc_id", pa.string()), pa.field("plain_bytes", pa.binary()), blob_field("media_blob")]
    )
    docs = pa.table(
        {
            "doc_id": pa.array(["a" * 16]),
            "plain_bytes": pa.array([b"x"], pa.binary()),
            "media_blob": blob_array([b"content"]),
        },
        schema=docs_schema,
    )
    lance.write_dataset(docs, str(tmp_path / "documents.lance"), data_storage_version="2.2")
    return tmp_path


def _declared(**overrides) -> Declared:
    base = {
        "identity": {"key_fields": ["doc_id", "chunk_id"]},
        "document": {"table": "documents", "media_blob": "media_blob"},
        "search": {
            "row_table": "chunks",
            "fts": {"table": "chunks", "column": "text"},
            "vectors": {
                "semantic": {"table": "chunks", "column": "text_embedding", "dim": DIM, "query_encoder": "text"}
            },
        },
    }
    base.update(overrides)
    return Declared.model_validate(base)


def test_valid_descriptor_has_no_problems(tiny_db) -> None:
    assert validate_descriptor(_declared(), discover_tables(tiny_db)) == []


def test_missing_identity_field_is_caught(tiny_db) -> None:
    declared = _declared(identity={"key_fields": ["doc_id", "nope"]})
    problems = validate_descriptor(declared, discover_tables(tiny_db))
    assert any("identity field 'nope'" in p for p in problems)


def test_non_blob_media_column_is_caught(tiny_db) -> None:
    declared = _declared(document={"table": "documents", "media_blob": "plain_bytes"})
    problems = validate_descriptor(declared, discover_tables(tiny_db))
    assert any("not a lance.blob.v2 column" in p for p in problems)


def test_wrong_vector_dim_is_caught(tiny_db) -> None:
    declared = _declared(
        search={
            "row_table": "chunks",
            "vectors": {
                "semantic": {"table": "chunks", "column": "text_embedding", "dim": DIM * 2, "query_encoder": "text"}
            },
        }
    )
    problems = validate_descriptor(declared, discover_tables(tiny_db))
    assert any(f"dim {DIM} != declared {DIM * 2}" in p for p in problems)


TRANSCRIPTS = Path("transcripts_v2.lance")


@pytest.mark.integration
@pytest.mark.skipif(not TRANSCRIPTS.exists(), reason="local transcripts_v2.lance dataset not present")
def test_shipped_transcripts_descriptor_validates_against_live_corpus() -> None:
    descriptor = load_dataset_descriptor(TRANSCRIPTS, "transcripts_v2", "config/descriptors")
    assert descriptor.declared.search is not None
    semantic = descriptor.declared.search.vectors["semantic"]
    live = descriptor.tables["chunks"].column("text_embedding")
    assert live is not None and live.vector_dim == semantic.dim == 2048
    media = descriptor.tables["documents"].column("media_blob")
    assert media is not None and media.is_blob
    for key in descriptor.declared.identity.key_fields:
        assert descriptor.tables["chunks"].column(key) is not None
    assert descriptor.capability_available("diarization")
    assert descriptor.capability_available("topics")
