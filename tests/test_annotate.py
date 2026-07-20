"""Annotation Arrow-IPC serving — the annotator's read wire.

The endpoint itself is proven end-to-end (live curl → 200 Arrow stream; playwright
renders the rows on the PixiJS canvas); these pin the serialization contract the
frontend `tableFromIPC` depends on.
"""

from __future__ import annotations

import pyarrow as pa

from backend.media_api.annotate import _EMPTY_SCHEMA, _ipc_stream


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
