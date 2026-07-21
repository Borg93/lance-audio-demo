"""Arrow-IPC wire serving — the annotator's READ path.

The PixiJS engine consumes Arrow IPC directly (zero-copy geometry views), so we stream
the Lance annotations table as an Arrow IPC stream — the same wire format the atlas
``/points`` uses. A missing table degrades to an empty stream (a dataset with no
annotations yet is not an error).
"""

from typing import Annotated

import pyarrow as pa
from fastapi import APIRouter, Query, Response

from backend.core.exceptions import NotFoundError
from backend.deps import StateDep
from backend.lancekit.reader import open_reader
from backend.media_api.annotations.schema import ANNOTATIONS_TABLE, EMPTY_SCHEMA
from backend.media_api.annotations.versions import checkout
from backend.media_api.media import DatasetParam, chunk_key_filter, table_dataset, validate_doc_key
from backend.state import dataset_handle

router = APIRouter(tags=["annotate"])

ARROW_STREAM = "application/vnd.apache.arrow.stream"


def ipc_stream(table: pa.Table) -> bytes:
    """Serialize an Arrow table to an IPC stream (the wire the engine parses)."""
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


@router.get("/annotations/{doc_id}/{speech_id}/{chunk_id}")
def annotations(
    state: StateDep,
    doc_id: str,
    speech_id: int,
    chunk_id: int,
    dataset: DatasetParam = None,
    version: Annotated[
        int | None, Query(ge=1, description="Read a historical version (time-travel).")
    ] = None,
) -> Response:
    """Arrow IPC stream of the annotations for one media unit (doc + identity keys).

    The keys map positionally onto the descriptor's identity fields (same shape as
    the chunk / chunk-frame routes). ``version`` reads a HISTORICAL snapshot (Lance
    time-travel) for the compare-versions view; omitted ⇒ the latest."""
    handle = dataset_handle(state, dataset)
    declared = handle.descriptor.declared
    doc_id = validate_doc_key(declared, doc_id)
    try:
        ds = table_dataset(handle, ANNOTATIONS_TABLE)
    except NotFoundError:
        # No annotations table yet — serve an empty stream so the client renders 0.
        return Response(
            content=ipc_stream(EMPTY_SCHEMA.empty_table()),
            media_type=ARROW_STREAM,
            headers={"Cache-Control": "no-store", "X-Annotations-Version": "0"},
        )
    where = chunk_key_filter(declared, doc_id, (speech_id, chunk_id))
    if version is not None:
        # A historical read is a direct time-travel snapshot (read-only, off the hot
        # path); the reader seam governs the current read.
        table = checkout(ds, version).to_table(filter=where)
        served_version = version
    else:
        # Reads flow through the reader seam (direct default = byte-identical; catalog
        # /query at merge) — open_reader was built for exactly this.
        reader = open_reader(
            dataset=ds, table_id=[handle.id, ANNOTATIONS_TABLE], settings=state.settings
        )
        table = reader.to_table(filter=where)
        served_version = int(ds.version)
    # The loaded Lance version — the client echoes it on Save for optimistic concurrency.
    return Response(
        content=ipc_stream(table),
        media_type=ARROW_STREAM,
        headers={"Cache-Control": "no-store", "X-Annotations-Version": str(served_version)},
    )
