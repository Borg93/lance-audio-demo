"""Annotation read/write — serves a Lance ``annotations`` table as Arrow IPC.

The annotator's data plane (RA_ANNO_MERGE.md §5d): the PixiJS engine consumes
Arrow IPC directly (zero-copy geometry views), so we stream the Lance annotations
table as an Arrow IPC stream — the same wire format the atlas ``/points`` uses.
GET reads the rows for one media unit; a missing table degrades to an empty
stream (a dataset with no annotations yet is not an error).
"""

import logging

import pyarrow as pa
from fastapi import APIRouter, Response

from backend.core.exceptions import NotFoundError
from backend.deps import StateDep
from backend.media_api.media import DatasetParam, chunk_key_filter, table_dataset, validate_doc_key
from backend.state import dataset_handle

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["annotate"])

_ARROW_STREAM = "application/vnd.apache.arrow.stream"
ANNOTATIONS_TABLE = "annotations"

#: The annotation contract — the schema of an EMPTY stream when a dataset has no
#: annotations table yet (so the client still parses). Aligned to the engine
#: (frontend/src/lib/engine/schema.ts) PLUS the active-learning columns
#: (confidence/uncertainty/source/model_version) so predictions round-trip and the
#: review queue can rank by them. Kept in one place; scripts/seed_annotations.py
#: writes the same columns.
_EMPTY_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("shape_type", pa.string()),
        ("x", pa.float32()),
        ("y", pa.float32()),
        ("width", pa.float32()),
        ("height", pa.float32()),
        ("rotation", pa.float32()),
        ("polygon", pa.list_(pa.float32())),
        ("text", pa.string()),
        ("label", pa.string()),
        ("status", pa.string()),
        ("source", pa.string()),
        ("reviewer", pa.string()),
        ("confidence", pa.float32()),
        ("uncertainty", pa.float32()),
        ("model_version", pa.string()),
        ("group", pa.string()),
        ("group_id", pa.string()),
        ("reading_order", pa.int32()),
        ("difficult", pa.bool_()),
        ("links", pa.string()),
        ("mask", pa.string()),
        ("metadata", pa.string()),
    ]
)


def _ipc_stream(table: pa.Table) -> bytes:
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
) -> Response:
    """Arrow IPC stream of the annotations for one media unit (doc + identity keys).

    The keys map positionally onto the descriptor's identity fields (same shape as
    the chunk / chunk-frame routes)."""
    handle = dataset_handle(state, dataset)
    declared = handle.descriptor.declared
    doc_id = validate_doc_key(declared, doc_id)
    try:
        ds = table_dataset(handle, ANNOTATIONS_TABLE)
    except NotFoundError:
        # No annotations table yet — serve an empty stream so the client renders 0.
        return Response(
            content=_ipc_stream(_EMPTY_SCHEMA.empty_table()),
            media_type=_ARROW_STREAM,
            headers={"Cache-Control": "no-store"},
        )
    where = chunk_key_filter(declared, doc_id, (speech_id, chunk_id))
    table = ds.to_table(filter=where)
    return Response(
        content=_ipc_stream(table),
        media_type=_ARROW_STREAM,
        headers={"Cache-Control": "no-store"},
    )
