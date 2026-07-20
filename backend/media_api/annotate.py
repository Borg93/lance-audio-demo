"""Annotation read/write — serves a Lance ``annotations`` table as Arrow IPC.

The annotator's data plane (RA_ANNO_MERGE.md §5d): the PixiJS engine consumes
Arrow IPC directly (zero-copy geometry views), so we stream the Lance annotations
table as an Arrow IPC stream — the same wire format the atlas ``/points`` uses.
GET reads the rows for one media unit; a missing table degrades to an empty
stream (a dataset with no annotations yet is not an error).
"""

import logging
from collections.abc import Mapping, Sequence

import pyarrow as pa
from fastapi import APIRouter, Response
from pydantic import BaseModel, Field

from backend.core.exceptions import ConflictError, NotFoundError
from backend.deps import StateDep
from backend.lancekit.descriptor import Declared
from backend.lancekit.lineage_emit import emit_save
from backend.lancekit.reader import open_reader
from backend.lancekit.writer import open_writer
from backend.media_api.media import DatasetParam, chunk_key_filter, table_dataset, validate_doc_key
from backend.state import dataset_handle

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["annotate"])

_ARROW_STREAM = "application/vnd.apache.arrow.stream"
ANNOTATIONS_TABLE = "annotations"

#: The fields a reviewer may edit — the local-first review overlay flushed on save.
#: Only these are patched; geometry + provenance columns are carried forward from the
#: current row, so a partial edit never wipes a shape.
_EDITABLE_FIELDS = ("label", "status", "text", "group", "reviewer")


class AnnotationEdit(BaseModel):
    """One reviewed annotation: its id + only the fields that changed."""

    id: str
    label: str | None = None
    status: str | None = None
    text: str | None = None
    group: str | None = None
    reviewer: str | None = None


class NewAnnotation(BaseModel):
    """A newly drawn shape. Geometry + attributes; the chunk identity columns are
    stamped server-side from the route keys, so the client sends only shape data.
    A human-drawn shape is ``accepted`` by construction, ``source=human``."""

    id: str
    shape_type: str
    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0
    rotation: float = 0.0
    polygon: list[float] = Field(default_factory=list)
    text: str = ""
    label: str = ""
    status: str = "accepted"
    source: str = "human"
    group: str = ""
    mask: str = ""


class SaveAnnotations(BaseModel):
    """The delta a Save flushes for one media unit: field edits + newly drawn shapes
    + deleted ids. All three commit together (edits+inserts in one merge_insert).

    ``base_version`` is the Lance version the client loaded — optimistic concurrency:
    the save 409s if the table advanced underneath it (someone else / a deriver wrote).
    """

    edits: list[AnnotationEdit] = Field(default_factory=list)
    inserts: list[NewAnnotation] = Field(default_factory=list)
    deletes: list[str] = Field(default_factory=list)
    base_version: int | None = None


class SaveResult(BaseModel):
    """One save. ``saved`` counts touched rows (edits+inserts+deletes)."""

    saved: int
    version: int

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


def _build_delta(current: pa.Table, edits_by_id: dict[str, dict[str, object]]) -> pa.Table:
    """The merge_insert source: current rows for the edited ids, editable fields
    patched, everything else (geometry, provenance) carried forward. Same schema as
    ``current`` so merge_insert updates in place."""
    patched = [
        {**row, **edits_by_id[row["id"]]}
        for row in current.to_pylist()
        if row["id"] in edits_by_id
    ]
    return pa.Table.from_pylist(patched, schema=current.schema)


def identity_values(declared: Declared, doc_id: str, rest: Sequence[int]) -> dict[str, object]:
    """The chunk identity columns as a dict — the same (doc key, *other key fields)
    mapping ``chunk_key_filter`` builds as a predicate, stamped onto new rows so a
    drawn shape carries its unit's identity. Arity-generic off the descriptor."""
    identity = declared.identity
    values: dict[str, object] = {identity.doc_key: doc_id}
    others = [f for f in identity.key_fields if f != identity.doc_key]
    for field, value in zip(others, rest, strict=False):
        values[field] = int(value)
    return values


def _new_rows(inserts: Sequence[NewAnnotation], ident: Mapping[str, object], schema: pa.Schema) -> pa.Table:
    """Full new-annotation rows: identity stamped + shape fields; columns absent from
    the payload fall to null via the schema. Same schema ⇒ merge_insert can insert."""
    rows = [{**ident, **ins.model_dump()} for ins in inserts]
    return pa.Table.from_pylist(rows, schema=schema)


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
            headers={"Cache-Control": "no-store", "X-Annotations-Version": "0"},
        )
    where = chunk_key_filter(declared, doc_id, (speech_id, chunk_id))
    # Reads flow through the reader seam (direct default = byte-identical; catalog /query
    # at merge) — open_reader was built for exactly this.
    reader = open_reader(
        dataset=ds, table_id=[handle.id, ANNOTATIONS_TABLE], settings=state.settings
    )
    table = reader.to_table(filter=where)
    # The loaded Lance version — the client echoes it on Save for optimistic concurrency.
    return Response(
        content=_ipc_stream(table),
        media_type=_ARROW_STREAM,
        headers={"Cache-Control": "no-store", "X-Annotations-Version": str(int(ds.version))},
    )


@router.post("/annotations/{doc_id}/{speech_id}/{chunk_id}")
def save_annotations(
    state: StateDep,
    doc_id: str,
    speech_id: int,
    chunk_id: int,
    body: SaveAnnotations,
    dataset: DatasetParam = None,
) -> SaveResult:
    """Flush a review delta to Lance — the local-first Save (NOT sync-per-edit).

    Edits accumulate client-side (in-memory overlay + undo/redo); one Save patches
    only the editable fields onto the current rows (geometry/provenance carried
    forward) and ``merge_insert("id")`` commits them as ONE atomic new version —
    the lakehouse "version IS the handshake". (The catalog-governed write path is
    the merge step; this is the direct-write prototype, mirroring the direct read.)
    """
    handle = dataset_handle(state, dataset)
    declared = handle.descriptor.declared
    doc_id = validate_doc_key(declared, doc_id)
    ds = table_dataset(handle, ANNOTATIONS_TABLE)  # raises NotFoundError if absent

    # Optimistic concurrency: reject if the table advanced since the client loaded.
    if body.base_version is not None and body.base_version != int(ds.version):
        raise ConflictError(
            f"annotations changed on the server (loaded v{body.base_version}, now v{int(ds.version)})"
        )

    where = chunk_key_filter(declared, doc_id, (speech_id, chunk_id))
    current = ds.to_table(filter=where)

    # edits (patch existing) + inserts (new shapes) commit in ONE merge_insert; the
    # source is the union, keyed by id — matched ⇒ update, unmatched ⇒ insert.
    edits_by_id: dict[str, dict[str, object]] = {
        e.id: e.model_dump(include=set(_EDITABLE_FIELDS), exclude_none=True) for e in body.edits
    }
    parts = [_build_delta(current, edits_by_id)]
    if body.inserts:
        ident = identity_values(declared, doc_id, (speech_id, chunk_id))
        parts.append(_new_rows(body.inserts, ident, current.schema))
    delta = pa.concat_tables(parts)

    # Writes flow through the writer seam (direct default = byte-identical; catalog
    # merge_insert/delete at merge, which yields OpenFGA + OpenLineage for free).
    writer = open_writer(
        dataset=ds, table_id=[handle.id, ANNOTATIONS_TABLE], settings=state.settings
    )
    touched = 0
    if delta.num_rows:
        writer.merge_upsert(delta, "id")
        touched += delta.num_rows
    if body.deletes:
        quoted = ", ".join(_sql_quote(d) for d in body.deletes)
        writer.delete(f"id IN ({quoted})")
        touched += len(body.deletes)

    if touched == 0:
        return SaveResult(saved=0, version=int(ds.version))

    fresh = table_dataset(handle, ANNOTATIONS_TABLE)
    new_version = int(fresh.version)
    # Pre-merge OpenLineage: emit a spec-2-0-2 RunEvent for the write (at merge the
    # catalog mover emits it instead). Sink from settings (log|stdout|none).
    emit_save(
        ds=fresh,
        table_uri=handle.table_uri(ANNOTATIONS_TABLE),
        table_name=ANNOTATIONS_TABLE,
        unit_key=f"{doc_id}/{speech_id}/{chunk_id}",
        sink=state.settings.lineage_sink,
    )
    logger.info(
        "saved %s→ v%d (%d edit+insert, %d delete)",
        doc_id,
        new_version,
        delta.num_rows,
        len(body.deletes),
    )
    return SaveResult(saved=touched, version=new_version)


def _sql_quote(value: str) -> str:
    """SQL single-quoted string literal (doubling quotes) — the injection guard for
    the delete predicate's id list."""
    return "'" + value.replace("'", "''") + "'"
