"""Annotation read/write — serves a Lance ``annotations`` table as Arrow IPC.

The annotator's data plane (RA_ANNO_MERGE.md §5d): the PixiJS engine consumes
Arrow IPC directly (zero-copy geometry views), so we stream the Lance annotations
table as an Arrow IPC stream — the same wire format the atlas ``/points`` uses.
GET reads the rows for one media unit; a missing table degrades to an empty
stream (a dataset with no annotations yet is not an error).
"""

import logging
from collections.abc import Mapping, Sequence
from typing import Annotated
from urllib.parse import quote

import pyarrow as pa
from fastapi import APIRouter, Query, Response
from pydantic import BaseModel, Field

from backend.core.exceptions import ConflictError, NotFoundError
from backend.deps import AuthorDep, StateDep
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
    t_start: float = 0.0
    t_end: float = 0.0
    text: str = ""
    label: str = ""
    status: str = "accepted"
    source: str = "human"
    group: str = ""
    mask: str = ""


class GeometryEdit(BaseModel):
    """A moved/resized EXISTING shape — spatial geometry only, keyed by id (canvas drag)."""

    id: str
    x: float
    y: float
    width: float
    height: float
    polygon: list[float] = Field(default_factory=list)


class TemporalEdit(BaseModel):
    """A resized audio/video SEGMENT — its times only, keyed by id (waveform region drag)."""

    id: str
    t_start: float
    t_end: float


class SaveAnnotations(BaseModel):
    """The delta a Save flushes for one media unit: field edits + newly drawn shapes
    + moved geometry + deleted ids. Edits/inserts/geometry commit together (one
    merge_insert), deletes follow.

    ``base_version`` is the Lance version the client loaded — optimistic concurrency:
    the save 409s if the table advanced underneath it (someone else / a deriver wrote).
    """

    edits: list[AnnotationEdit] = Field(default_factory=list)
    inserts: list[NewAnnotation] = Field(default_factory=list)
    geometry: list[GeometryEdit] = Field(default_factory=list)
    temporal: list[TemporalEdit] = Field(default_factory=list)
    deletes: list[str] = Field(default_factory=list)
    base_version: int | None = None


class SaveResult(BaseModel):
    """One save. ``saved`` counts touched rows (edits+inserts+deletes)."""

    saved: int
    version: int


class TagWrite(BaseModel):
    """One chunk (or its unit) + the tag labels to set on it. ``keys`` are the NON-doc
    identity fields, positional — they pair with ``descriptor.identity.key_fields`` minus
    the doc key, exactly like ``identity_values`` / ``chunk_key_filter`` (a chunk tag
    sends ``keys=[speech_id, chunk_id]``)."""

    doc_id: str
    keys: list[int] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)


class TagBatch(BaseModel):
    """A workflow's chunk-tags promoted to annotation ROWS across many units, committed
    in ONE version. ``removes`` untags (deletes the deterministic tag rows). Author is
    stamped server-side; ``source``/``status`` are fixed for a human set."""

    adds: list[TagWrite] = Field(default_factory=list)
    removes: list[TagWrite] = Field(default_factory=list)
    base_version: int | None = None


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
        # temporal facet — audio segments + a shape pinned to a video moment (seconds)
        ("t_start", pa.float32()),
        ("t_end", pa.float32()),
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
        {**row, **edits_by_id[row["id"]]} for row in current.to_pylist() if row["id"] in edits_by_id
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


def _new_rows(
    inserts: Sequence[NewAnnotation], ident: Mapping[str, object], schema: pa.Schema
) -> pa.Table:
    """Full new-annotation rows: identity stamped + shape fields; columns absent from
    the payload fall to null via the schema. Same schema ⇒ merge_insert can insert."""
    rows = [{**ident, **ins.model_dump()} for ins in inserts]
    return pa.Table.from_pylist(rows, schema=schema)


def tag_id(doc_id: str, keys: Sequence[int], label: str) -> str:
    """Deterministic id for a chunk-level TAG row — ``tag:{doc}:{keys}:{enc(label)}``.
    Idempotent (re-tagging the same chunk maps to the same id, so no duplicates); the
    ``tag:`` prefix + percent-encoded label guarantee it can never collide with a drawn
    shape's random id (so a tag merge_insert never clobbers a shape's geometry)."""
    parts = [doc_id, *(str(k) for k in keys), quote(label, safe="")]
    return "tag:" + ":".join(parts)


def _tag_rows(
    adds: Sequence[TagWrite], declared: Declared, author: str, schema: pa.Schema
) -> pa.Table:
    """Full annotation rows for chunk tags — the tag↔annotation unification: per-row
    identity stamped, ``shape_type='tag'``, label carried, human provenance + server
    author, geometry/temporal zeroed; unlisted columns fall to null via the schema.
    Same schema ⇒ merge_insert inserts by the deterministic id. Rows are DEDUPED by id
    (a batch may repeat a chunk+label): Lance merge_insert does not dedupe duplicate
    source keys, so two identical-id rows would both insert — dedup here prevents that."""
    by_id: dict[str, dict[str, object]] = {}
    for w in adds:
        doc = validate_doc_key(declared, w.doc_id)
        ident = identity_values(declared, doc, w.keys)
        for label in w.labels:
            tid = tag_id(doc, w.keys, label)
            by_id.setdefault(  # first wins — every row for a given id is identical anyway
                tid,
                {
                    **ident,
                    "id": tid,
                    "shape_type": "tag",
                    "x": 0.0,
                    "y": 0.0,
                    "width": 0.0,
                    "height": 0.0,
                    "rotation": 0.0,
                    "polygon": [],
                    "t_start": 0.0,
                    "t_end": 0.0,
                    "label": label,
                    "text": "",
                    "group": "",
                    "status": "accepted",
                    "source": "human",
                    "reviewer": author,
                    "mask": "",
                },
            )
    return pa.Table.from_pylist(list(by_id.values()), schema=schema)


@router.get("/annotations/{doc_id}/{speech_id}/{chunk_id}")
def annotations(
    state: StateDep,
    doc_id: str,
    speech_id: int,
    chunk_id: int,
    dataset: DatasetParam = None,
    version: Annotated[int | None, Query(ge=1, description="Read a historical version (time-travel).")] = None,
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
            content=_ipc_stream(_EMPTY_SCHEMA.empty_table()),
            media_type=_ARROW_STREAM,
            headers={"Cache-Control": "no-store", "X-Annotations-Version": "0"},
        )
    where = chunk_key_filter(declared, doc_id, (speech_id, chunk_id))
    if version is not None:
        # A historical read is a direct time-travel snapshot (read-only, off the hot
        # path); the reader seam governs the current read.
        snapshot = ds.checkout_version(version)
        table = snapshot.to_table(filter=where)
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
        content=_ipc_stream(table),
        media_type=_ARROW_STREAM,
        headers={"Cache-Control": "no-store", "X-Annotations-Version": str(served_version)},
    )


class AnnotationVersion(BaseModel):
    """One point in a unit's edit history — a Lance version + when it was committed +
    how many annotations this unit had at it (the audit/compare-versions trail)."""

    version: int
    timestamp: str
    count: int


@router.get("/annotations/{doc_id}/{speech_id}/{chunk_id}/versions")
def annotation_versions(
    state: StateDep,
    doc_id: str,
    speech_id: int,
    chunk_id: int,
    dataset: DatasetParam = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 20,
) -> list[AnnotationVersion]:
    """The unit's edit history (most-recent first, capped): each Lance version + its
    timestamp + the count of THIS unit's annotations at it. Powers the compare-versions
    panel — the read-side of the write-plane provenance story (who=reviewer, when=version,
    what=lineage)."""
    handle = dataset_handle(state, dataset)
    declared = handle.descriptor.declared
    doc_id = validate_doc_key(declared, doc_id)
    try:
        ds = table_dataset(handle, ANNOTATIONS_TABLE)
    except NotFoundError:
        return []
    where = chunk_key_filter(declared, doc_id, (speech_id, chunk_id))
    out: list[AnnotationVersion] = []
    for v in list(reversed(ds.versions()))[:limit]:  # most recent first, capped
        vnum = int(v["version"])
        count = ds.checkout_version(vnum).to_table(filter=where, columns=["id"]).num_rows
        ts = v.get("timestamp")
        out.append(AnnotationVersion(version=vnum, timestamp=_iso(ts), count=count))
    return out


def _iso(ts: object) -> str:
    """A Lance version timestamp → ISO string (datetime or already-string)."""
    isofmt = getattr(ts, "isoformat", None)
    return isofmt() if callable(isofmt) else str(ts or "")


@router.post("/annotations/{doc_id}/{speech_id}/{chunk_id}")
def save_annotations(
    state: StateDep,
    author: AuthorDep,
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
    # Geometry moves + segment-time resizes patch the same rows (by id) — merged in.
    for g in body.geometry:
        edits_by_id.setdefault(g.id, {}).update(
            {"x": g.x, "y": g.y, "width": g.width, "height": g.height, "polygon": g.polygon}
        )
    for tseg in body.temporal:
        edits_by_id.setdefault(tseg.id, {}).update({"t_start": tseg.t_start, "t_end": tseg.t_end})
    # Governance: the SERVER stamps who wrote each touched row (not the client's claim) —
    # the per-user seam. lance-ns's OpenFGA keys on this author at merge.
    for fields in edits_by_id.values():
        fields["reviewer"] = author
    parts = [_build_delta(current, edits_by_id)]
    if body.inserts:
        ident = {**identity_values(declared, doc_id, (speech_id, chunk_id)), "reviewer": author}
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


@router.post("/annotations/tags")
def add_tags(
    state: StateDep,
    author: AuthorDep,
    body: TagBatch,
    dataset: DatasetParam = None,
) -> SaveResult:
    """Promote workflow chunk-tags to annotation ROWS (the tag↔annotation unification) —
    a human ``set`` over a chunk-level selection, written across MANY units in ONE
    merge_insert version. Not a model deriver, so it does NOT funnel through the jobs
    seam; the annotations table stays mode-blind (a tag is source='human'/status=
    'accepted'). Idempotent by the deterministic tag id (a re-Save adds no duplicates)."""
    handle = dataset_handle(state, dataset)
    declared = handle.descriptor.declared
    ds = table_dataset(handle, ANNOTATIONS_TABLE)  # raises NotFoundError if unseeded

    # Optimistic concurrency degrades to table-GLOBAL for a cross-unit batch; optional +
    # last-write-wins per deterministic id is the pragmatic contract.
    if body.base_version is not None and body.base_version != int(ds.version):
        raise ConflictError(
            f"annotations changed on the server (loaded v{body.base_version}, now v{int(ds.version)})"
        )

    delta = _tag_rows(body.adds, declared, author, ds.schema)
    writer = open_writer(
        dataset=ds, table_id=[handle.id, ANNOTATIONS_TABLE], settings=state.settings
    )
    touched = 0
    if delta.num_rows:
        # INSERT-ONLY (not upsert): a tag that already exists is left untouched, so a
        # re-save never clobbers a reviewer's edits to that tag row (status/label/group).
        # Re-tagging stays idempotent; humans own the row once it exists.
        writer.merge_insert_only(delta, "id")
        touched += delta.num_rows

    remove_ids = [
        tag_id(validate_doc_key(declared, w.doc_id), w.keys, label)
        for w in body.removes
        for label in w.labels
    ]
    if remove_ids:
        quoted = ", ".join(_sql_quote(i) for i in remove_ids)
        writer.delete(f"id IN ({quoted})")
        touched += len(remove_ids)

    if touched == 0:
        return SaveResult(saved=0, version=int(ds.version))

    fresh = table_dataset(handle, ANNOTATIONS_TABLE)
    new_version = int(fresh.version)
    emit_save(
        ds=fresh,
        table_uri=handle.table_uri(ANNOTATIONS_TABLE),
        table_name=ANNOTATIONS_TABLE,
        unit_key=f"tags:{len(body.adds)}+{len(body.removes)}",
        sink=state.settings.lineage_sink,
    )
    logger.info(
        "tagged → v%d (%d add-rows, %d remove)", new_version, delta.num_rows, len(remove_ids)
    )
    return SaveResult(saved=touched, version=new_version)


def _sql_quote(value: str) -> str:
    """SQL single-quoted string literal (doubling quotes) — the injection guard for
    the delete predicate's id list."""
    return "'" + value.replace("'", "''") + "'"
