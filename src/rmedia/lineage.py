"""OpenLineage emission for rmedia stages — DRAFT (spec 2-0-2), opt-in and inert.

Status: a first cut, NOT wired into the driver yet. It builds the pieces a stage
run must contribute to lance-ns's lineage graph, mirroring their emit contract so
it drops in at merge:

    lance-ns mover (services/medallion): transform_stage → WriteResult
        (version, row_count, size_bytes, fields, column_map) → build_run_event → publish

Here we produce the SAME `WriteResult` from a stage we wrote, plus the
`column_map` (field→field edges) DECLARED from our `Stage` model — the input the
harness turns into the `columnLineage` facet.

Two modes, one seam (`emit_stage_lineage`):
  * MERGED  — pass lance-ns's own `build_run_event` (import
    `medallion.schemas.events.build_run_event`) as `builder=`; we supply the
    measured `WriteResult` + `column_map`, they own the wire event. This is the
    intended end-state — do NOT reimplement their emitter in production.
  * STANDALONE — `builder=None` uses `build_run_event` below, a minimal spec-2-0-2
    mirror, so a pre-merge Ray/CLI run can emit real events to a `LineageSink`
    (file/stdout/HTTP). Kept faithful to their pinned constants (`SCHEMA_URL`,
    `run_id_for`, the facet `_schemaURL`s) so a standalone event and a merged event
    describe the same run identically.

Nothing here imports a model client or Ray; it's pure over a written dataset path
+ the `Stage` declaration, so it's callable from the driver, a test, or a backfill.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, Protocol, cast

import lance
import pyarrow as pa
from pydantic import BaseModel, Field

from rmedia.core.blobs import blob_field_names
from rmedia.core.registry import Stage, StageShape

if TYPE_CHECKING:
    from collections.abc import Callable

# ── Spec constants — MUST match lance-ns services/common/openlineage.py at merge ──
SCHEMA_URL = "https://openlineage.io/spec/2-0-2/OpenLineage.json#/$defs/RunEvent"
_SCHEMA_FACET_URL = "https://openlineage.io/spec/facets/1-1-1/SchemaDatasetFacet.json"
_OUTPUT_STATS_FACET_URL = (
    "https://openlineage.io/spec/facets/1-0-2/OutputStatisticsOutputDatasetFacet.json"
)
_COLUMN_LINEAGE_FACET_URL = (
    "https://openlineage.io/spec/facets/1-2-0/ColumnLineageDatasetFacet.json"
)
_DATASOURCE_FACET_URL = "https://openlineage.io/spec/facets/1-0-1/DatasourceDatasetFacet.json"
PRODUCER = "https://github.com/Borg93/lance-audio/tree/main/src/rmedia"
# Same UUID5 namespace lance-ns uses, so a run id computed here == the one computed
# there for the same seed (redeliveries MERGE onto one :Run, never duplicate).
_RUN_ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "https://github.com/Borg93/lance-ns")

#: One field→field edge: (output_field, input_field, transformation_subtype).
#: Carried columns are "IDENTITY"; derived artifacts are "TRANSFORMATION".
ColumnEdge = tuple[str, str, str]


class WriteResult(BaseModel):
    """The measured outcome of one stage write — mirrors lance-ns's WriteResult.

    Field names/shape match theirs on purpose so `build_run_event` (ours or theirs)
    consumes it unchanged.
    """

    version: int
    row_count: int
    size_bytes: int
    fields: list[dict[str, str]] = Field(default_factory=list)
    column_map: list[ColumnEdge] = Field(default_factory=list)


def _type_label(dtype: pa.DataType) -> str:
    """Concise lineage type label — vector/binary specialised, else pyarrow repr
    (blobs are labelled in `facet_fields`, which knows the blob column names)."""
    if pa.types.is_fixed_size_list(dtype) or pa.types.is_list(dtype) or pa.types.is_large_list(dtype):
        return f"array<{dtype.value_type}>"
    if pa.types.is_binary(dtype) or pa.types.is_large_binary(dtype):
        return "binary"
    return str(dtype)


def facet_fields(schema: pa.Schema) -> list[dict[str, str]]:
    """`SchemaDatasetFacet.fields` — `[{"name","type"}]` per column, blob-aware.
    Mirrors lance-ns common/schema.facet_fields (blob → "blob")."""
    blobs = set(blob_field_names(schema))
    return [
        {"name": f.name, "type": "blob" if f.name in blobs else _type_label(f.type)}
        for f in schema
    ]


def column_map(stage: Stage) -> list[ColumnEdge]:
    """The DECLARED field→field edges of a stage, from its `Stage` model alone.

    This is the "declared" column lineage (the in-process path in lance-ns). The
    distributed path RECONSTRUCTS the same edges from the two on-disk schemas; the
    two must agree, so this stays a pure function of the declaration:

    * carried key/read columns → (col, col, "IDENTITY")
    * each output column       → (out, primary_input, "TRANSFORMATION")

    where primary_input is the blob column (BLOB/APPEND over a blob) or the first
    read column (SCAN). A minted key not present upstream (e.g. `frame_idx`) is
    TRANSFORMATION off the primary input.
    """
    primary_input = stage.blob_column or (stage.read_columns[0] if stage.read_columns else "")
    carried = set(stage.read_columns)
    edges: list[ColumnEdge] = []
    # Identity: every read column carried through (for APPEND_ROWS these are the
    # parent keys that identify the child rows).
    for col in stage.read_columns:
        edges.append((col, col, "IDENTITY"))
    # Transformation: each declared output column, plus any key column that is
    # minted by the stage (present in key_columns but not carried from upstream).
    derived = list(stage.output_columns)
    if stage.shape is StageShape.APPEND_ROWS:
        derived += [k for k in stage.key_columns if k not in carried]
    for out in derived:
        if primary_input:
            edges.append((out, primary_input, "TRANSFORMATION"))
    return edges


def measure_stage(uri: str, storage_options: dict[str, str] | None = None) -> WriteResult:
    """Read a just-written stage's version + exact output stats + schema fields.

    Mirrors lance-ns's `measure`: version, `count_rows`, summed on-disk bytes, and
    the blob/vector-aware schema-facet fields. `column_map` is left empty here —
    attach it from `column_map(stage)` (declared) or reconstruct from schemas.
    """
    ds = lance.dataset(uri, storage_options=storage_options)
    # lance annotates DataStatistics.fields as a single FieldStatistics but returns a
    # list at runtime (upstream stub bug) — cast to the real shape, as lance-ns does.
    field_stats = cast("list[Any]", ds.stats.data_stats().fields)
    size_bytes = sum(getattr(s, "bytes_on_disk", 0) for s in field_stats)
    return WriteResult(
        version=int(ds.version),
        row_count=ds.count_rows(),
        size_bytes=size_bytes,
        fields=facet_fields(ds.schema),
    )


def run_id_for(seed: str) -> str:
    """Deterministic spec-valid UUID runId for a seed (e.g. `"<stage>-<token>"`).
    Same uuid5 namespace as lance-ns so redeliveries merge onto one run."""
    return str(uuid.uuid5(_RUN_ID_NAMESPACE, seed))


def build_run_event(
    *,
    operation: str,
    job_namespace: str,
    job_name: str,
    inputs: list[tuple[str, str]],
    output_namespace: str,
    output_name: str,
    event_time: str,
    result: WriteResult,
    source_uri: str | None = None,
    event_type: str = "COMPLETE",
    error_message: str | None = None,
    seed: str | None = None,
) -> dict[str, Any]:
    """A minimal, spec-2-0-2 OpenLineage `RunEvent` (standalone mirror).

    Prefer lance-ns's `medallion.schemas.events.build_run_event` when merged — pass
    it to `emit_stage_lineage(builder=...)`. This exists so a pre-merge run can emit
    the same-shaped event; keep it faithful, don't extend it past their facets.
    """
    run_id = run_id_for(seed or f"{operation}-{output_name}")
    output_facets: dict[str, Any] = {
        "schema": {"_producer": PRODUCER, "_schemaURL": _SCHEMA_FACET_URL, "fields": result.fields},
        "outputStatistics": {
            "_producer": PRODUCER,
            "_schemaURL": _OUTPUT_STATS_FACET_URL,
            "rowCount": result.row_count,
            "size": result.size_bytes,
        },
    }
    if result.column_map:
        output_facets["columnLineage"] = _column_lineage_facet(result.column_map, inputs)
    if source_uri is not None:
        output_facets["dataSource"] = {
            "_producer": PRODUCER,
            "_schemaURL": _DATASOURCE_FACET_URL,
            "name": output_name,
            "uri": source_uri,
        }
    run_facets: dict[str, Any] = {}
    if error_message is not None:
        run_facets["errorMessage"] = {
            "_producer": PRODUCER,
            "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/ErrorMessageRunFacet.json",
            "message": error_message,
            "programmingLanguage": "PYTHON",
        }
    return {
        "eventType": event_type,
        "eventTime": event_time,
        "producer": PRODUCER,
        "schemaURL": SCHEMA_URL,
        "run": {"runId": run_id, "facets": run_facets},
        "job": {"namespace": job_namespace, "name": job_name},
        "inputs": [{"namespace": ns, "name": name} for ns, name in inputs],
        "outputs": [{"namespace": output_namespace, "name": output_name, "facets": output_facets}],
    }


def _column_lineage_facet(
    edges: list[ColumnEdge], inputs: list[tuple[str, str]]
) -> dict[str, Any]:
    """A `ColumnLineageDatasetFacet`: per output field, its input field(s) + subtype."""
    in_ns, in_name = inputs[0] if inputs else ("", "")
    by_output: dict[str, list[dict[str, str]]] = {}
    for out_field, in_field, subtype in edges:
        by_output.setdefault(out_field, []).append(
            {
                "namespace": in_ns,
                "name": in_name,
                "field": in_field,
                "transformationType": "DIRECT",
                "transformationSubtype": subtype,
            }
        )
    return {
        "_producer": PRODUCER,
        "_schemaURL": _COLUMN_LINEAGE_FACET_URL,
        "fields": {out: {"inputFields": ins} for out, ins in by_output.items()},
    }


class LineageSink(Protocol):
    """Where a built RunEvent goes. Standalone: a file/stdout/HTTP sink. Merged: the
    Dapr outbox publish is lance-ns's job — pass their `build_run_event` and publish
    downstream instead of using this."""

    def emit(self, event: dict[str, Any]) -> None: ...


def emit_stage_lineage(
    *,
    stage: Stage,
    output_uri: str,
    output_namespace: str,
    output_name: str,
    input_datasets: list[tuple[str, str]],
    event_time: str,
    sink: LineageSink,
    storage_options: dict[str, str] | None = None,
    builder: Callable[..., dict[str, Any]] | None = None,
    operation: str = "TRANSFORM",
    job_namespace: str = "rmedia",
) -> dict[str, Any]:
    """Measure a written stage, attach its declared column lineage, build the
    RunEvent, and hand it to `sink`. Returns the event (also for tests).

    `builder=None` uses the standalone mirror; pass lance-ns's `build_run_event`
    when merged. The one call the driver would add after a successful stage write.
    """
    result = measure_stage(output_uri, storage_options)
    result.column_map = column_map(stage)
    make = builder or build_run_event
    event = make(
        operation=operation,
        job_namespace=job_namespace,
        job_name=f"{job_namespace}.{stage.name}",
        inputs=input_datasets,
        output_namespace=output_namespace,
        output_name=output_name,
        event_time=event_time,
        result=result,
        source_uri=output_uri,
        seed=f"{stage.name}-{output_name}-{result.version}",
    )
    sink.emit(event)
    return event
