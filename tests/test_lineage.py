"""OpenLineage draft — the pure/local pieces a stage contributes to lance-ns's graph."""

from __future__ import annotations

from pathlib import Path

import lance
import pyarrow as pa
from lance import blob_array, blob_field

from ratch.core.registry import ActorConfig, Stage, StageShape
from ratch.lineage import (
    SCHEMA_URL,
    build_run_event,
    column_map,
    emit_stage_lineage,
    facet_fields,
    measure_stage,
    run_id_for,
)


def test_primitives_are_kernel_owned_and_reexported() -> None:
    """The facet primitives live in common.lancekit.openlineage; ratch.lineage
    re-exports the SAME objects (identity), so the import direction is common←ratch
    and both call sites share one contract."""
    from common.lancekit import openlineage as kernel
    from ratch import lineage as pipeline

    assert pipeline.WriteResult is kernel.WriteResult
    assert pipeline.build_run_event is kernel.build_run_event
    assert pipeline.facet_fields is kernel.facet_fields
    assert pipeline.SCHEMA_URL == kernel.SCHEMA_URL


def _scan_stage() -> Stage:
    return Stage(
        name="text_embedding",
        shape=StageShape.SCAN_COLUMN,
        table="chunks",
        read_columns=("doc_id", "speech_id", "chunk_id", "text"),
        key_columns=("doc_id", "speech_id", "chunk_id"),
        output_columns=("text_embedding",),
        client="embed",
        actor=ActorConfig(max_actors=1, batch_rows=8),
    )


def _append_stage() -> Stage:
    return Stage(
        name="extract_frames",
        shape=StageShape.APPEND_ROWS,
        table="chunks",
        read_columns=("doc_id", "speech_id", "chunk_id"),
        key_columns=("doc_id", "speech_id", "chunk_id", "frame_idx"),
        output_table="chunk_frames",
        client="frames",
        actor=ActorConfig(max_actors=1, batch_rows=8),
    )


class TestColumnMap:
    def test_scan_column_carries_reads_and_derives_output(self) -> None:
        edges = column_map(_scan_stage())
        # every read column is carried IDENTITY
        assert ("text", "text", "IDENTITY") in edges
        assert ("doc_id", "doc_id", "IDENTITY") in edges
        # the output column is TRANSFORMATION off the first read column
        assert ("text_embedding", "doc_id", "TRANSFORMATION") in edges

    def test_append_rows_mints_key_as_transformation(self) -> None:
        edges = column_map(_append_stage())
        # frame_idx is a key NOT present upstream → minted → TRANSFORMATION
        assert ("frame_idx", "doc_id", "TRANSFORMATION") in edges
        # parent keys are carried IDENTITY
        assert ("doc_id", "doc_id", "IDENTITY") in edges

    def test_blob_column_input_is_the_blob(self) -> None:
        stage = Stage(
            name="caption",
            shape=StageShape.BLOB_COLUMN,
            table="chunk_frames",
            blob_column="frame_blob",
            key_columns=("doc_id",),
            output_columns=("caption",),
            client="caption",
            actor=ActorConfig(max_actors=1, batch_rows=8),
        )
        assert ("caption", "frame_blob", "TRANSFORMATION") in column_map(stage)


class TestFacetFields:
    def test_blob_and_vector_labels(self, tmp_path: Path) -> None:
        uri = str(tmp_path / "t.lance")
        table = pa.table(
            {
                "id": pa.array([1]),
                "vec": pa.array([[0.0, 1.0]], type=pa.list_(pa.float32(), 2)),
                "media": blob_array([b"x"]),
            },
            schema=pa.schema([pa.field("id", pa.int64()), pa.field("vec", pa.list_(pa.float32(), 2)), blob_field("media")]),
        )
        lance.write_dataset(table, uri, data_storage_version="2.2")
        fields = {f["name"]: f["type"] for f in facet_fields(lance.dataset(uri).schema)}
        assert fields["media"] == "blob"
        assert fields["vec"].startswith("array<")
        assert fields["id"] == "int64"


class TestMeasureAndEvent:
    def _written(self, tmp_path: Path) -> str:
        uri = str(tmp_path / "chunks.lance")
        lance.write_dataset(
            pa.table({"doc_id": ["a", "b", "c"], "text": ["x", "y", "z"]}),
            uri,
            data_storage_version="2.2",
        )
        return uri

    def test_measure_reports_version_rows_fields(self, tmp_path: Path) -> None:
        r = measure_stage(self._written(tmp_path))
        assert r.row_count == 3
        assert r.version >= 1
        assert r.size_bytes > 0
        assert {f["name"] for f in r.fields} == {"doc_id", "text"}

    def test_emit_builds_spec_2_0_2_event_with_facets(self, tmp_path: Path) -> None:
        events: list[dict] = []

        class ListSink:
            def emit(self, event: dict) -> None:
                events.append(event)

        event = emit_stage_lineage(
            stage=_scan_stage(),
            output_uri=self._written(tmp_path),
            output_namespace="silver",
            output_name="chunks",
            input_datasets=[("bronze", "chunks")],
            event_time="2026-07-20T00:00:00Z",
            sink=ListSink(),
        )
        assert events == [event]
        assert event["schemaURL"] == SCHEMA_URL
        assert event["eventType"] == "COMPLETE"
        out = event["outputs"][0]
        assert out["namespace"] == "silver" and out["name"] == "chunks"
        assert out["facets"]["outputStatistics"]["rowCount"] == 3
        # column lineage present, keyed by output field
        assert "text_embedding" in out["facets"]["columnLineage"]["fields"]

    def test_run_id_is_deterministic(self) -> None:
        assert run_id_for("extract_frames-chunk_frames-2") == run_id_for("extract_frames-chunk_frames-2")
        assert run_id_for("a") != run_id_for("b")

    def test_builder_override_is_used(self, tmp_path: Path) -> None:
        # the merge seam: pass a foreign builder (e.g. lance-ns's build_run_event)
        sentinel = {"from": "lance-ns"}

        def fake_builder(**_kw) -> dict:
            return sentinel

        class NullSink:
            def emit(self, event: dict) -> None: ...

        got = emit_stage_lineage(
            stage=_scan_stage(),
            output_uri=self._written(tmp_path),
            output_namespace="silver",
            output_name="chunks",
            input_datasets=[("bronze", "chunks")],
            event_time="2026-07-20T00:00:00Z",
            sink=NullSink(),
            builder=fake_builder,
        )
        assert got is sentinel


def test_standalone_and_declared_shapes_are_consistent() -> None:
    # build_run_event consumes a WriteResult with a column_map exactly as measure+column_map produce
    from ratch.lineage import WriteResult

    r = WriteResult(version=1, row_count=2, size_bytes=10, fields=[{"name": "a", "type": "int64"}])
    r.column_map = column_map(_scan_stage())
    event = build_run_event(
        operation="TRANSFORM",
        job_namespace="ratch",
        job_name="ratch.text_embedding",
        inputs=[("bronze", "chunks")],
        output_namespace="silver",
        output_name="chunks",
        event_time="2026-07-20T00:00:00Z",
        result=r,
    )
    assert event["outputs"][0]["facets"]["columnLineage"]["fields"]["text_embedding"]["inputFields"]
