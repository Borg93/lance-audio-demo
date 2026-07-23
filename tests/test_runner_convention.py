"""The stage↔runner convention: a new model is 1 runner dir + 1 Stage entry.

The proof drives the REAL append path (``run_append_stage`` → convention
resolution → ``run_append_rows_stage`` → local Ray ``map_batches``) with a fake
runner injected under ``runners.fake_ocr.actor`` — zero edits to the driver or
the composition root. Also pins the two refusal shapes: topics (corpus-global —
its actor module raises its own explanation) and kg (job-only, no actor module).
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType

import lance
import pyarrow as pa
import pytest

from ratch.core.registry import ActorConfig, Stage, StageShape
from ratch.core.runners import resolve_runner_actor
from ratch.errors import RatchError
from ratch.features.ray_av import run_append_stage

FAKE_OCR_SCHEMA = pa.schema(
    [("doc_id", pa.string()), ("chunk_id", pa.int32()), ("ocr", pa.string())]
)

FAKE_OCR_STAGE = Stage(
    name="fake_ocr",
    runner="fake_ocr",
    shape=StageShape.APPEND_ROWS,
    table="chunks",
    read_columns=("doc_id", "chunk_id", "text"),
    key_columns=("doc_id", "chunk_id"),
    output_table="ocr_lines",
    actor=ActorConfig(min_actors=1, max_actors=1, batch_rows=2),
)


def _install_fake_runner(monkeypatch) -> None:
    """A fake runner actor module, by convention name.

    ``compute_factory`` is a nested function so cloudpickle ships it BY VALUE to
    the Ray workers (they cannot import this test module or the injected module).
    """

    def compute_factory(_ctx):
        def compute(batch: pa.Table) -> pa.Table:
            return pa.table(
                {
                    "doc_id": batch["doc_id"],
                    "chunk_id": batch["chunk_id"],
                    "ocr": pa.array(
                        [(t or "").upper() for t in batch["text"].to_pylist()], pa.string()
                    ),
                },
                schema=FAKE_OCR_SCHEMA,
            )

        return compute

    module = ModuleType("runners.fake_ocr.actor")
    module.compute_factory = compute_factory  # ty: ignore[unresolved-attribute]
    module.OUTPUT_SCHEMA = FAKE_OCR_SCHEMA  # ty: ignore[unresolved-attribute]
    monkeypatch.setitem(sys.modules, "runners.fake_ocr.actor", module)


@pytest.fixture(scope="module")
def ray_cluster():
    import ray

    ray.init(num_cpus=2, include_dashboard=False, ignore_reinit_error=True)
    yield
    ray.shutdown()


@pytest.fixture
def chunks_db(tmp_path):
    table = pa.table(
        {
            "doc_id": pa.array(["d1", "d1", "d2", "d2", "d2"]),
            "chunk_id": pa.array([0, 1, 0, 1, 2], pa.int32()),
            "text": pa.array(["a", "b", "c", "d", "e"]),
        }
    )
    lance.write_dataset(table, str(tmp_path / "chunks.lance"))
    return tmp_path


class TestConventionResolution:
    def test_real_runners_conform(self) -> None:
        # Importing the actor modules is light (schemas only) — the models load
        # lazily inside compute_factory, on the workers.
        from ratch.model.schema import SPEAKER_EMBEDDINGS_SCHEMA, SPEAKER_TURNS_SCHEMA

        assert resolve_runner_actor("diarize").OUTPUT_SCHEMA is SPEAKER_TURNS_SCHEMA
        assert resolve_runner_actor("voiceprint").OUTPUT_SCHEMA is SPEAKER_EMBEDDINGS_SCHEMA

    def test_topics_refuses_with_its_own_explanation(self) -> None:
        with pytest.raises(RatchError, match="corpus-global"):
            resolve_runner_actor("topics")
        with pytest.raises(RatchError, match="corpus-global"):
            importlib.import_module("runners.topics.actor")

    def test_job_only_runner_points_at_the_jobs_seam(self) -> None:
        with pytest.raises(RatchError, match="job-only"):
            resolve_runner_actor("kg")

    def test_missing_exports_fail_the_contract(self, monkeypatch) -> None:
        module = ModuleType("runners.half_baked.actor")
        module.compute_factory = lambda _ctx: None  # ty: ignore[unresolved-attribute]
        monkeypatch.setitem(sys.modules, "runners.half_baked.actor", module)
        with pytest.raises(RatchError, match="OUTPUT_SCHEMA"):
            resolve_runner_actor("half_baked")


class TestNewModelIsOneRunnerDirAndOneStageEntry:
    @pytest.mark.usefixtures("ray_cluster")
    def test_fake_runner_drives_the_append_path(self, chunks_db, monkeypatch) -> None:
        _install_fake_runner(monkeypatch)

        appended = run_append_stage(chunks_db, FAKE_OCR_STAGE)
        assert appended == 5

        out = lance.dataset(str(chunks_db / "ocr_lines.lance")).to_table().to_pydict()
        assert sorted(out["ocr"]) == ["A", "B", "C", "D", "E"]

        # Resume is the key diff: a second run appends nothing.
        assert run_append_stage(chunks_db, FAKE_OCR_STAGE) == 0

    def test_unbound_pure_stage_is_actionable(self, chunks_db) -> None:
        stage = FAKE_OCR_STAGE.model_copy(update={"name": "mystery", "runner": None})
        with pytest.raises(RatchError, match=r"Stage\.runner"):
            run_append_stage(chunks_db, stage)
