"""P1.2: the scan driver really wires read_lance → map_batches → merge_insert.

Runs a tiny scan stage end-to-end on a local Ray cluster with a fake compute
factory: fills the column, converges to zero pending on re-run (resume is the
read), and never touches rows outside the NULL filter.
"""

from __future__ import annotations

import lance
import pyarrow as pa
import pytest

from ratch.core.driver import run_scan_column_stage
from ratch.core.registry import ActorConfig, Stage, StageShape

STAGE = Stage(
    name="shout",
    shape=StageShape.SCAN_COLUMN,
    table="chunks",
    read_columns=("text",),
    key_columns=("doc_id", "chunk_id"),
    output_columns=("shout",),
    actor=ActorConfig(min_actors=1, max_actors=2, batch_rows=2),
)


def make_fake_shout_factory():
    """Built as a closure: cloudpickle ships closures BY VALUE, so Ray workers
    never need to import this test module (which isn't importable on workers)."""

    def factory():
        def compute(batch: pa.Table) -> pa.Array:
            return pa.array([(t or "").upper() for t in batch.column("text").to_pylist()], pa.string())

        return compute

    return factory


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


def test_scan_stage_fills_then_converges(ray_cluster, chunks_db) -> None:
    factory = make_fake_shout_factory()
    written = run_scan_column_stage(chunks_db, STAGE, factory=factory, output_type=pa.string())
    assert written == 5

    ds = lance.dataset(str(chunks_db / "chunks.lance"))
    got = ds.to_table(columns=["text", "shout"]).to_pydict()
    assert got["shout"] == [t.upper() for t in got["text"]]

    # Resume is the read: a second run sees zero NULL rows and writes nothing.
    assert run_scan_column_stage(chunks_db, STAGE, factory=factory, output_type=pa.string()) == 0
