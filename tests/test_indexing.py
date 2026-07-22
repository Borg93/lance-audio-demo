"""P1.6: index plans keep the engine's gates — never train on NULLs or tiny tables."""

from __future__ import annotations

import lance
import numpy as np
import pyarrow as pa

from ratch.features.indexing import plan_indexes

DIM = 8


def _chunks(tmp_path, rows: int, *, nulls: int = 0):
    vectors: list[list[float] | None] = [
        None if i < nulls else np.random.default_rng(i).normal(size=DIM).tolist()
        for i in range(rows)
    ]
    table = pa.table(
        {
            "doc_id": pa.array([f"d{i:015d}" for i in range(rows)]),
            "text": pa.array(["x"] * rows),
            "audio_path": pa.array(["x.mp4"] * rows),
            "text_embedding": pa.array(vectors, pa.list_(pa.float32(), DIM)),
        }
    )
    lance.write_dataset(table, str(tmp_path / "chunks.lance"))
    return tmp_path


def _plan_for(plans, column):
    [plan] = [p for p in plans if p.column == column]
    return plan


def test_ivf_pq_blocked_while_nulls_remain(tmp_path) -> None:
    plans = plan_indexes(_chunks(tmp_path, rows=300, nulls=3))
    plan = _plan_for(plans, "text_embedding")
    assert plan.blocked is not None and "NULL" in plan.blocked


def test_ivf_pq_blocked_below_num_partitions(tmp_path) -> None:
    plans = plan_indexes(_chunks(tmp_path, rows=90))
    plan = _plan_for(plans, "text_embedding")
    assert plan.blocked is not None and "num_partitions" in plan.blocked


def test_ivf_pq_ready_when_populated_and_large_enough(tmp_path) -> None:
    plans = plan_indexes(_chunks(tmp_path, rows=300))
    plan = _plan_for(plans, "text_embedding")
    assert plan.blocked is None
    assert (plan.kind, plan.num_partitions, plan.num_sub_vectors) == ("IVF_PQ", 256, 64)


def test_missing_column_is_blocked_not_crash(tmp_path) -> None:
    table = pa.table({"doc_id": pa.array(["d1"]), "text": pa.array(["x"]), "audio_path": pa.array(["y"])})
    lance.write_dataset(table, str(tmp_path / "chunks.lance"))
    plan = _plan_for(plan_indexes(tmp_path), "text_embedding")
    assert plan.blocked == "column missing"


def test_missing_table_is_skipped_entirely(tmp_path) -> None:
    assert plan_indexes(tmp_path) == []


def test_scalar_indexes_have_no_row_gate(tmp_path) -> None:
    plans = plan_indexes(_chunks(tmp_path, rows=10))
    assert _plan_for(plans, "text").blocked is None  # FTS
    assert _plan_for(plans, "doc_id").blocked is None  # BTREE
