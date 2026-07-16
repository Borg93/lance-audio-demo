"""P1.8 resume/idempotency: SIGKILL a Ray scan stage mid-fill, re-run, converge.

Builds a scratch DB from the parity chunks table (embedding column dropped),
then runs the REAL text-embedding stage (small batches so several merge_insert
commits happen) in a child process and SIGKILLs it after the first commits
land. Evidence printed: NULL-residual counts before / after the kill / after
the re-run, plus the duplicate-key count (must be 0), then ``RESUME OK``.
"""

from __future__ import annotations

import argparse
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import lance
import pyarrow as pa

CHILD = """
from functools import partial
import sys
from rmedia.core.driver import run_scan_column_stage
from rmedia.core.registry import ActorConfig, Stage, StageShape
from rmedia.features.embed_columns import VECTOR_TYPE
from rmedia.features.ray_bindings import text_embed_compute

stage = Stage(
    name="text_embedding",
    shape=StageShape.SCAN_COLUMN,
    table="chunks",
    read_columns=("text",),
    key_columns=("doc_id", "speech_id", "chunk_id"),
    output_columns=("text_embedding",),
    actor=ActorConfig(min_actors=1, max_actors=1, batch_rows=8),
)
run_scan_column_stage(sys.argv[1], stage, factory=partial(text_embed_compute, sys.argv[2]), output_type=VECTOR_TYPE)
print("CHILD-DONE")
"""


def nulls(db: Path) -> int:
    return lance.dataset(str(db / "chunks.lance")).count_rows(filter="text_embedding IS NULL")


def duplicate_keys(db: Path) -> int:
    ds = lance.dataset(str(db / "chunks.lance"))
    keys = ds.to_table(columns=["doc_id", "speech_id", "chunk_id"])
    unique = len(
        set(
            zip(
                keys["doc_id"].to_pylist(),
                keys["speech_id"].to_pylist(),
                keys["chunk_id"].to_pylist(),
                strict=True,
            )
        )
    )
    return keys.num_rows - unique


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="parity_new.lance")
    parser.add_argument("--db", default="resume_scratch.lance")
    parser.add_argument("--embed-url", default="http://127.0.0.1:8011")
    args = parser.parse_args()

    scratch = Path(args.db)
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir()
    source = lance.dataset(str(Path(args.source) / "chunks.lance"))
    stripped = source.to_table(columns=["doc_id", "speech_id", "chunk_id", "text"])
    # Tile to ~1800 rows with UNIQUE keys so the fill spans many 8-row commits —
    # a wide-open window for a genuinely mid-stage SIGKILL.
    tiles = []
    for i in range(20):
        tile = stripped.set_column(
            2,
            "chunk_id",
            pa.array([c + 10_000 * i for c in stripped["chunk_id"].to_pylist()], pa.int32()),
        )
        tiles.append(tile)
    tiled = pa.concat_tables(tiles)
    lance.write_dataset(tiled, str(scratch / "chunks.lance"))
    total = tiled.num_rows

    print(f"  NULL-residual before: {total}/{total} (column not yet added)")

    child = subprocess.Popen(
        [sys.executable, "-c", CHILD, str(scratch), args.embed_url],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    killed_at = None
    for _ in range(6000):
        time.sleep(0.05)
        if child.poll() is not None:
            break
        try:
            remaining = nulls(scratch)
        except Exception:  # noqa: BLE001 — column not added yet
            continue
        # Kill ONLY in a provably partial state: some commits landed, some pending.
        if 8 < remaining < total - 8:
            child.send_signal(signal.SIGKILL)
            child.wait()
            killed_at = nulls(scratch)
            break
    if killed_at is None:
        raise SystemExit("child never observed in a partial state — enlarge the scratch table")
    print(f"  SIGKILLed mid-stage: NULL-residual after kill = {killed_at}/{total} (partial fill persisted)")

    subprocess.run(
        [sys.executable, "-c", CHILD, str(scratch), args.embed_url],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    after = nulls(scratch)
    dupes = duplicate_keys(scratch)
    print(f"  NULL-residual after re-run: {after}/{total}")
    print(f"  duplicate-key count: {dupes}")
    if after != 0 or dupes != 0:
        raise SystemExit("RESUME FAILED")
    print("RESUME OK")


if __name__ == "__main__":
    main()
