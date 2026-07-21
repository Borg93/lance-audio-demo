"""Seed a Lance ``annotations`` table for the annotator wire.

Writes ``<db>/annotations.lance`` with a few sample shapes on a real frame. The
contract columns come STRAIGHT from the backend's ``EMPTY_SCHEMA`` (the single source
of truth — ``backend/media_api/annotations/schema.py``); this script only prepends the
demo descriptor's identity columns. A test asserts the seeded dataset matches the
composition, so drift fails loudly.

    uv run python scripts/seed_annotations.py [db_path] [doc_id]
"""

from __future__ import annotations

import sys
from pathlib import Path

import lance
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root → `backend` importable

from annotator.annotations.schema import EMPTY_SCHEMA

SCHEMA = pa.schema(
    [
        # identity (the demo descriptor's key fields) — the ONLY columns this script
        # defines; everything else is the backend contract, imported verbatim.
        ("doc_id", pa.string()),
        ("speech_id", pa.int64()),
        ("chunk_id", pa.int64()),
        ("frame_idx", pa.int64()),
        *EMPTY_SCHEMA,
    ]
)


def seed(db_path: str, doc_id: str) -> str:
    """3 sample rows: 2 model predictions (varying confidence/uncertainty) + 1 human-accepted."""
    rows = {
        "doc_id": [doc_id] * 3,
        "speech_id": [0] * 3,
        "chunk_id": [19] * 3,
        "frame_idx": [0] * 3,
        "id": ["a1", "a2", "a3"],
        "shape_type": ["rectangle", "rectangle", "polygon"],
        "x": [40.0, 210.0, 0.0],
        "y": [40.0, 150.0, 0.0],
        "width": [160.0, 120.0, 0.0],
        "height": [90.0, 80.0, 0.0],
        "rotation": [0.0, 0.0, 0.0],
        "polygon": [[], [], [120.0, 240.0, 260.0, 250.0, 190.0, 320.0]],
        "t_start": [0.0, 0.0, 0.0],  # image annotations have no time axis
        "t_end": [0.0, 0.0, 0.0],
        "text": ["regeringen", "principmodellen", "region"],
        "label": ["text-line", "text-line", "figure"],
        "status": ["prediction", "prediction", "accepted"],
        "source": ["model:htr-trocr@v1", "model:htr-trocr@v1", "manual"],
        "reviewer": ["", "", "gabriel"],
        "confidence": [0.88, 0.61, 1.0],  # a2 is low-confidence → top of the review queue
        "uncertainty": [0.18, 0.72, 0.0],
        "model_version": ["htr-trocr@v1", "htr-trocr@v1", ""],
        "group": ["lines", "lines", "figures"],
        "group_id": ["", "", ""],
        "reading_order": [0, 1, -1],
        "difficult": [False, True, False],
        "links": ["[]", "[]", "[]"],
        "mask": ["", "", ""],
        "metadata": ["{}", "{}", "{}"],
    }
    uri = f"{db_path.rstrip('/')}/annotations.lance"
    lance.write_dataset(
        pa.table(rows, schema=SCHEMA),
        uri,
        mode="overwrite",
        data_storage_version="2.2",
        enable_stable_row_ids=True,
    )
    return uri


if __name__ == "__main__":
    db = sys.argv[1] if len(sys.argv) > 1 else "transcripts_v2.lance"
    doc = sys.argv[2] if len(sys.argv) > 2 else "fe00cd746463ad2c"
    out = seed(db, doc)
    ds = lance.dataset(out)
    print(f"seeded {ds.count_rows()} annotations → {out} (v{ds.version})")
    print("columns:", [f.name for f in ds.schema])
