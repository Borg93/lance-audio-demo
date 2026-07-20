"""Seed a Lance ``annotations`` table (ra-anno engine schema) for the annotator wire.

Writes ``<db>/annotations.lance`` with a few sample shapes on a real frame, so the
`/annotate` route + `GET /api/annotations/{doc}/{speech}/{chunk}` have real data to
serve as Arrow IPC. The schema matches the vendored engine's ANNOTATION_COLUMNS
(frontend/src/lib/engine/schema.ts).

    uv run python scripts/seed_annotations.py [db_path] [doc_id]
"""

from __future__ import annotations

import sys

import lance
import pyarrow as pa

SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("doc_id", pa.string()),
        ("speech_id", pa.int64()),
        ("chunk_id", pa.int64()),
        ("frame_idx", pa.int64()),
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
        ("group", pa.string()),
        ("group_id", pa.string()),
        ("difficult", pa.bool_()),
        ("mask", pa.string()),
        ("metadata", pa.string()),
    ]
)


def seed(db_path: str, doc_id: str) -> str:
    """Write 3 sample annotations (2 predicted text-lines + 1 accepted figure)."""
    rows = {
        "id": ["a1", "a2", "a3"],
        "doc_id": [doc_id] * 3,
        "speech_id": [0] * 3,
        "chunk_id": [19] * 3,
        "frame_idx": [0] * 3,
        "shape_type": ["rectangle", "rectangle", "polygon"],
        "x": [40.0, 210.0, 0.0],
        "y": [40.0, 150.0, 0.0],
        "width": [160.0, 120.0, 0.0],
        "height": [90.0, 80.0, 0.0],
        "rotation": [0.0, 0.0, 0.0],
        "polygon": [[], [], [120.0, 240.0, 260.0, 250.0, 190.0, 320.0]],
        "text": ["regeringen", "principmodellen", "region"],
        "label": ["text-line", "text-line", "figure"],
        "status": ["prediction", "prediction", "accepted"],
        "group": ["lines", "lines", "figures"],
        "group_id": ["", "", ""],
        "difficult": [False, False, False],
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
