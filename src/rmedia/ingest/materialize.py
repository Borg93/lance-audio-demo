"""Materialize external blob-v2 columns into Lance-managed bytes — the lance-ns way.

lance-ns's medallion ingest (``services/medallion/services/ingest.py::ingest_to_bronze``)
writes media as a **managed** blob-v2 column — the bytes live *in* the dataset — so a
plain directory copy to S3 carries them and they resolve anywhere, with no client S3
creds and no dangling external pointer. Our ingest writes ``documents.media_blob`` as an
**external** ``Blob.from_uri`` (``file://``) reference, which only resolves where that path
exists (fine on the build box, broken in a pod / on S3-only compute).

This converts external → managed **in place**, mirroring lance-ns's own re-wrap
(``compute._carry_forward``: ``read_blobs`` resolves each pointer to bytes, then
``blob_array`` re-writes them managed at file format 2.2). Run it locally (where the
``file://`` sources still resolve) *before* moving the dataset to S3; afterwards the
dataset is fully self-contained — the RASK_LANDING §4.4 remediation, as code.

    uv run rmedia --db parity_new.lance materialize-blobs
    uv run rmedia --db parity_new.lance materialize-blobs --table documents
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import lance
import pyarrow as pa
from lance import blob_array, blob_field

from rmedia.core.blobs import blob_field_names
from rmedia.core.dataset import overwrite_dataset

logger = logging.getLogger(__name__)


def materialize_blobs(db_path: str | Path, table: str = "documents") -> dict[str, dict[str, int]]:
    """Rewrite every blob-v2 column of ``table`` as Lance-managed bytes (in place).

    External ``Blob.from_uri`` pointers become managed bytes; already-managed columns
    (e.g. inline thumbnails) are re-wrapped identically (a no-op in effect). The table is
    overwritten via the sanctioned :func:`~rmedia.core.dataset.overwrite_dataset` path so
    the 2.2 + stable-row-id invariants and any descriptor stamp survive.

    Returns ``{column: {"rows": n, "bytes": total}}`` for each materialized blob column.
    """
    uri = str(Path(db_path) / f"{table}.lance")
    ds = lance.dataset(uri)
    schema = ds.schema
    blob_cols = blob_field_names(schema)
    if not blob_cols:
        logger.info("%s: no blob columns — nothing to materialize", table)
        return {}

    n = ds.count_rows()
    non_blob = [f.name for f in schema if f.name not in blob_cols]
    base = ds.to_table(columns=non_blob) if non_blob else pa.table({})

    columns: dict[str, Any] = {}
    fields: list[pa.Field] = []
    stats: dict[str, dict[str, int]] = {}
    for field in schema:
        if field.name in blob_cols:
            # read_blobs resolves the pointer (external file:// or inline) to bytes;
            # blob_array(bytes) re-writes them MANAGED — the lance-ns re-wrap contract.
            payloads = [payload for _addr, payload in ds.read_blobs(field.name, indices=list(range(n)))]
            columns[field.name] = blob_array(payloads)
            fields.append(blob_field(field.name))
            stats[field.name] = {"rows": len(payloads), "bytes": sum(len(p) for p in payloads)}
        else:
            columns[field.name] = base.column(field.name)
            fields.append(field)

    out = pa.table(
        {field.name: columns[field.name] for field in schema},
        schema=pa.schema(fields, metadata=schema.metadata),
    )
    overwrite_dataset(uri, out)
    logger.info("materialized %d blob column(s) of %s: %s", len(blob_cols), table, stats)
    return stats
