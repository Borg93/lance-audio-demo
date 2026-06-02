"""Attach derived columns to a Lance dataset via *data evolution* (``add_columns``).

Type-agnostic core: the output column can be any Arrow type (a vector, a string,
a struct), so the same two functions back text embeddings, frame embeddings,
captions, and summaries. Lance writes one new column file alongside the existing
fragments and links it with a metadata-only op — far cheaper than ``merge_insert``,
which rewrites every matched fragment.

Two shapes, by where the input lives:

* :func:`upsert_scan_column` — single pass; the UDF derives each batch's values
  from that batch's own *scanned* columns (e.g. ``text``). Supports incremental
  null-fill so rows added by a later ingest get backfilled.
* :func:`upsert_blob_column` — two passes keyed by ``_rowid``; the input lives in
  a Blob V2 column a scan can't materialise (e.g. frame JPEGs), so we compute by
  id first, then attach by id — order-independent and race-free.

Pure functions over a dataset path + a ``compute`` callable; they import no model
client, so a Ray/Ray Data driver can fan them out later.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import lance
import pyarrow as pa

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import lancedb

logger = logging.getLogger(__name__)


def upsert_scan_column(
    dataset_path: str | Path,
    *,
    name: str,
    output_type: pa.DataType,
    key_columns: list[str],
    read_columns: list[str],
    compute: Callable[[pa.RecordBatch], pa.Array],
    batch_rows: int = 256,
    checkpoint_file: str | Path | None = None,
    overwrite: bool = False,
    progress: Callable[[int], None] | None = None,
) -> int:
    """Attach/refresh a scan-derived column. Returns the number of rows written.

    * Column absent → add it for every row with a single-pass checkpointed UDF.
    * Column present, ``overwrite`` → drop and rebuild.
    * Column present, not ``overwrite`` → fill only residual ``NULL`` rows (rows
      a later ingest appended) via ``merge_insert`` on ``key_columns``.

    ``compute`` receives a batch carrying ``read_columns`` (the fill path also
    includes ``key_columns``, which ``compute`` ignores) and returns the output
    array for that batch.
    """
    ds = lance.dataset(str(dataset_path))
    has_col = name in ds.schema.names

    if has_col and overwrite:
        ds.drop_columns([name])
        ds = lance.dataset(str(dataset_path))
        has_col = False

    if has_col:
        return _fill_null_scan_column(
            ds,
            name=name,
            key_columns=key_columns,
            read_columns=read_columns,
            compute=compute,
            batch_rows=batch_rows,
            progress=progress,
        )

    total = ds.count_rows()
    if total == 0:
        return 0

    schema = pa.schema([pa.field(name, output_type, nullable=True)])

    @lance.batch_udf(output_schema=schema, checkpoint_file=_as_str(checkpoint_file))
    def udf(batch: pa.RecordBatch) -> pa.RecordBatch:
        out = compute(batch)
        if progress is not None:
            progress(batch.num_rows)
        return pa.RecordBatch.from_arrays([out], names=[name])

    logger.info(f"adding column {name} ({output_type}) to {total} row(s) via add_columns")
    ds.add_columns(udf, read_columns=read_columns, batch_size=batch_rows)
    return total


def _fill_null_scan_column(
    ds: lance.LanceDataset,
    *,
    name: str,
    key_columns: list[str],
    read_columns: list[str],
    compute: Callable[[pa.RecordBatch], pa.Array],
    batch_rows: int,
    progress: Callable[[int], None] | None,
) -> int:
    """Fill only ``NULL`` rows of an existing column via ``merge_insert`` on the key."""
    pending = ds.to_table(columns=[*key_columns, *read_columns], filter=f"{name} IS NULL")
    if pending.num_rows == 0:
        logger.info(f"{name} already populated — nothing to fill")
        return 0

    logger.info(f"filling {pending.num_rows} NULL {name} row(s) via merge_insert")
    for batch in pending.to_batches(max_chunksize=batch_rows):
        update = pa.table(
            {**{k: batch.column(k) for k in key_columns}, name: compute(batch)}
        )
        ds.merge_insert(key_columns).when_matched_update_all().execute(update)
        if progress is not None:
            progress(batch.num_rows)
    return pending.num_rows


def upsert_blob_column(
    dataset_path: str | Path,
    *,
    name: str,
    output_type: pa.DataType,
    blob_column: str,
    compute: Callable[[list[bytes]], pa.Array],
    batch_rows: int = 256,
    checkpoint_file: str | Path | None = None,
    overwrite: bool = False,
    progress: Callable[[int], None] | None = None,
) -> int:
    """Attach a column derived from a Blob V2 column. Returns the number of rows written.

    Two passes keyed by ``_rowid`` (stable within a dataset version): compute a
    ``{row_id: value}`` map from the blob payloads, then an ``add_columns`` UDF
    places each value by id. A row id missing from the map raises (loud failure)
    rather than misaligning, so the result is correct regardless of scan order.
    No null-fill: the column is all-or-nothing (``overwrite`` rebuilds).
    """
    ds = lance.dataset(str(dataset_path))
    if name in ds.schema.names:
        if not overwrite:
            logger.info(f"{name} already exists — nothing to do (pass overwrite=True)")
            return 0
        ds.drop_columns([name])
        ds = lance.dataset(str(dataset_path))

    total = ds.count_rows()
    if total == 0:
        return 0

    value_by_row_id: dict[int, Any] = {}
    for batch in ds.to_batches(columns=[], with_row_id=True, batch_size=batch_rows):
        row_ids = batch.column("_rowid").to_pylist()
        values = compute(_read_blobs(ds, blob_column, row_ids)).to_pylist()
        for row_id, value in zip(row_ids, values, strict=True):
            value_by_row_id[row_id] = value
        if progress is not None:
            progress(len(row_ids))

    schema = pa.schema([pa.field(name, output_type, nullable=True)])

    @lance.batch_udf(output_schema=schema, checkpoint_file=_as_str(checkpoint_file))
    def attach(batch: pa.RecordBatch) -> pa.RecordBatch:
        row_ids = batch.column("_rowid").to_pylist()
        out = pa.array([value_by_row_id[r] for r in row_ids], type=output_type)
        return pa.RecordBatch.from_arrays([out], names=[name])

    logger.info(f"attaching column {name} ({output_type}) for {len(value_by_row_id)} row(s)")
    ds.add_columns(attach, read_columns=["_rowid"], batch_size=batch_rows)
    return total


def _read_blobs(ds: lance.LanceDataset, column: str, row_ids: list[int]) -> list[bytes]:
    """Read each row's blob payload by id (Blob V2 columns aren't materialised in a scan)."""
    payloads: list[bytes] = []
    for blob in ds.take_blobs(column, ids=row_ids):
        with blob as handle:
            payloads.append(handle.read())
    return payloads


def ensure_vector_index(
    table: lancedb.table.Table,
    column: str,
    *,
    num_partitions: int,
    num_sub_vectors: int,
    metric: str = "cosine",
) -> bool:
    """Build an IVF_PQ index on ``column`` once it is fully populated.

    Returns ``True`` if an index was (re)built, ``False`` if skipped. Two
    preconditions are checked, both of which otherwise crash Lance's trainer:

    * every row must have a vector — the IVF trainer rejects partial-``NULL``
      columns;
    * the table must have at least ``num_partitions`` rows — IVF k-means needs at
      least one training vector per partition. Below that, flat (brute-force)
      search is used until the table grows, which is fine at small scale.

    Rebuilding after compaction is the caller's responsibility (compaction
    invalidates the row addresses the index points at).
    """
    nulls = table.count_rows(filter=f"{column} IS NULL")
    if nulls > 0:
        logger.warning(f"skipping index on {column}: {nulls} row(s) still NULL")
        return False
    row_count = table.count_rows()
    if row_count < num_partitions:
        logger.warning(
            f"skipping index on {column}: {row_count} row(s) < num_partitions={num_partitions} "
            f"(IVF_PQ needs ≥ num_partitions rows to train; flat search is used until then)"
        )
        return False
    logger.info(
        f"building IVF_PQ {metric} index on {column} "
        f"(num_partitions={num_partitions}, num_sub_vectors={num_sub_vectors})"
    )
    table.create_index(
        metric=metric,
        vector_column_name=column,
        index_type="IVF_PQ",
        num_partitions=num_partitions,
        num_sub_vectors=num_sub_vectors,
        replace=True,
    )
    return True


def _as_str(path: str | Path | None) -> str | None:
    """Coerce an optional path to ``str`` for APIs that don't take ``Path``."""
    return None if path is None else str(path)
