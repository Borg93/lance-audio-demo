"""Ray Data drivers — distribute a stage's compute, keep every commit driver-side.

Three drivers, one per :class:`~rmedia.core.registry.StageShape`, all built on
the same topology (LANCE_MEDIA_MERGE §4.3):

* **actors compute, the driver commits** — ``lance_ray.read_lance`` fans the
  pending rows out to a ``map_batches`` actor pool (each actor holds a warm
  client via the injected zero-arg ``factory``), and the driver serially
  applies ``merge_insert`` / ``add_columns`` as result batches stream back.
  Only plain appends may involve parallel workers (Appends never conflict).
* **heavy blobs never transit Ray Data blocks** — blob stages ship only
  ``_rowid``s; each actor opens the dataset itself and reads payloads lazily
  via ``take_blobs``.
* **resume is a property of the read**, not bookkeeping: scan stages read
  ``WHERE <column> IS NULL``, blob stages skip checkpointed row ids, append
  stages diff existing output keys — so re-running a killed stage always
  converges with no duplicates.

The module imports no modality or client code; compute factories arrive from
the composition root already bound to their clients.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import lance
import lance_ray
import pyarrow as pa

from rmedia.core.engine import _ValueCheckpoint, attach_values_by_rowid

if TYPE_CHECKING:
    from collections.abc import Callable

    import ray.data

    from rmedia.core.registry import Stage

logger = logging.getLogger(__name__)


def _table_uri(db_path: str | Path, table: str) -> str:
    # Resolved absolute path: a relative URI would be re-rooted inside Ray's
    # runtime-env working-dir COPY on the workers (a stale snapshot), silently
    # missing data files written after job submission.
    return str((Path(db_path) / f"{table}.lance").resolve())


class _ScanActor:
    """Generic scan-stage actor: warm compute over scanned columns."""

    def __init__(self, factory: Callable[[], Callable[[pa.Table], pa.Array]], out_name: str) -> None:
        self._fn = factory()
        self._out = out_name

    def __call__(self, batch: pa.Table) -> pa.Table:
        return batch.append_column(self._out, self._fn(batch))


class _BlobActor:
    """Generic blob-stage actor: reads its own payloads lazily by ``_rowid``.

    Ray Data blocks carry only row ids — the actor opens the dataset once and
    streams bytes via ``take_blobs`` (never materialising blobs into the
    object store), skipping ids the driver already has checkpointed.
    """

    def __init__(
        self,
        factory: Callable[[], Callable[[list[bytes]], pa.Array]],
        dataset_uri: str,
        blob_column: str,
        out_name: str,
        done_ids: frozenset[int],
    ) -> None:
        self._fn = factory()
        self._ds = lance.dataset(dataset_uri)
        self._blob_column = blob_column
        self._out = out_name
        self._done = done_ids

    def __call__(self, batch: pa.Table) -> pa.Table:
        row_ids = [r for r in batch.column("_rowid").to_pylist() if r not in self._done]
        if not row_ids:
            return pa.table({"_rowid": pa.array([], pa.uint64()), self._out: pa.array([], pa.string())}).cast(
                pa.schema([("_rowid", pa.uint64()), (self._out, pa.string())])
            )
        payloads: list[bytes] = []
        for blob in self._ds.take_blobs(self._blob_column, ids=row_ids):
            with blob as handle:
                payloads.append(handle.read())
        values = self._fn(payloads)
        return pa.table({"_rowid": pa.array(row_ids, pa.uint64()), self._out: values})


class _RowsActor:
    """Generic append-stage actor: source rows in, output-table rows out."""

    def __init__(self, factory: Callable[[], Callable[[pa.Table], pa.Table]]) -> None:
        self._fn = factory()

    def __call__(self, batch: pa.Table) -> pa.Table:
        return self._fn(batch)


def _map_batches(source: ray.data.Dataset, actor_cls: type, stage: Stage, **ctor: Any) -> ray.data.Dataset:
    return source.map_batches(
        actor_cls,
        batch_format="pyarrow",
        batch_size=stage.actor.batch_rows,
        concurrency=(stage.actor.min_actors, stage.actor.max_actors),
        num_cpus=stage.actor.num_cpus,
        num_gpus=stage.actor.num_gpus or None,
        fn_constructor_kwargs=ctor,
    )


def _gate_filter(db_path: str | Path, stage: Stage) -> str | None:
    """SQL filter dropping docs the stage's MIME gate rejects (skips counted by caller).

    Doc-level MIME lives on the documents table; per LANCE_MEDIA_MERGE §4.3
    gated stages *skip* non-matching docs rather than crash. Returns a
    ``doc_id IN (...)`` filter, or ``None`` when the stage is ungated.
    """
    if stage.media_gate is None:
        return None
    docs = lance.dataset(_table_uri(db_path, "documents")).to_table(columns=["doc_id", "media_mime"])
    admitted = [
        doc_id
        for doc_id, mime in zip(docs["doc_id"].to_pylist(), docs["media_mime"].to_pylist(), strict=True)
        if stage.media_gate.admits(mime)
    ]
    skipped = docs.num_rows - len(admitted)
    if skipped:
        logger.info("stage %s: media gate skipped %s/%s document(s)", stage.name, skipped, docs.num_rows)
    quoted = ", ".join(f"'{d}'" for d in admitted)
    return f"doc_id IN ({quoted})" if admitted else "doc_id IN ('')"


def run_scan_column_stage(
    db_path: str | Path,
    stage: Stage,
    *,
    factory: Callable[[], Callable[[pa.Table], pa.Array]],
    output_type: pa.DataType,
) -> int:
    """Backfill ``stage.output_columns[0]`` for every NULL row. Returns rows written.

    The column is null-added first (metadata-only) when absent, which makes the
    whole stage one idempotent fill: read ``WHERE col IS NULL`` via lance-ray,
    compute on the actor pool, and ``merge_insert`` per result batch —
    serialized here on the driver (Merge conflicts with nearly everything).
    """
    [name] = stage.output_columns
    uri = _table_uri(db_path, stage.table)
    ds = lance.dataset(uri)
    if name not in ds.schema.names:
        ds.add_columns(pa.field(name, output_type, nullable=True))
        ds = lance.dataset(uri)

    pending = ds.count_rows(filter=f"{name} IS NULL")
    if pending == 0:
        logger.info("stage %s: nothing to fill", stage.name)
        return 0

    filters = [f"{name} IS NULL"]
    gate = _gate_filter(db_path, stage)
    if gate is not None:
        filters.append(gate)
    source = lance_ray.read_lance(
        uri,
        columns=[*stage.key_columns, *stage.read_columns],
        filter=" AND ".join(filters),
    )
    results = _map_batches(source, _ScanActor, stage, factory=factory, out_name=name)

    written = 0
    key_columns = list(stage.key_columns)
    for batch in (cast("pa.Table", b) for b in results.iter_batches(batch_format="pyarrow")):
        if batch.num_rows == 0:
            continue
        update = batch.select([*key_columns, name])
        ds.merge_insert(key_columns).when_matched_update_all().execute(update)
        written += batch.num_rows
        logger.info("stage %s: %s/%s row(s) committed", stage.name, written, pending)
    return written


def run_blob_column_stage(
    db_path: str | Path,
    stage: Stage,
    *,
    factory: Callable[[], Callable[[list[bytes]], pa.Array]],
    output_type: pa.DataType,
    checkpoint_file: str | Path | None = None,
    overwrite: bool = False,
) -> int:
    """Derive a column from blob payloads, distributed. Returns rows written.

    Two passes keyed by ``_rowid`` (the engine's blob shape, compute pass
    distributed): actors read their own payloads and stream ``(_rowid, value)``
    pairs back; the driver persists each result batch to the JSONL sidecar
    (kill-resumable), then attaches the full map via the engine's
    ``add_columns`` UDF. All-or-nothing per column, like the engine.
    """
    [name] = stage.output_columns
    if stage.blob_column is None:  # narrowed by the registry validator; keeps ty happy
        raise ValueError(f"stage {stage.name}: blob stage without blob_column")
    uri = _table_uri(db_path, stage.table)
    ds = lance.dataset(uri)
    if name in ds.schema.names:
        if not overwrite:
            logger.info("stage %s: %s already exists — nothing to do", stage.name, name)
            return 0
        ds.drop_columns([name])
        ds = lance.dataset(uri)

    total = ds.count_rows()
    if total == 0:
        return 0

    ckpt = _ValueCheckpoint(checkpoint_file)
    value_by_row_id: dict[int, Any] = ckpt.load()

    source = lance_ray.read_lance(uri, columns=[], scanner_options={"with_row_id": True})
    results = _map_batches(
        source,
        _BlobActor,
        stage,
        factory=factory,
        dataset_uri=uri,
        blob_column=stage.blob_column,
        out_name=name,
        done_ids=frozenset(value_by_row_id),
    )
    for batch in (cast("pa.Table", b) for b in results.iter_batches(batch_format="pyarrow")):
        pairs = list(zip(batch.column("_rowid").to_pylist(), batch.column(name).to_pylist(), strict=True))
        ckpt.extend(pairs)  # persist before progress — a kill costs at most the in-flight batch
        value_by_row_id.update(pairs)
        logger.info("stage %s: %s/%s value(s) computed", stage.name, len(value_by_row_id), total)

    attach_values_by_rowid(
        uri,
        name=name,
        output_type=output_type,
        value_by_row_id=value_by_row_id,
        batch_rows=stage.actor.batch_rows,
        checkpoint_file=checkpoint_file,
    )
    ckpt.cleanup()
    return total


def run_append_rows_stage(
    db_path: str | Path,
    stage: Stage,
    *,
    factory: Callable[[], Callable[[pa.Table], pa.Table]],
    output_schema: pa.Schema,
    create_output: Callable[[], object],
) -> int:
    """Append derived rows to ``stage.output_table``. Returns rows appended.

    Resume = key diff: source rows whose ``stage.key_columns`` already exist in
    the output table are dropped before fan-out. Result batches are appended by
    the driver (plain Appends are the one concurrency-safe write, but a single
    appender keeps ordering deterministic at this scale). ``create_output``
    creates the output table through :mod:`rmedia.core.dataset` when missing.
    """
    from rmedia.core.dataset import append_rows  # local import: dataset ← driver, never the reverse

    uri = _table_uri(db_path, stage.table)
    out_uri = _table_uri(db_path, stage.output_table or "")

    filters = []
    gate = _gate_filter(db_path, stage)
    if gate is not None:
        filters.append(gate)

    key_columns = list(stage.key_columns)
    done: set[tuple[Any, ...]] = set()
    if Path(out_uri).exists():
        existing = lance.dataset(out_uri).to_table(columns=key_columns)
        done = set(zip(*(existing[k].to_pylist() for k in key_columns), strict=True)) if existing.num_rows else set()
    else:
        create_output()

    source = lance_ray.read_lance(
        uri,
        columns=list(dict.fromkeys([*stage.key_columns, *stage.read_columns])),
        filter=" AND ".join(filters) if filters else None,
    )
    if done:
        done_broadcast = done  # captured by the lambda; small (key tuples only)

        def _drop_done(batch: pa.Table) -> pa.Table:
            keys = zip(*(batch[k].to_pylist() for k in key_columns), strict=True)
            mask = pa.array([k not in done_broadcast for k in keys], pa.bool_())
            return batch.filter(mask)

        source = source.map_batches(_drop_done, batch_format="pyarrow")

    results = _map_batches(source, _RowsActor, stage, factory=factory)

    appended = 0
    for batch in (cast("pa.Table", b) for b in results.iter_batches(batch_format="pyarrow")):
        if batch.num_rows == 0:
            continue
        append_rows(out_uri, batch.cast(output_schema))
        appended += batch.num_rows
        logger.info("stage %s: %s row(s) appended", stage.name, appended)
    return appended
