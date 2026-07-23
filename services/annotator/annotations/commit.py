"""The shared commit choreography — one write ritual for every annotations save path.

Both save routes (the per-unit review Save and the batch tag write) perform the same
sequence around their different deltas: optimistic-concurrency check → write via the
writer seam → delete-by-ids → re-open → OpenLineage emit → SaveResult. Extracted here
so the ritual exists ONCE and a change (e.g. the catalog-governed write at merge)
lands in one place.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from annotator.annotations.schema import ANNOTATIONS_TABLE, SaveResult
from common.core.exceptions import ConflictError
from common.lancekit.lineage_emit import emit_save
from common.lancekit.predicate import isin
from common.lancekit.registry import table_dataset

if TYPE_CHECKING:
    from collections.abc import Sequence

    import lance

    from common.lancekit.registry import DatasetHandle
    from common.lancekit.writer import TableWriter


def check_base_version_value(current: int, base_version: int | None) -> None:
    """Optimistic concurrency, source-agnostic: 409 if the table advanced past the
    version the client loaded — ``current`` may be a direct ``ds.version`` or the
    catalog's version primitive. ``None`` skips it (last-write-wins per id)."""
    if base_version is not None and base_version != current:
        raise ConflictError(
            f"annotations changed on the server (loaded v{base_version}, now v{current})"
        )


def check_base_version(ds: lance.LanceDataset, base_version: int | None) -> None:
    """Optimistic-concurrency check against a dataset's current version."""
    check_base_version_value(int(ds.version), base_version)


def delete_by_ids(writer: TableWriter, ids: Sequence[str]) -> None:
    """Delete rows by id through the writer seam — quoted, injection-guarded."""
    writer.delete(isin("id", ids))


def finalize_commit(
    handle: DatasetHandle,
    ds: lance.LanceDataset,
    *,
    touched: int,
    unit_key: str,
    sink: str,
) -> SaveResult:
    """Close out a write: no-op → current version; else re-open the table for the new
    version, emit the pre-merge OpenLineage RunEvent (at merge the catalog mover emits
    instead), and report. Sink from settings (log|stdout|none)."""
    if touched == 0:
        return SaveResult(saved=0, version=int(ds.version))
    fresh = table_dataset(handle, ANNOTATIONS_TABLE)
    emit_save(
        ds=fresh,
        table_uri=handle.table_uri(ANNOTATIONS_TABLE),
        table_name=ANNOTATIONS_TABLE,
        unit_key=unit_key,
        sink=sink,
    )
    return SaveResult(saved=touched, version=int(fresh.version))
