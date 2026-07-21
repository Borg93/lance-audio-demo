"""The shared commit choreography — one write ritual for every annotations save path.

Both save routes (the per-unit review Save and the batch tag write) perform the same
sequence around their different deltas: optimistic-concurrency check → write via the
writer seam → delete-by-ids → re-open → OpenLineage emit → SaveResult. Extracted here
so the ritual exists ONCE and a change (e.g. the catalog-governed write at merge)
lands in one place.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from backend.core.exceptions import ConflictError
from backend.lancekit.lineage_emit import emit_save
from backend.media_api.annotations.schema import ANNOTATIONS_TABLE, SaveResult
from backend.media_api.media import table_dataset

if TYPE_CHECKING:
    from collections.abc import Sequence

    import lance

    from backend.lancekit.registry import DatasetHandle
    from backend.lancekit.writer import TableWriter


def check_base_version(ds: lance.LanceDataset, base_version: int | None) -> None:
    """Optimistic concurrency: 409 if the table advanced since the client loaded.
    ``None`` skips the check (last-write-wins per id is then the contract)."""
    if base_version is not None and base_version != int(ds.version):
        raise ConflictError(
            f"annotations changed on the server (loaded v{base_version}, now v{int(ds.version)})"
        )


def delete_by_ids(writer: TableWriter, ids: Sequence[str]) -> None:
    """Delete rows by id through the writer seam — quoted, injection-guarded."""
    quoted = ", ".join(_sql_quote(i) for i in ids)
    writer.delete(f"id IN ({quoted})")


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


def _sql_quote(value: str) -> str:
    """SQL single-quoted string literal (doubling quotes) — the injection guard for
    the delete predicate's id list."""
    return "'" + value.replace("'", "''") + "'"
