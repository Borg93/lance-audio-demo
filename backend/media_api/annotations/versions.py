"""Version history + time-travel — the compare-versions read-side.

The audit trail of the write plane: who = ``reviewer``, when = the Lance version's
timestamp, what = lineage; this module serves the WHEN (per-unit history + historical
snapshots) that powers the annotator's compare-versions panel.
"""

from __future__ import annotations

from typing import Annotated

import lance
from fastapi import APIRouter, Query
from pydantic import BaseModel

from backend.core.exceptions import NotFoundError
from backend.deps import StateDep
from backend.media_api.annotations.schema import ANNOTATIONS_TABLE
from backend.media_api.media import DatasetParam, chunk_key_filter, table_dataset, validate_doc_key
from backend.state import dataset_handle

router = APIRouter(tags=["annotate"])


class AnnotationVersion(BaseModel):
    """One point in a unit's edit history — a Lance version + when it was committed +
    how many annotations this unit had at it (the audit/compare-versions trail)."""

    version: int
    timestamp: str
    count: int


@router.get("/annotations/{doc_id}/{speech_id}/{chunk_id}/versions")
def annotation_versions(
    state: StateDep,
    doc_id: str,
    speech_id: int,
    chunk_id: int,
    dataset: DatasetParam = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 20,
) -> list[AnnotationVersion]:
    """The unit's edit history (most-recent first, capped): each Lance version + its
    timestamp + the count of THIS unit's annotations at it. Powers the compare-versions
    panel — the read-side of the write-plane provenance story (who=reviewer, when=version,
    what=lineage)."""
    handle = dataset_handle(state, dataset)
    declared = handle.descriptor.declared
    doc_id = validate_doc_key(declared, doc_id)
    try:
        ds = table_dataset(handle, ANNOTATIONS_TABLE)
    except NotFoundError:
        return []
    where = chunk_key_filter(declared, doc_id, (speech_id, chunk_id))
    out: list[AnnotationVersion] = []
    for v in list(reversed(ds.versions()))[:limit]:  # most recent first, capped
        vnum = int(v["version"])
        count = ds.checkout_version(vnum).to_table(filter=where, columns=["id"]).num_rows
        ts = v.get("timestamp")
        out.append(AnnotationVersion(version=vnum, timestamp=iso(ts), count=count))
    return out


def iso(ts: object) -> str:
    """A Lance version timestamp → ISO string (datetime or already-string)."""
    isofmt = getattr(ts, "isoformat", None)
    return isofmt() if callable(isofmt) else str(ts or "")


def checkout(ds: lance.LanceDataset, version: int) -> lance.LanceDataset:
    """Time-travel to a version, translating Lance's raw not-found (an out-of-range or
    reclaimed version) into a clean NotFoundError — mirroring ``table_dataset`` so a bad
    ``?version`` is a 404, not an opaque 500."""
    try:
        return ds.checkout_version(version)
    except (ValueError, OSError) as e:
        raise NotFoundError(f"annotations version {version} not found") from e
