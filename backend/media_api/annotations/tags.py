"""Batch tag writes — the tag↔annotation unification.

Workflow chunk-tags become annotation ROWS: a human ``set`` over a chunk-level
selection, written across MANY units in ONE merge_insert version. Not a model deriver,
so it does NOT funnel through the jobs seam; the annotations table stays mode-blind
(a tag is ``source='human'`` / ``status='accepted'``).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from urllib.parse import quote

import pyarrow as pa
from fastapi import APIRouter

from backend.deps import AuthorDep, StateDep
from backend.lancekit.writer import open_writer
from backend.media_api.annotations.commit import check_base_version, delete_by_ids, finalize_commit
from backend.media_api.annotations.schema import (
    ANNOTATIONS_TABLE,
    NewAnnotation,
    SaveResult,
    TagBatch,
    identity_values,
)
from backend.media_api.media import DatasetParam, table_dataset, validate_doc_key
from backend.state import dataset_handle

if TYPE_CHECKING:
    from collections.abc import Sequence

    from backend.lancekit.descriptor import Declared
    from backend.media_api.annotations.schema import TagWrite

logger = logging.getLogger(__name__)

router = APIRouter(tags=["annotate"])


def tag_id(doc_id: str, keys: Sequence[int], label: str) -> str:
    """Deterministic id for a chunk-level TAG row — ``tag:{doc}:{keys}:{enc(label)}``.
    Idempotent (re-tagging the same chunk maps to the same id, so no duplicates); the
    ``tag:`` prefix + percent-encoded label guarantee it can never collide with a drawn
    shape's random id (so a tag merge_insert never clobbers a shape's geometry)."""
    parts = [doc_id, *(str(k) for k in keys), quote(label, safe="")]
    return "tag:" + ":".join(parts)


def tag_rows(
    adds: Sequence[TagWrite], declared: Declared, author: str, schema: pa.Schema
) -> pa.Table:
    """Full annotation rows for chunk tags: per-row identity stamped, ``shape_type='tag'``,
    label carried, server author; every OTHER field comes from ``NewAnnotation``'s
    defaults (human provenance, geometry/temporal zeroed) — the one row contract, not a
    parallel literal. Rows are DEDUPED by id (a batch may repeat a chunk+label): Lance
    merge_insert does not dedupe duplicate source keys, so two identical-id rows would
    both insert — dedup here prevents that."""
    by_id: dict[str, dict[str, object]] = {}
    for w in adds:
        doc = validate_doc_key(declared, w.doc_id)
        ident = {**identity_values(declared, doc, w.keys), "reviewer": author}
        for label in w.labels:
            tid = tag_id(doc, w.keys, label)
            if tid in by_id:
                continue  # first wins — every row for a given id is identical anyway
            row = NewAnnotation(id=tid, shape_type="tag", label=label).model_dump()
            by_id[tid] = {**ident, **row}
    return pa.Table.from_pylist(list(by_id.values()), schema=schema)


@router.post("/annotations/tags")
def add_tags(
    state: StateDep,
    author: AuthorDep,
    body: TagBatch,
    dataset: DatasetParam = None,
) -> SaveResult:
    """Write a TagBatch: adds insert-if-absent (never clobbering a reviewer's edits to
    an existing tag row), removes delete by the deterministic id. Idempotent re-Save."""
    handle = dataset_handle(state, dataset)
    declared = handle.descriptor.declared
    ds = table_dataset(handle, ANNOTATIONS_TABLE)  # raises NotFoundError if unseeded
    # Optimistic concurrency degrades to table-GLOBAL for a cross-unit batch; optional +
    # last-write-wins per deterministic id is the pragmatic contract.
    check_base_version(ds, body.base_version)

    delta = tag_rows(body.adds, declared, author, ds.schema)
    writer = open_writer(
        dataset=ds, table_id=[handle.id, ANNOTATIONS_TABLE], settings=state.settings
    )
    touched = 0
    if delta.num_rows:
        # INSERT-ONLY (not upsert): a tag that already exists is left untouched, so a
        # re-save never clobbers a reviewer's edits to that tag row (status/label/group).
        # Re-tagging stays idempotent; humans own the row once it exists.
        writer.merge_insert_only(delta, "id")
        touched += delta.num_rows

    remove_ids = [
        tag_id(validate_doc_key(declared, w.doc_id), w.keys, label)
        for w in body.removes
        for label in w.labels
    ]
    if remove_ids:
        delete_by_ids(writer, remove_ids)
        touched += len(remove_ids)

    result = finalize_commit(
        handle,
        ds,
        touched=touched,
        unit_key=f"tags:{len(body.adds)}+{len(body.removes)}",
        sink=state.settings.lineage_sink,
    )
    if touched:
        logger.info(
            "tagged → v%d (%d add-rows, %d remove)", result.version, delta.num_rows, len(remove_ids)
        )
    return result
