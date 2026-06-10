"""Lance Blob V2 + HTTP-Range primitives over Lance datasets.

Blob reads go through ``ds.take_blobs(..., ids=[rowid])`` — lazy, seekable
``BlobFile`` handles — so an HTTP Range maps directly to ``seek(start) +
read(length)``. ``ids`` are stable logical row ids that survive deletes and
compaction; positional ``indices`` are not, so they are never used here.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

import lance

from backend.core.exceptions import ValidationError

_DOC_ID_RE = re.compile(r"^[a-f0-9]{16}$")
_STREAM_CHUNK = 1 << 20  # 1 MiB: amortizes seek cost, bounds per-stream memory


def valid_doc_id(doc_id: str) -> None:
    if not _DOC_ID_RE.match(doc_id):
        raise ValidationError("invalid doc_id")


def _rowid_for_filter(ds: lance.LanceDataset, filter_expr: str) -> int | None:
    """Resolve a SQL filter to a single stable ``_rowid`` (None if no match).

    Callers sanitize values in ``filter_expr`` (doc_id is validated by
    :func:`valid_doc_id`).
    """
    t = ds.to_table(columns=["doc_id"], filter=filter_expr, with_row_id=True)
    if t.num_rows == 0:
        return None
    return int(t.column("_rowid")[0].as_py())


def rowid_for_doc_id(ds: lance.LanceDataset, doc_id: str) -> int | None:
    return _rowid_for_filter(ds, f"doc_id = '{doc_id}'")


def stream_blob_range(
    ds: lance.LanceDataset, column: str, rowid: int, *, start: int, end: int
) -> Iterator[bytes]:
    """Yield bytes of the inclusive ``[start, end]`` range from a blob column."""
    blob = ds.take_blobs(column, ids=[rowid])[0]
    with blob as f:
        f.seek(start)
        remaining = end - start + 1
        while remaining > 0:
            chunk = f.read(min(_STREAM_CHUNK, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


def parse_range(header: str, total: int) -> tuple[int, int] | None:
    """Parse a single ``bytes=start-end`` range header, clamped to ``total``."""
    m = re.match(r"^\s*bytes=(\d*)-(\d*)\s*$", header)
    if not m:
        return None
    s, e = m.group(1), m.group(2)
    if s == "" and e == "":
        return None
    if s == "":
        length = int(e)
        start = max(0, total - length)
        end = total - 1
    else:
        start = int(s)
        end = int(e) if e else total - 1
    if start > end or start >= total:
        return None
    return start, min(end, total - 1)


def doc_blob_size(ds: lance.LanceDataset, column: str, rowid: int) -> int:
    """Probe a blob's size without reading its contents."""
    blob = ds.take_blobs(column, ids=[rowid])[0]
    with blob as f:
        try:
            return f.size()  # type: ignore[attr-defined]
        except AttributeError:
            f.seek(0, 2)
            return f.tell()
