"""Parity: the catalog read path returns row-identical results to direct Lance.

Proves :class:`CatalogTableReader` — reading through the lance-ns catalog
``/query`` contract via the in-process :class:`LocalCatalogTransport` (the
catalog's native behavior: translate the ``QueryTableRequest`` into a scan and
return Arrow IPC **file** bytes) — matches :class:`LanceTableReader` across every
scan pattern the backend uses: full scan, column projection, SQL filter, limit,
offset, ``with_row_id``, and ``count_rows``. The live REST path shares the exact
request-build + Arrow-file-decode code, so this pins the client contract; whether
a running server accepts the empty-vector scan is the separate live-probe item
noted in ``reader.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import lance
import pyarrow as pa
import pytest

from common.core.config import Settings
from common.lancekit.reader import (
    CatalogTableReader,
    LanceTableReader,
    LocalCatalogTransport,
    TableReader,
    open_reader,
)

if TYPE_CHECKING:
    from pathlib import Path

# The lance-ns catalog client is an optional runtime dep (only MEDIA_READ_BACKEND=
# catalog needs it): install with `uv pip install lance-namespace-urllib3-client`.
# The parity comparison needs it (CatalogTableReader.to_table lazy-imports it), so
# skip the whole module when it's absent rather than fail.
pytest.importorskip("lance_namespace_urllib3_client")


@pytest.fixture
def dataset(tmp_path: Path) -> lance.LanceDataset:
    """A small deterministic Lance table with mixed column types."""
    statuses = ["prediction", "accepted", "rejected", "reviewed"]
    table = pa.table(
        {
            "doc_id": [f"d{i:02d}" for i in range(20)],
            "status": [statuses[i % 4] for i in range(20)],
            "score": [i / 20 for i in range(20)],
            "text": [f"row {i}" for i in range(20)],
        }
    )
    uri = str(tmp_path / "t.lance")
    lance.write_dataset(table, uri)
    return lance.dataset(uri)


def _pair(dataset: lance.LanceDataset) -> tuple[LanceTableReader, CatalogTableReader]:
    return (
        LanceTableReader(dataset),
        CatalogTableReader(LocalCatalogTransport(dataset), ["ns", "t"]),
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"columns": ["doc_id", "status"]},
        {"filter": "status = 'prediction'"},
        {"columns": ["doc_id"], "filter": "score > 0.5"},
        {"limit": 5},
        {"limit": 5, "offset": 3},
        {"with_row_id": True},
        {
            "columns": ["status"],
            "filter": "status != 'rejected'",
            "limit": 4,
            "offset": 1,
            "with_row_id": True,
        },
    ],
)
def test_catalog_matches_direct(dataset: lance.LanceDataset, kwargs: dict[str, object]) -> None:
    direct, catalog = _pair(dataset)
    dt = direct.to_table(**kwargs)  # ty: ignore[invalid-argument-type]
    ct = catalog.to_table(**kwargs)  # ty: ignore[invalid-argument-type]
    assert ct.num_rows == dt.num_rows
    assert set(ct.column_names) == set(dt.column_names)
    # same columns, same rows in the same (storage-stable) order
    assert ct.select(dt.column_names).to_pylist() == dt.to_pylist()


def test_count_rows_matches(dataset: lance.LanceDataset) -> None:
    direct, catalog = _pair(dataset)
    assert catalog.count_rows() == direct.count_rows() == 20
    assert catalog.count_rows("status = 'prediction'") == direct.count_rows("status = 'prediction'")


def test_both_satisfy_tablereader_protocol(dataset: lance.LanceDataset) -> None:
    direct, catalog = _pair(dataset)
    assert isinstance(direct, TableReader)
    assert isinstance(catalog, TableReader)
    # schema is reachable through the catalog path too (a limit-0 scan)
    assert catalog.schema.names == direct.schema.names


def test_open_reader_flag_selects_backend(
    dataset: lance.LanceDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MEDIA_READ_BACKEND", raising=False)
    direct = open_reader(dataset=dataset, table_id=["ns", "t"], settings=Settings())
    assert isinstance(direct, LanceTableReader)

    monkeypatch.setenv("MEDIA_READ_BACKEND", "catalog")
    catalog = open_reader(dataset=dataset, table_id=["ns", "t"], settings=Settings())
    assert isinstance(catalog, CatalogTableReader)
    # in-process catalog (no MEDIA_CATALOG_URI) is row-identical to direct
    assert catalog.to_table(limit=3).to_pylist() == direct.to_table(limit=3).to_pylist()


def test_invalid_read_backend_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEDIA_READ_BACKEND", "nonsense")
    with pytest.raises(ValueError, match="MEDIA_READ_BACKEND"):
        Settings()
