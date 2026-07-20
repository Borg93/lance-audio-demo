"""Parity: the catalog write path produces a table row-identical to direct Lance.

Proves :class:`CatalogTableWriter` over the in-process
:class:`LocalCatalogWriteTransport` (the catalog's native merge_insert/delete
behavior) leaves a table byte-identical to :class:`LanceTableWriter` after the two
ops annotate.py's Save performs — a merge UPSERT (update matched + insert unmatched)
and a delete-by-predicate. The live REST path shares the same seam. No catalog client
needed (the in-process transport is pure Lance), so this always runs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import lance
import pyarrow as pa
import pytest
from backend.core.config import Settings
from backend.lancekit.writer import (
    CatalogTableWriter,
    LanceTableWriter,
    LocalCatalogWriteTransport,
    TableWriter,
    open_writer,
)

if TYPE_CHECKING:
    from pathlib import Path


def _base() -> pa.Table:
    return pa.table(
        {
            "id": ["a", "b", "c"],
            "status": ["prediction", "accepted", "prediction"],
            "x": pa.array([1.0, 2.0, 3.0], pa.float32()),
        }
    )


@pytest.fixture
def two(tmp_path: Path) -> tuple[lance.LanceDataset, lance.LanceDataset, str, str]:
    u1, u2 = str(tmp_path / "direct.lance"), str(tmp_path / "catalog.lance")
    lance.write_dataset(_base(), u1)
    lance.write_dataset(_base(), u2)
    return lance.dataset(u1), lance.dataset(u2), u1, u2


def test_catalog_write_matches_direct(two: tuple[lance.LanceDataset, lance.LanceDataset, str, str]) -> None:
    ds1, ds2, u1, u2 = two
    direct = LanceTableWriter(ds1)
    catalog = CatalogTableWriter(LocalCatalogWriteTransport(ds2), ["ns", "t"])

    # same upsert (update `a`, insert `d`) + same delete (`b`) through both writers
    delta = pa.table(
        {"id": ["a", "d"], "status": ["rejected", "accepted"], "x": pa.array([9.0, 4.0], pa.float32())}
    )
    direct.merge_upsert(delta, "id")
    catalog.merge_upsert(delta, "id")
    direct.delete("id IN ('b')")
    catalog.delete("id IN ('b')")

    t1 = {r["id"]: r for r in lance.dataset(u1).to_table().to_pylist()}
    t2 = {r["id"]: r for r in lance.dataset(u2).to_table().to_pylist()}
    assert t1 == t2  # row-identical (merge_insert may reorder — compare by id)
    assert t1["a"]["status"] == "rejected"  # updated
    assert "d" in t1 and t1["d"]["x"] == 4.0  # inserted
    assert "b" not in t1  # deleted


def test_open_writer_flag_selects_backend(
    two: tuple[lance.LanceDataset, lance.LanceDataset, str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    ds1, ds2, *_ = two
    monkeypatch.delenv("MEDIA_WRITE_BACKEND", raising=False)
    assert isinstance(open_writer(dataset=ds1, table_id=["ns", "t"], settings=Settings()), LanceTableWriter)
    monkeypatch.setenv("MEDIA_WRITE_BACKEND", "catalog")
    # catalog + no MEDIA_CATALOG_URI ⇒ in-process transport (no client needed)
    assert isinstance(
        open_writer(dataset=ds2, table_id=["ns", "t"], settings=Settings()), CatalogTableWriter
    )


def test_both_satisfy_tablewriter_protocol(
    two: tuple[lance.LanceDataset, lance.LanceDataset, str, str],
) -> None:
    ds1, ds2, *_ = two
    assert isinstance(LanceTableWriter(ds1), TableWriter)
    assert isinstance(CatalogTableWriter(LocalCatalogWriteTransport(ds2), ["ns", "t"]), TableWriter)


def test_invalid_write_backend_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEDIA_WRITE_BACKEND", "nonsense")
    with pytest.raises(ValueError, match="backend must be"):
        Settings()
