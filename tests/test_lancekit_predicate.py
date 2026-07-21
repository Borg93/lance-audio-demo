"""The shared predicate contract + the descriptor drift-sync it composes with.

One quoting implementation for every seam (docs/LANCEDB_SDK_AUDIT.md backlog #1):
these tests pin the literal rendering, the validation-only ``validate_doc_key``
contract (raw key stored, escaping only at render time), and the per-table
descriptor refresh triggered by an observed version drift (backlog #2).
"""

from __future__ import annotations

from pathlib import Path

import lance
import lancedb
import pyarrow as pa
import pytest
from backend.core.exceptions import ValidationError
from backend.lancekit.descriptor import DatasetDescriptor, Declared
from backend.lancekit.introspect import discover_tables
from backend.lancekit.predicate import and_, eq, isin, ne, quote_literal
from backend.lancekit.registry import DatasetHandle
from backend.media_api.media import chunk_key_filter, table_dataset, validate_doc_key


class TestQuoteLiteral:
    def test_string_is_quoted_and_escaped(self) -> None:
        assert quote_literal("O'Brien") == "'O''Brien'"

    def test_every_quote_doubled(self) -> None:
        assert quote_literal("a'b'c") == "'a''b''c'"

    def test_int_is_bare(self) -> None:
        assert quote_literal(7) == "7"

    def test_float_is_bare(self) -> None:
        assert quote_literal(1.5) == "1.5"

    def test_bool_is_sql_keyword_not_int(self) -> None:
        assert quote_literal(True) == "TRUE"
        assert quote_literal(False) == "FALSE"


class TestEqIsinAnd:
    def test_eq_renders_field_and_literal(self) -> None:
        assert eq("doc_id", "a'b") == "doc_id = 'a''b'"

    def test_eq_int_bare(self) -> None:
        assert eq("frame_idx", 0) == "frame_idx = 0"

    def test_isin_dedupes_sorts_and_escapes(self) -> None:
        assert isin("id", ["b", "a", "b", "o'k"]) == "id IN ('a', 'b', 'o''k')"

    def test_and_skips_empty_clauses(self) -> None:
        assert and_("a = 1", "", "b = 2") == "a = 1 AND b = 2"

    def test_ne_renders_field_and_literal(self) -> None:
        assert ne("doc_id", "a'b") == "doc_id != 'a''b'"

    def test_isin_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one value"):
            isin("id", [])


PERMISSIVE_DECLARED = Declared.model_validate(
    {"identity": {"key_fields": ["doc_id", "part"], "doc_key_pattern": r"^[A-Za-z' -]{1,64}$"}}
)


class TestValidateDocKeyIsValidationOnly:
    """The raw key is what gets STORED (row identity, tag ids) — escaping is the
    predicate renderer's job, never the validator's."""

    def test_returns_raw_key_unescaped(self) -> None:
        assert validate_doc_key(PERMISSIVE_DECLARED, "O'Brien") == "O'Brien"

    def test_chunk_key_filter_escapes_at_render_time(self) -> None:
        raw = validate_doc_key(PERMISSIVE_DECLARED, "O'Brien")
        assert chunk_key_filter(PERMISSIVE_DECLARED, raw, (3,)) == (
            "doc_id = 'O''Brien' AND part = 3"
        )

    def test_still_rejects_pattern_violations(self) -> None:
        with pytest.raises(ValidationError):
            validate_doc_key(PERMISSIVE_DECLARED, "nope_underscore")


@pytest.fixture
def drift_handle(tmp_path: Path) -> DatasetHandle:
    """A DatasetHandle over a real tmp dataset with one table, descriptor cached."""
    root = tmp_path / "demo.lance"
    root.mkdir()
    rows = pa.table({"doc_id": ["a"], "part": pa.array([0], pa.int32())})
    lance.write_dataset(rows, str(root / "rows.lance"))
    declared = Declared.model_validate({"identity": {"key_fields": ["doc_id", "part"]}})
    descriptor = DatasetDescriptor(id="demo", tables=discover_tables(root), declared=declared)
    return DatasetHandle(
        id="demo", path=root, uri=str(root), db=lancedb.connect(str(root)), descriptor=descriptor
    )


class TestSyncTableInfo:
    def test_write_drift_refreshes_cached_table_info(self, drift_handle: DatasetHandle) -> None:
        cached = drift_handle.descriptor.tables["rows"]
        ds = lance.dataset(str(drift_handle.path / "rows.lance"))
        lance.write_dataset(
            pa.table({"doc_id": ["b"], "part": pa.array([1], pa.int32())}),
            ds.uri,
            mode="append",
        )
        table_dataset(drift_handle, "rows")
        refreshed = drift_handle.descriptor.tables["rows"]
        assert refreshed.version > cached.version
        assert refreshed.row_count == 2

    def test_table_created_after_open_appears_in_descriptor(
        self, drift_handle: DatasetHandle
    ) -> None:
        assert "late" not in drift_handle.descriptor.tables
        lance.write_dataset(
            pa.table({"id": ["x"]}), str(drift_handle.path / "late.lance")
        )
        table_dataset(drift_handle, "late")
        assert drift_handle.descriptor.tables["late"].row_count == 1

    def test_no_drift_no_reintrospection(self, drift_handle: DatasetHandle) -> None:
        before = drift_handle.descriptor.tables["rows"]
        table_dataset(drift_handle, "rows")
        # same version observed → the cached object is untouched (identity check)
        assert drift_handle.descriptor.tables["rows"] is before
