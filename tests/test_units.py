"""Pure-function unit tests — no Lance dataset or vLLM server required.

Covers the small, self-contained logic that the rest of the system leans on:
timestamp formatting, HTTP Range parsing, doc-id validation, SQL-filter
composition, reciprocal-rank fusion, and alignment JSON decoding. These run
everywhere (unlike the dataset-gated smoke tests in ``test_backend_smoke.py``).
"""

from __future__ import annotations

from typing import Any

import pytest
from backend.media.blobs import parse_range, valid_doc_id
from backend.search.service import _build_where_clause, _rrf_fuse
from fastapi import HTTPException

from raudio.media.frames import sample_times
from raudio.retrieval.search import extract_query_terms, parse_alignments_json, timecode


class TestTimecode:
    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [(0, "00:00"), (5, "00:05"), (65, "01:05"), (3661, "01:01:01")],
    )
    def test_plain(self, seconds: float, expected: str) -> None:
        assert timecode(seconds) == expected

    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [(0.5, "00:00.500"), (65.123, "01:05.123"), (3661.4567, "01:01:01.457")],
    )
    def test_millis(self, seconds: float, expected: str) -> None:
        assert timecode(seconds, millis=True) == expected


class TestParseRange:
    def test_closed_range(self) -> None:
        assert parse_range("bytes=0-99", total=1000) == (0, 99)

    def test_open_ended_clamps_to_total(self) -> None:
        assert parse_range("bytes=100-", total=1000) == (100, 999)

    def test_suffix_range_counts_from_end(self) -> None:
        assert parse_range("bytes=-500", total=1000) == (500, 999)

    def test_end_past_total_is_clamped(self) -> None:
        assert parse_range("bytes=0-5000", total=1000) == (0, 999)

    @pytest.mark.parametrize("header", ["", "bytes=", "kbytes=0-1", "bytes=abc-def"])
    def test_malformed_returns_none(self, header: str) -> None:
        assert parse_range(header, total=1000) is None

    def test_start_beyond_total_returns_none(self) -> None:
        assert parse_range("bytes=2000-3000", total=1000) is None


class TestBuildWhereClause:
    def test_no_filters_is_none(self) -> None:
        assert _build_where_clause(language=None, namn=None, referenskod=None, extraid=None) is None

    def test_single_equality(self) -> None:
        assert (
            _build_where_clause(language="sv", namn=None, referenskod=None, extraid=None)
            == "language = 'sv'"
        )

    def test_multiple_joined_with_and(self) -> None:
        clause = _build_where_clause(language="en", namn="Palme", referenskod=None, extraid=None)
        assert clause == "language = 'en' AND namn LIKE '%Palme%'"

    def test_single_quote_is_escaped(self) -> None:
        # A lone apostrophe must be doubled so it can't break out of the literal.
        clause = _build_where_clause(language=None, namn="O'Brien", referenskod=None, extraid=None)
        assert clause == "namn LIKE '%O''Brien%'"


class TestRrfFuse:
    def test_dedupes_and_orders_by_summed_score(self) -> None:
        # Item A is rank 0 in list 1 and rank 0 in list 2 → highest fused score.
        a = {"doc_id": "d1", "chunk_id": 0, "text": "a"}
        b = {"doc_id": "d1", "chunk_id": 1, "text": "b"}
        c = {"doc_id": "d2", "chunk_id": 0, "text": "c"}
        fused = _rrf_fuse([[a, b], [a, c]])
        keys = [(h["doc_id"], h["chunk_id"]) for h in fused]
        assert keys[0] == ("d1", 0)  # appears top of both lists
        assert len(fused) == 3  # a deduped, b and c kept

    def test_empty_input(self) -> None:
        assert _rrf_fuse([]) == []


class TestExtractQueryTerms:
    def test_strips_operators_and_lowercases(self) -> None:
        assert extract_query_terms("Klimat AND Miljö") == ["klimat", "miljö"]

    def test_empty(self) -> None:
        assert extract_query_terms("") == []


class TestValidDocId:
    def test_accepts_16_lowercase_hex(self) -> None:
        assert valid_doc_id("0123456789abcdef") is None

    @pytest.mark.parametrize(
        "bad",
        [
            "",  # empty
            "0123456789abcde",  # 15 chars
            "0123456789abcdef0",  # 17 chars
            "0123456789ABCDEF",  # uppercase rejected
            "0123456789abcdeg",  # non-hex char
            "not-a-valid-id",
        ],
    )
    def test_rejects_non_16_hex(self, bad: str) -> None:
        with pytest.raises(HTTPException) as exc:
            valid_doc_id(bad)
        assert exc.value.status_code == 400


class TestParseAlignmentsJson:
    def test_decodes_json_string_to_list(self) -> None:
        assert parse_alignments_json('[{"a": 1}]') == [{"a": 1}]

    def test_already_decoded_list_passes_through(self) -> None:
        assert parse_alignments_json([{"a": 1}]) == [{"a": 1}]

    @pytest.mark.parametrize(
        "bad",
        [None, "", "not json", '{"a": 1}', 42],  # null / empty / malformed / non-list / scalar
    )
    # `Any`, not `object`: the parametrize values are deliberately OUTSIDE the
    # declared input union — the test pins the defensive empty-list fallback.
    def test_non_list_or_malformed_is_empty(self, bad: Any) -> None:
        assert parse_alignments_json(bad) == []


class TestSampleTimes:
    def test_single_frame_when_no_interval(self) -> None:
        assert sample_times(5.0, 30.0, 0.0) == [5.0]

    def test_interval_samples_across_chunk(self) -> None:
        assert sample_times(0.0, 10.0, 4.0) == [0.0, 4.0, 8.0]

    def test_never_steps_past_end(self) -> None:
        assert sample_times(0.0, 5.0, 2.0) == [0.0, 2.0, 4.0]

    def test_degenerate_end_le_start_returns_start(self) -> None:
        assert sample_times(5.0, 5.0, 2.0) == [5.0]
