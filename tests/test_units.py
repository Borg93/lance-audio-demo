"""Pure-function unit tests — no Lance dataset or vLLM server required.

Covers the small, self-contained logic that the rest of the system leans on:
timestamp formatting, HTTP Range parsing, doc-id validation, SQL-filter
composition, reciprocal-rank fusion, and alignment JSON decoding. These run
everywhere (unlike the dataset-gated smoke tests in ``test_backend_smoke.py``).
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import HTTPException

from common.lancekit.descriptor import Declared
from common.lancekit.keys import validate_doc_key
from rmedia.modalities.av.frames import sample_times
from rmedia.retrieval.search import extract_query_terms, parse_alignments_json, timecode
from search.services.filters import build_where_clause
from search.services.postprocess import rrf_fuse
from viewer.api.v1.endpoints.media import IGNORE_RANGE, parse_range


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
    def test_unhonored_range_is_ignored_not_416(self, header: str) -> None:
        # RFC 9110 §14.2: an unknown unit / malformed / unsupported form must be
        # ignored (serve 200), never answered with 416.
        assert parse_range(header, total=1000) == IGNORE_RANGE

    def test_multi_range_is_ignored(self) -> None:
        assert parse_range("bytes=0-10,20-30", total=1000) == IGNORE_RANGE

    def test_start_beyond_total_returns_none(self) -> None:
        assert parse_range("bytes=2000-3000", total=1000) is None


class TestBuildWhereClause:
    FILTERABLE = ("language", "speaker_name")

    def test_no_filters_is_none(self) -> None:
        assert build_where_clause(filters={}, filterable=self.FILTERABLE) is None

    def test_single_equality(self) -> None:
        clause = build_where_clause(filters={"language": "sv"}, filterable=self.FILTERABLE)
        assert clause == "language = 'sv'"

    def test_multiple_joined_with_and(self) -> None:
        clause = build_where_clause(
            filters={"language": "en", "speaker_name": "Palme"}, filterable=self.FILTERABLE
        )
        assert clause == "language = 'en' AND speaker_name = 'Palme'"

    def test_single_quote_is_escaped(self) -> None:
        # A lone apostrophe must be doubled so it can't break out of the literal.
        clause = build_where_clause(filters={"speaker_name": "O'Brien"}, filterable=self.FILTERABLE)
        assert clause == "speaker_name = 'O''Brien'"


class TestRrfFuse:
    def test_dedupes_and_orders_by_summed_score(self) -> None:
        # Item A is rank 0 in list 1 and rank 0 in list 2 → highest fused score.
        a = {"doc_id": "d1", "chunk_id": 0, "text": "a"}
        b = {"doc_id": "d1", "chunk_id": 1, "text": "b"}
        c = {"doc_id": "d2", "chunk_id": 0, "text": "c"}
        fused = rrf_fuse([[a, b], [a, c]], key_fields=["doc_id", "chunk_id"])
        keys = [(h["doc_id"], h["chunk_id"]) for h in fused]
        assert keys[0] == ("d1", 0)  # appears top of both lists
        assert len(fused) == 3  # a deduped, b and c kept

    def test_empty_input(self) -> None:
        assert rrf_fuse([], key_fields=["doc_id"]) == []


class TestExtractQueryTerms:
    def test_strips_operators_and_lowercases(self) -> None:
        assert extract_query_terms("Klimat AND Miljö") == ["klimat", "miljö"]

    def test_empty(self) -> None:
        assert extract_query_terms("") == []


HEX_DECLARED = Declared.model_validate(
    {"identity": {"key_fields": ["doc_id"], "doc_key_pattern": "^[a-f0-9]{16}$"}}
)


class TestValidDocId:
    def test_accepts_16_lowercase_hex(self) -> None:
        assert validate_doc_key(HEX_DECLARED, "0123456789abcdef") == "0123456789abcdef"

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
            validate_doc_key(HEX_DECLARED, bad)
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
