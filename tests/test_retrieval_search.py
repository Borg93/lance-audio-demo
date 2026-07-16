"""Alignment JSONB decoding + word-match extraction (pure, no Lance/vLLM).

``parse_alignments_json`` is a boundary decoder — Lance JSONB hands back either a
raw string or an already-decoded value, and every malformed/odd shape must
degrade to ``[]`` so the annotated return type holds. ``iter_matching_words``
returns typed ``WordMatch`` objects the CLI reads by attribute.
"""

from __future__ import annotations

import json

from rmedia.retrieval.search import WordMatch, iter_matching_words, parse_alignments_json


class TestParseAlignmentsJson:
    def test_none_is_empty(self) -> None:
        assert parse_alignments_json(None) == []

    def test_empty_string_is_empty(self) -> None:
        assert parse_alignments_json("") == []

    def test_valid_json_string_is_decoded(self) -> None:
        assert parse_alignments_json('[{"text": "hi"}]') == [{"text": "hi"}]

    def test_already_decoded_list_passes_through(self) -> None:
        rows = [{"text": "hi"}]
        assert parse_alignments_json(rows) is rows

    def test_malformed_json_string_is_empty(self) -> None:
        assert parse_alignments_json("{not valid json") == []

    def test_non_list_json_is_empty(self) -> None:
        # A JSON object (not a list) decodes fine but isn't the alignment shape.
        assert parse_alignments_json('{"a": 1}') == []


class TestIterMatchingWords:
    @staticmethod
    def _row(words: list[dict[str, object]]) -> dict[str, object]:
        return {"alignments_json": json.dumps([{"words": words}])}

    def test_returns_typed_wordmatch(self) -> None:
        row = self._row([{"text": "Sverige", "start": 1.0, "end": 1.5, "score": 0.9}])
        out = iter_matching_words(row, ["sverige"])
        assert len(out) == 1
        assert isinstance(out[0], WordMatch)
        assert (out[0].text, out[0].start, out[0].end, out[0].score) == ("Sverige", 1.0, 1.5, 0.9)

    def test_match_is_case_insensitive_and_punctuation_stripped(self) -> None:
        # "Sverige." matches the term "sverige"; the raw (punctuated) text is kept.
        row = self._row([{"text": "Sverige.", "start": 1.0, "end": 1.5, "score": 0.9}])
        out = iter_matching_words(row, ["sverige"])
        assert out and out[0].text == "Sverige."

    def test_missing_score_defaults_to_none(self) -> None:
        row = self._row([{"text": "Sverige", "start": 1.0, "end": 1.5}])
        out = iter_matching_words(row, ["sverige"])
        assert out and out[0].score is None

    def test_no_terms_is_empty(self) -> None:
        row = self._row([{"text": "Sverige", "start": 1.0, "end": 1.5, "score": 0.9}])
        assert iter_matching_words(row, []) == []

    def test_no_match_is_empty(self) -> None:
        row = self._row([{"text": "Norge", "start": 0.0, "end": 1.0, "score": 0.5}])
        assert iter_matching_words(row, ["sverige"]) == []
