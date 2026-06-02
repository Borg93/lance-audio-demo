"""Write-path tests — the highest-value gap from docs/TESTING.md.

These are pure, deterministic, no-GPU/no-network tests of the transform that
turns transcriber JSON into Lance rows. A bug here writes wrong-but-parseable
values to every chunk and never raises, so the dataset-gated smoke test can't
catch it. Plus one real Lance ingest round-trip (lance + lancedb only).

Run:  uv run pytest tests/test_ingest.py -v
"""

from __future__ import annotations

import pytest

from raudio.ingest.ingest import (
    _doc_id,
    _metadata_for,
    _pick_alignments,
    flatten_chunks,
    ingest_many,
    load_metadata_csv,
)
from raudio.model.datamodel import (
    AlignmentSegment,
    AudioChunk,
    AudioMetadata,
    SpeechSegment,
    WordSegment,
)
from raudio.retrieval.search import nearest_chunks


def _doc(
    *,
    audio_path: str = "input/T0001.mp4",
    sample_rate: int = 16000,
    duration: float = 30.0,
    speeches: list[SpeechSegment] | None = None,
) -> AudioMetadata:
    """A minimal AudioMetadata; override fields via keyword."""
    return AudioMetadata(
        audio_path=audio_path, sample_rate=sample_rate, duration=duration, speeches=speeches
    )


class TestPickAlignments:
    def _align(self, start: float, end: float) -> AlignmentSegment:
        return AlignmentSegment(start=start, end=end, text="w")

    def test_closed_interval_keeps_exact_bounds(self):
        # [start, end] is inclusive on BOTH ends — an alignment flush against
        # either boundary is kept.
        out = _pick_alignments([self._align(0.0, 30.0)], 0.0, 30.0)
        assert len(out) == 1

    @pytest.mark.parametrize("a_start,a_end", [(-0.1, 5.0), (10.0, 30.1)])
    def test_outside_either_bound_is_dropped(self, a_start, a_end):
        out = _pick_alignments([self._align(a_start, a_end)], 0.0, 30.0)
        assert out == []

    def test_words_are_copied_with_float_cast_and_optional_score(self):
        seg = AlignmentSegment(
            start=1.0, end=2.0, text="hi there",
            words=[WordSegment(text="hi", start=1, end=1.5, score=0.9),
                   WordSegment(text="there", start=1.5, end=2)],  # score omitted → None
        )
        [row] = _pick_alignments([seg], 0.0, 30.0)
        assert [w["text"] for w in row["words"]] == ["hi", "there"]
        assert row["words"][0]["start"] == 1.0 and isinstance(row["words"][0]["start"], float)
        assert row["words"][1]["score"] is None


class TestFlattenChunks:
    def test_language_fallback_per_chunk_then_doc(self):
        doc = _doc(speeches=[SpeechSegment(speech_id=0, chunks=[
            AudioChunk(start=0, end=5, text="a", language="en"),  # explicit
            AudioChunk(start=5, end=10, text="b"),                # falls back
        ])])
        rows = list(flatten_chunks(doc, doc_language="sv"))
        assert [r["language"] for r in rows] == ["en", "sv"]

    def test_speech_id_none_becomes_zero_and_str_is_int_cast(self):
        doc = _doc(speeches=[
            SpeechSegment(speech_id=None, chunks=[AudioChunk(start=0, end=1, text="x")]),
            SpeechSegment(speech_id="3", chunks=[AudioChunk(start=0, end=1, text="y")]),
        ])
        rows = list(flatten_chunks(doc))
        assert [r["speech_id"] for r in rows] == [0, 3]
        assert all(isinstance(r["speech_id"], int) for r in rows)

    def test_chunk_id_enumerates_per_speech(self):
        doc = _doc(speeches=[
            SpeechSegment(speech_id=0, chunks=[AudioChunk(start=0, end=1, text="a"),
                                               AudioChunk(start=1, end=2, text="b")]),
            SpeechSegment(speech_id=1, chunks=[AudioChunk(start=0, end=1, text="c")]),
        ])
        rows = list(flatten_chunks(doc))
        assert [(r["speech_id"], r["chunk_id"]) for r in rows] == [(0, 0), (0, 1), (1, 0)]

    def test_none_text_coerces_to_empty_string(self):
        # CHUNK_SCHEMA.text is non-nullable; None must become "".
        doc = _doc(speeches=[SpeechSegment(speech_id=0, chunks=[AudioChunk(start=0, end=1)])])
        [row] = list(flatten_chunks(doc))
        assert row["text"] == ""

    def test_no_speeches_yields_no_rows(self):
        assert list(flatten_chunks(_doc(speeches=None))) == []

    def test_metadata_join_by_bildid_stem(self):
        doc = _doc(audio_path="input/T0001.mp4",
                   speeches=[SpeechSegment(speech_id=0, chunks=[AudioChunk(start=0, end=1, text="x")])])
        idx = {"T0001": {"referenskod": "SE/RA/1", "namn": "Möte", "bildid": "T0001", "extraid": "E1"}}
        [row] = list(flatten_chunks(doc, metadata_index=idx))
        assert (row["referenskod"], row["namn"], row["extraid"]) == ("SE/RA/1", "Möte", "E1")
        # No match → all-None metadata
        [row2] = list(flatten_chunks(_doc(audio_path="input/OTHER.mp4",
                     speeches=[SpeechSegment(speech_id=0, chunks=[AudioChunk(start=0, end=1, text="x")])]),
                     metadata_index=idx))
        assert row2["referenskod"] is None and row2["namn"] is None


class TestMetadataCsv:
    def test_parses_semicolon_skips_blank_bildid_and_strips(self, tmp_path):
        csv = tmp_path / "meta.csv"
        csv.write_text(
            "referenskod;namn;extraid;bildid\n"
            "SE/RA/1; Möte ;E1;T0001\n"   # whitespace around namn
            "SE/RA/2;X;E2;\n"             # blank bildid → skipped
            , encoding="utf-8",
        )
        idx = load_metadata_csv(csv)
        assert set(idx) == {"T0001"}
        assert idx["T0001"]["namn"] == "Möte"  # stripped

    def test_metadata_for_keys_on_stem_and_misses_are_all_none(self):
        idx = {"T0001": {"referenskod": "r", "namn": "n", "bildid": "T0001", "extraid": "e"}}
        assert _metadata_for("input/sv/T0001.mp4", idx)["referenskod"] == "r"  # stem match
        miss = _metadata_for("input/sv/NOPE.mp4", idx)
        assert miss == {"referenskod": None, "namn": None, "bildid": None, "extraid": None}


class TestDocId:
    def test_stable_deterministic_16_hex(self):
        a = _doc_id("input/T0001.mp4")
        assert a == _doc_id("input/T0001.mp4")          # deterministic
        assert a != _doc_id("input/T0002.mp4")          # path-sensitive
        assert len(a) == 16 and all(c in "0123456789abcdef" for c in a)


class TestIngestRoundTrip:
    """The single highest-value integration test: real Lance write + FTS query,
    no GPU/vLLM — just lance + lancedb."""

    def test_ingest_then_fts_query_back(self, tmp_path):
        db = tmp_path / "rt.lance"
        doc = _doc(audio_path="input/roundtrip.mp4", speeches=[SpeechSegment(
            speech_id=0,
            chunks=[AudioChunk(start=0.0, end=5.0, text="the quick brown fox jumps")],
        )])
        ingest_many(db, [doc])  # writes chunks table + FTS index, no documents (no audio_root)
        hits = nearest_chunks(db, "brown", limit=5)
        assert hits, "FTS over the freshly-ingested chunk should find 'brown'"
        assert "quick brown fox" in hits[0]["text"]
