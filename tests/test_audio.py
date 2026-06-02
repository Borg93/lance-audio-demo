"""Pure URI/MIME helper tests (docs/TESTING.md) — deterministic, no I/O except
one tmp file for the file:// path. A slash/precedence regression here silently
breaks media fetch downstream without raising.

Run:  uv run pytest tests/test_audio.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from raudio.ingest.audio import compose_media_uri, guess_mime


class TestComposeMediaUri:
    def test_base_uri_without_trailing_slash_adds_exactly_one(self):
        assert compose_media_uri(audio_path="x/T1.mp4", source_path=None,
                                 base_uri="hf://b/v") == "hf://b/v/T1.mp4"

    def test_base_uri_with_trailing_slash_no_double(self):
        assert compose_media_uri(audio_path="x/T1.mp4", source_path=None,
                                 base_uri="hf://b/v/") == "hf://b/v/T1.mp4"

    def test_base_uri_takes_precedence_over_source_path(self):
        assert compose_media_uri(audio_path="T1.mp4", source_path=Path("/abs/T1.mp4"),
                                 base_uri="s3://b/") == "s3://b/T1.mp4"

    def test_source_path_becomes_absolute_file_uri(self, tmp_path):
        f = tmp_path / "v.mp4"
        f.write_bytes(b"x")
        out = compose_media_uri(audio_path="v.mp4", source_path=f, base_uri=None)
        assert out is not None
        assert out == f.resolve().as_uri() and out.startswith("file://")

    def test_none_when_no_base_and_no_source(self):
        assert compose_media_uri(audio_path="v.mp4", source_path=None, base_uri=None) is None

    def test_only_the_basename_is_used_as_the_key(self):
        assert compose_media_uri(audio_path="deep/nested/T1.mp4", source_path=None,
                                 base_uri="hf://b/") == "hf://b/T1.mp4"


class TestGuessMime:
    @pytest.mark.parametrize("name,expected", [("x.mp4", "video/mp4"), ("x.mp3", "audio/mpeg")])
    def test_known_extensions(self, name, expected):
        assert guess_mime(name) == expected

    def test_unknown_extension_falls_back_never_none(self):
        assert guess_mime("x.zzz") == "application/octet-stream"
