"""Minimal msgspec data model mirroring ``easytranscriber.data.datamodel``.

Vendored here so the ingest path can parse transcriber output with type-safe
``msgspec.json.decode`` without pulling in easytranscriber's heavy runtime
dependencies (torch, transformers, pyannote, ...). If you do have
``easytranscriber`` installed in the same environment, you can swap these
imports for the upstream ones — the field names and shapes match exactly.
"""

from __future__ import annotations

import msgspec


class WordSegment(msgspec.Struct):
    text: str
    start: float
    end: float
    score: float | None = None


class AudioChunk(msgspec.Struct):
    start: float
    end: float
    text: str | None = None
    duration: float | None = None
    audio_frames: int | None = None
    num_logits: int | None = None
    language: str | None = None
    language_prob: float | None = None
    id: str | int | None = None


class AlignmentSegment(msgspec.Struct):
    start: float
    end: float
    text: str
    words: list[WordSegment] = []
    id: str | int | None = None
    duration: float | None = None
    score: float | None = None


class SpeechSegment(msgspec.Struct):
    speech_id: str | int | None = None
    start: float | None = None
    end: float | None = None
    text: str | None = None
    text_spans: list[tuple[int, int]] | None = None
    chunks: list[AudioChunk] = []
    alignments: list[AlignmentSegment] = []
    duration: float | None = None
    audio_frames: int | None = None
    probs_path: str | None = None
    metadata: dict | None = None


class AudioMetadata(msgspec.Struct):
    audio_path: str
    sample_rate: int
    duration: float
    id: str | int | None = None
    speeches: list[SpeechSegment] | None = None
    metadata: dict | None = None
