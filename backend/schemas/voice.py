"""Response models for the voice-similarity endpoints (``/api/voice``)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class VoiceStatusResponse(BaseModel):
    """Whether the voice tables exist, plus their row counts.

    ``built`` is ``False`` (with zero counts) until ``embed-speaker-turns`` has
    produced ``speaker_embeddings.lance``; ``speakers`` stays 0 until
    ``build-speakers`` has produced the per-speaker centroid table.
    """

    built: bool
    turns: int = 0
    speakers: int = 0


class VoiceAnchor(BaseModel):
    """The resolved anchor voiceprint a ``/similar`` query ranked against.

    ``speaker_label`` is pyannote's per-video local label (``SPEAKER_00`` …),
    stable only within ``doc_id``. The turn fields are ``None`` for the
    per-speaker-centroid anchor form (``doc_id`` + ``speaker``), which ranks
    against the speaker's duration-weighted mean voiceprint, not one turn.
    """

    doc_id: str
    speaker_label: str
    turn_id: int | None = None
    turn_start: float | None = None
    turn_end: float | None = None


class VoiceSimilarResponse(BaseModel):
    """Voice-ranked hits in the uniform search Hit shape.

    Each hit is the max-overlap ``chunks`` row for a matched speaker turn,
    augmented with ``speaker_label`` / ``turn_id`` / ``turn_start`` /
    ``turn_end`` / ``_distance`` (cosine) / ``turn_score`` (1 − distance).
    Deliberately ``dict`` hits, mirroring ``/api/search`` — the frontend
    renders one card type for everything.
    """

    query: VoiceAnchor
    hits: list[dict[str, Any]] = Field(default_factory=list)
