"""Voice endpoints — table status + "find this voice elsewhere" similarity.

Thin HTTP layer over :func:`backend.voice.service.similar_voices`: it validates
the path-free query params (``doc_id`` is whitelisted before any service code
inlines it into a Lance filter literal) and pulls the optional voice-table
handles off ``StateDep``. Both handlers stay sync — every read is a blocking
Lance call, which the threadpool absorbs.

No ``from __future__ import annotations`` here: FastAPI introspects these
signatures at runtime, so the annotations stay real objects.
"""

from fastapi import APIRouter

from backend.deps import StateDep
from backend.media.blobs import valid_doc_id
from backend.schemas.voice import VoiceSimilarResponse, VoiceStatusResponse
from backend.voice.service import similar_voices

router = APIRouter(prefix="/api/voice", tags=["voice"])


@router.get("/status")
def voice_status(state: StateDep) -> VoiceStatusResponse:
    """Whether the voice tables exist + their row counts (no error when absent)."""
    if state.speaker_embeddings_tbl is None:
        return VoiceStatusResponse(built=False)
    return VoiceStatusResponse(
        built=True,
        turns=state.speaker_embeddings_tbl.count_rows(),
        speakers=state.speakers_tbl.count_rows() if state.speakers_tbl is not None else 0,
    )


@router.get("/similar")
def voice_similar(
    state: StateDep,
    doc_id: str,
    turn_id: int | None = None,
    speaker: str | None = None,
    t: float | None = None,
    n: int = 20,
    exclude_same_doc: bool = True,
) -> VoiceSimilarResponse:
    """Voice-ranked hits for exactly one anchor: ``turn_id`` | ``speaker`` | ``t``.

    The anchor embedding is read from Lance (no encoder at query time); ``n``
    is clamped to the service's cap. ``rerank`` is deliberately not offered —
    the cross-encoder scores transcript text, which says nothing about voice.
    """
    valid_doc_id(doc_id)
    return similar_voices(
        state.speaker_embeddings_tbl,
        state.speakers_tbl,
        state.chunks_ds,
        state.chunk_frames_tbl,
        doc_id=doc_id,
        turn_id=turn_id,
        speaker=speaker,
        t=t,
        n=n,
        exclude_same_doc=exclude_same_doc,
    )
