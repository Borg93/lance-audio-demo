"""Voice endpoints — table status + "find this voice elsewhere" similarity.

Thin HTTP layer over :mod:`backend.voice.service`: it validates the path-free
query params (``doc_id`` is whitelisted before any service code inlines it
into a Lance filter literal) and pulls the optional voice-table handles off
``StateDep``. The GET/status handlers stay sync — every read is a blocking
Lance call, which the threadpool absorbs; the upload POST is async to await
the multipart body (mirrors ``POST /api/search``), then offloads the blocking
decode + CPU embed + Lance reads to the threadpool.

No ``from __future__ import annotations`` here: FastAPI introspects these
signatures at runtime, so the annotations stay real objects.
"""

from typing import Annotated

from fastapi import APIRouter, File, UploadFile
from starlette.concurrency import run_in_threadpool

from backend.deps import StateDep
from backend.media.blobs import valid_doc_id
from backend.schemas.voice import VoiceSimilarResponse, VoiceStatusResponse
from backend.voice.encoder import ensure_voice_encoder
from backend.voice.service import _MAX_UPLOAD_BYTES, similar_voices, similar_voices_for_upload

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


@router.post("/similar")
async def voice_similar_upload(
    state: StateDep,
    file: Annotated[UploadFile, File()],
    n: int = 20,
) -> VoiceSimilarResponse:
    """Voice-ranked hits for an uploaded snippet (any container ffmpeg decodes).

    ``n`` is a query param like the GET's (the multipart body carries only
    ``file``); ``exclude_same_doc`` is deliberately not a param — an upload
    belongs to no doc. The body read stops one byte past the size cap, so an
    oversize upload 400s (in the service) without ever being buffered whole.
    The encoder getter is passed as a thunk so the model only loads if the
    snippet survives the size/decode/duration guards.
    """
    file_bytes = await file.read(_MAX_UPLOAD_BYTES + 1)
    return await run_in_threadpool(
        similar_voices_for_upload,
        state.speaker_embeddings_tbl,
        state.chunks_ds,
        state.chunk_frames_tbl,
        lambda: ensure_voice_encoder(state),
        file_bytes=file_bytes,
        n=n,
    )
