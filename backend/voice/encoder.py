"""Lazy voice-encoder accessor bound to :class:`~backend.state.AppState`.

Mirrors :mod:`backend.clients`: return the cached encoder if present, else
construct it on first use and write it back onto the same ``AppState``, mapping
any failure to a ``ServiceUnavailableError`` (HTTP 503). Unlike the vLLM
clients this loads a model **in-process** (pyannote's WeSpeaker encoder, ~30 s
of import + weights), so first-load is guarded by ``state.voice_encoder_lock``
— sync handlers run in a threadpool, and two concurrent first uploads must not
both construct it. The ``raudio.media.voiceprint`` import is deferred inside
the function: it keeps upload-free deployments free of torch/pyannote, and it
is the seam tests monkeypatch (tests usually just pre-set ``state.voice_encoder``
to a :class:`~raudio.media.voiceprint.TurnBatchEncoder` fake instead).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from backend.core.exceptions import ServiceUnavailableError

if TYPE_CHECKING:
    from backend.state import AppState
    from raudio.media.voiceprint import VoiceEncoder

logger = logging.getLogger(__name__)


def ensure_voice_encoder(state: AppState) -> VoiceEncoder:
    """Return the app's voice encoder, loading it on first use (503 on failure)."""
    if state.voice_encoder is not None:
        return state.voice_encoder
    with state.voice_encoder_lock:
        if state.voice_encoder is not None:  # lost the race; the winner loaded it
            return state.voice_encoder
        try:
            from raudio.media.voiceprint import VoiceEncoder

            # CPU by design: the upload path must never contend for the GPUs.
            encoder = VoiceEncoder(device="cpu")
        except Exception as e:
            logger.exception("failed to load voice encoder")
            raise ServiceUnavailableError("voice encoder unavailable") from e
        state.voice_encoder = encoder
        return encoder
