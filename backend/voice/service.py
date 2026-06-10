"""Decoupled voice-similarity retrieval — anchor resolve → kNN → chunk join.

``similar_voices`` answers "where else does this voice speak?": it reads the
anchor voiceprint **from Lance** (no encoder runs at query time), kNN-ranks the
per-turn ``speaker_embeddings`` table by cosine distance, joins each matched
turn back to its max-overlap ``chunks`` row, and finishes with the shared
:func:`~backend.search.postprocess._postprocess_hits` so hits keep the uniform
search Hit shape (plus the voice fields). Exactly one anchor form is accepted:

* ``turn_id`` — that turn's embedding (one ``speaker_embeddings`` row);
* ``speaker`` — the speaker's duration-weighted centroid (``speakers`` row);
* ``t`` — the turn covering second ``t`` of the video.

``similar_voices_for_upload`` is the fourth, Lance-free anchor form: an
uploaded audio/video snippet is ffmpeg-decoded and embedded **in-process on the
CPU** (the lazily-loaded encoder the router passes in), then joins the same
:func:`rank_similar_turns` path, so upload hits are shaped exactly like the
GET's.

Like :mod:`backend.search.service`, this module takes plain Lance handles (no
FastAPI imports beyond the domain exceptions); the router wires it to the
request. A cross-encoder rerank is deliberately NOT supported — it reads
transcript text, which is meaningless for voice identity.
"""

from __future__ import annotations

import logging
import math
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from backend.core.exceptions import NotFoundError, ServiceUnavailableError, ValidationError
from backend.schemas.voice import VoiceAnchor, VoiceSimilarResponse

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

    from raudio.media.voiceprint import TurnBatchEncoder
from backend.search.constants import (
    _PAYLOAD_COLUMNS,
    _VECTOR_MAX_NPROBES,
    _VECTOR_NPROBES,
    _VECTOR_REFINE_FACTOR,
)
from backend.search.filters import _sql_quote
from backend.search.postprocess import _postprocess_hits

logger = logging.getLogger(__name__)

#: Hard cap on the result count — a turn→chunk join runs per hit.
_MAX_N = 100

#: speaker_embeddings columns a kNN hit needs (the join + voice fields).
_TURN_HIT_COLUMNS = ["doc_id", "turn_id", "speaker_label", "start", "end"]

#: Hard cap on an uploaded snippet's size — decode + embed run in-process.
_MAX_UPLOAD_BYTES = 25 * 1024 * 1024

#: Embed at most the first this-many seconds of an upload — the encoder was
#: trained on 5 s chunks, so longer audio adds CPU cost, not signal.
_UPLOAD_EMBED_CAP_S = 30.0

#: ffmpeg timeout for a ≤25 MB upload (diarize's 1800 s default is sized for
#: full-length videos, not snippets).
_UPLOAD_FFMPEG_TIMEOUT_S = 60.0


def _anchor_rows(table: Any, filter_expr: str) -> list[dict[str, Any]]:
    """Filtered scan of a voice table (embedding included) as Python rows."""
    return table.to_lance().to_table(filter=filter_expr).to_pylist()


def _resolve_turn_anchor(
    speaker_embeddings: Any, doc_id: str, turn_id: int
) -> tuple[list[float], VoiceAnchor]:
    rows = _anchor_rows(speaker_embeddings, f"doc_id = '{doc_id}' AND turn_id = {int(turn_id)}")
    if not rows:
        raise NotFoundError("anchor turn not found")
    return _turn_row_to_anchor(rows[0], doc_id)


def _resolve_time_anchor(
    speaker_embeddings: Any, doc_id: str, t: float
) -> tuple[list[float], VoiceAnchor]:
    rows = _anchor_rows(
        speaker_embeddings, f"doc_id = '{doc_id}' AND start <= {float(t)} AND end >= {float(t)}"
    )
    if not rows:
        raise NotFoundError("no speaker turn at that time")
    # Overlapped speech can stack several turns over one instant; the most
    # recently started one is the active speaker — and a deterministic pick.
    return _turn_row_to_anchor(max(rows, key=lambda r: float(r["start"])), doc_id)


def _turn_row_to_anchor(row: dict[str, Any], doc_id: str) -> tuple[list[float], VoiceAnchor]:
    anchor = VoiceAnchor(
        doc_id=doc_id,
        speaker_label=row["speaker_label"],
        turn_id=int(row["turn_id"]),
        turn_start=float(row["start"]),
        turn_end=float(row["end"]),
    )
    return row["embedding"], anchor


def _resolve_speaker_anchor(
    speakers: Any | None, doc_id: str, speaker: str
) -> tuple[list[float], VoiceAnchor]:
    if speakers is None:
        raise ServiceUnavailableError("speakers table not built yet — run `raudio build-speakers`")
    rows = _anchor_rows(
        speakers, f"doc_id = '{doc_id}' AND speaker_label = '{_sql_quote(speaker)}'"
    )
    if not rows:
        raise NotFoundError("anchor speaker not found")
    row = rows[0]
    return row["embedding"], VoiceAnchor(doc_id=doc_id, speaker_label=row["speaker_label"])


def _search_turns(
    speaker_embeddings: Any,
    vec: list[float],
    *,
    n: int,
    exclude_doc_id: str | None,
) -> list[dict[str, Any]]:
    """Cosine kNN over per-turn voiceprints (mirrors ``_vector_search``)."""
    try:
        qb = (
            speaker_embeddings.search(vec, vector_column_name="embedding")
            .distance_type("cosine")
            .minimum_nprobes(_VECTOR_NPROBES)
            .maximum_nprobes(_VECTOR_MAX_NPROBES)
            .refine_factor(_VECTOR_REFINE_FACTOR)
            .select([*_TURN_HIT_COLUMNS, "_distance"])
            .limit(n)
        )
        if exclude_doc_id is not None:
            # doc_id is whitelisted 16-char hex by the router — safe to inline.
            qb = qb.where(f"doc_id != '{exclude_doc_id}'", prefilter=True)
        return qb.to_list()
    except Exception as e:
        logger.warning("voice search failed", exc_info=True)
        raise ValidationError("voice search failed") from e


def _chunk_for_turn(
    chunks_ds: Any, doc_id: str, turn_start: float, turn_end: float
) -> dict[str, Any] | None:
    """The max-overlap ``chunks`` row for a turn span (None if no chunk overlaps)."""
    flt = f"doc_id = '{_sql_quote(doc_id)}' AND start < {turn_end} AND end > {turn_start}"
    rows = chunks_ds.to_table(columns=_PAYLOAD_COLUMNS, filter=flt).to_pylist()
    if not rows:
        return None
    return max(
        rows,
        key=lambda r: min(float(r["end"]), turn_end) - max(float(r["start"]), turn_start),
    )


def similar_voices(
    speaker_embeddings,  # lancedb Table for the per-turn voiceprints, or None if unbuilt
    speakers,  # lancedb Table for the per-speaker centroids, or None if unbuilt
    chunks_ds,  # lance.LanceDataset over chunks (the shared startup handle)
    chunk_frames,  # lancedb Table for the caption attach, or None
    *,
    doc_id: str,
    turn_id: int | None,
    speaker: str | None,
    t: float | None,
    n: int,
    exclude_same_doc: bool,
) -> VoiceSimilarResponse:
    """Rank speaker turns by voice similarity to one anchor; return chunk hits.

    Hits keep the kNN's voice ranking: each is the matched turn's max-overlap
    chunk row augmented with ``speaker_label`` / ``turn_id`` / ``turn_start`` /
    ``turn_end`` / ``_distance`` / ``turn_score``. Turns whose span overlaps no
    chunk (diarized speech the ASR produced nothing for) are dropped, so fewer
    than ``n`` hits can come back.
    """
    if speaker_embeddings is None:
        raise ServiceUnavailableError(
            "voice embeddings not built yet — run `raudio embed-speaker-turns`"
        )
    anchors = [a for a in (turn_id, speaker, t) if a is not None]
    if len(anchors) != 1:
        raise ValidationError("provide exactly one anchor: turn_id, speaker, or t")
    if t is not None and not math.isfinite(t):
        raise ValidationError("t must be a finite number of seconds")

    if turn_id is not None:
        vec, anchor = _resolve_turn_anchor(speaker_embeddings, doc_id, turn_id)
    elif speaker is not None:
        vec, anchor = _resolve_speaker_anchor(speakers, doc_id, speaker)
    elif t is not None:
        vec, anchor = _resolve_time_anchor(speaker_embeddings, doc_id, t)
    else:
        # Unreachable — the exactly-one check above guarantees a branch.
        raise AssertionError("unhandled anchor form")

    hits = rank_similar_turns(
        speaker_embeddings,
        chunks_ds,
        chunk_frames,
        vec,
        n=n,
        exclude_doc_id=doc_id if exclude_same_doc else None,
    )
    return VoiceSimilarResponse(query=anchor, hits=hits)


def rank_similar_turns(
    speaker_embeddings: Any,
    chunks_ds: Any,
    chunk_frames: Any,
    vec: list[float],
    *,
    n: int,
    exclude_doc_id: str | None,
) -> list[dict[str, Any]]:
    """The shared post-anchor path: clamp ``n`` → kNN → turn→chunk join → postprocess.

    Every anchor form (the GET's Lance-resolved voiceprints and the POST's
    uploaded-snippet embedding) funnels through here, so hits keep one shape.
    """
    n = max(1, min(n, _MAX_N))
    turn_hits = _search_turns(speaker_embeddings, vec, n=n, exclude_doc_id=exclude_doc_id)

    hits: list[dict[str, Any]] = []
    for th in turn_hits:
        turn_start, turn_end = float(th["start"]), float(th["end"])
        chunk = _chunk_for_turn(chunks_ds, th["doc_id"], turn_start, turn_end)
        if chunk is None:
            continue
        distance = float(th["_distance"])
        chunk["speaker_label"] = th["speaker_label"]
        chunk["turn_id"] = int(th["turn_id"])
        chunk["turn_start"] = turn_start
        chunk["turn_end"] = turn_end
        chunk["_distance"] = distance
        chunk["turn_score"] = 1.0 - distance
        hits.append(chunk)

    return _postprocess_hits(hits, chunk_frames)


def _decode_upload_wav(file_bytes: bytes) -> np.ndarray:
    """ffmpeg-decode uploaded bytes → 16 kHz mono float32 samples in [-1, 1].

    Mirrors :meth:`raudio.media.diarize.Diarizer.diarize`'s temp-WAV flow: the
    bytes go to a temp file (ffmpeg sniffs the container itself — wav/mp3/mp4/
    m4a/…, no trust in the filename), get transcoded to the canonical 16 kHz
    mono PCM16 WAV, and are loaded with the encoder's own loader. Undecodable
    input maps to a 400 — it's the uploader's bytes, not our state.
    """
    from raudio.media.diarize import _extract_wav_16k_mono
    from raudio.media.voiceprint import load_wav_16k_mono

    with tempfile.TemporaryDirectory(prefix="raudio-voice-upload-") as tmp:
        src = Path(tmp) / "upload.bin"
        src.write_bytes(file_bytes)
        wav = Path(tmp) / "audio_16k_mono.wav"
        try:
            _extract_wav_16k_mono(src, wav, timeout=_UPLOAD_FFMPEG_TIMEOUT_S)
        except RuntimeError as e:
            logger.info("upload decode failed: %s", e)
            raise ValidationError("could not decode the upload as audio") from e
        return load_wav_16k_mono(wav)


def similar_voices_for_upload(
    speaker_embeddings,  # lancedb Table for the per-turn voiceprints, or None if unbuilt
    chunks_ds,  # lance.LanceDataset over chunks (the shared startup handle)
    chunk_frames,  # lancedb Table for the caption attach, or None
    get_encoder: Callable[[], TurnBatchEncoder],
    *,
    file_bytes: bytes,
    n: int,
) -> VoiceSimilarResponse:
    """Rank speaker turns by voice similarity to an uploaded audio/video snippet.

    The snippet is decoded via ffmpeg, capped to its first
    :data:`_UPLOAD_EMBED_CAP_S` seconds, embedded whole (one "turn") with the
    lazily-loaded CPU encoder, L2-normalized, and ranked through the same
    :func:`rank_similar_turns` path as the GET. The query has no Lance-side
    anchor, so the response's ``query`` is an all-``None`` :class:`VoiceAnchor`;
    ``exclude_same_doc`` does not apply (an upload belongs to no doc).
    """
    import numpy as np

    from raudio.media.diarize import TARGET_SAMPLE_RATE
    from raudio.media.voiceprint import MIN_TURN_DURATION_S, l2_normalize

    if speaker_embeddings is None:
        raise ServiceUnavailableError(
            "voice embeddings not built yet — run `raudio embed-speaker-turns`"
        )
    if len(file_bytes) > _MAX_UPLOAD_BYTES:
        raise ValidationError("upload too large (max 25 MB)")

    wav = _decode_upload_wav(file_bytes)
    if wav.shape[0] < round(MIN_TURN_DURATION_S * TARGET_SAMPLE_RATE):
        raise ValidationError(f"snippet too short for a voiceprint (min {MIN_TURN_DURATION_S} s)")
    wav = wav[: round(_UPLOAD_EMBED_CAP_S * TARGET_SAMPLE_RATE)]

    raw = get_encoder().embed_batch([wav])
    unit = l2_normalize(raw)[0]
    # Degenerate audio (e.g. pure silence) can yield a zero or non-finite raw
    # embedding — l2_normalize's norm floor keeps a zero row zero rather than
    # NaN, hence the norm check. Either would make the cosine kNN undefined,
    # so reject it as the uploader's-input problem it is.
    if not np.all(np.isfinite(unit)) or float(np.linalg.norm(unit)) < 0.5:
        raise ValidationError("could not extract a voiceprint from the upload")
    vec: list[float] = unit.tolist()

    hits = rank_similar_turns(
        speaker_embeddings, chunks_ds, chunk_frames, vec, n=n, exclude_doc_id=None
    )
    return VoiceSimilarResponse(query=VoiceAnchor(), hits=hits)
