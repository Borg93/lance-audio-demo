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

Like :mod:`backend.search.service`, this module takes plain Lance handles (no
FastAPI imports beyond the domain exceptions); the router wires it to the
request. A cross-encoder rerank is deliberately NOT supported — it reads
transcript text, which is meaningless for voice identity.
"""

from __future__ import annotations

import logging
from typing import Any

from backend.core.exceptions import NotFoundError, ServiceUnavailableError, ValidationError
from backend.schemas.voice import VoiceAnchor, VoiceSimilarResponse
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

    if turn_id is not None:
        vec, anchor = _resolve_turn_anchor(speaker_embeddings, doc_id, turn_id)
    elif speaker is not None:
        vec, anchor = _resolve_speaker_anchor(speakers, doc_id, speaker)
    elif t is not None:
        vec, anchor = _resolve_time_anchor(speaker_embeddings, doc_id, t)
    else:
        # Unreachable — the exactly-one check above guarantees a branch.
        raise AssertionError("unhandled anchor form")

    n = max(1, min(n, _MAX_N))
    turn_hits = _search_turns(
        speaker_embeddings,
        vec,
        n=n,
        exclude_doc_id=doc_id if exclude_same_doc else None,
    )

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

    return VoiceSimilarResponse(query=anchor, hits=_postprocess_hits(hits, chunk_frames))
