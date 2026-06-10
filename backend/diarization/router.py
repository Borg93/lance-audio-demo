"""Diarization endpoint — serve a video's speaker turns for the player timeline.

A diarization pass writes one set of rows per video into an append-only
``speaker_turns.lance`` table: ``doc_id``, ``turn_id``, ``speaker_label`` (e.g.
``SPEAKER_00``) and absolute-video-second ``start`` / ``end`` floats. This one
read-only route filters that table by ``doc_id`` and hands the frontend the
turns (sorted by start) plus the sorted-distinct speaker set.

``speaker_turns.lance`` is read on demand (mirroring :mod:`backend.topics.router`)
rather than at startup, so a diarization rebuild is picked up without restarting
the server — and an absent table or doc simply returns ``built: false``.
"""

import lance
from fastapi import APIRouter

from backend.deps import StateDep
from backend.media.blobs import valid_doc_id
from backend.schemas.diarization import DiarizationResponse, SpeakerTurn

router = APIRouter(prefix="/api/diarization", tags=["diarization"])


def _not_built(doc_id: str) -> DiarizationResponse:
    """The empty contract payload for a missing table / doc / row set."""
    return DiarizationResponse(built=False, doc_id=doc_id)


@router.get("/{doc_id}")
def get_diarization(doc_id: str, state: StateDep) -> DiarizationResponse:
    """A video's speaker turns (sorted by start), or ``built: false`` if absent."""
    # Whitelist the path param (16-char hex) BEFORE it is inlined into the Lance
    # filter literal below — otherwise a crafted doc_id is a SQL-injection vector.
    valid_doc_id(doc_id)
    path = state.db_path / "speaker_turns.lance"
    if not path.exists():
        return _not_built(doc_id)
    rows = lance.dataset(str(path)).to_table(filter=f"doc_id = '{doc_id}'").to_pylist()
    if not rows:
        return _not_built(doc_id)
    # The columns are a writer-guaranteed contract (every row carries all five,
    # non-null), so index directly — a schema drift should 500 loudly, not
    # silently return a built-but-empty payload the timeline can't render.
    turns = sorted(
        (
            SpeakerTurn(
                turn_id=int(row["turn_id"]),
                speaker=row["speaker_label"],
                start=float(row["start"]),
                end=float(row["end"]),
            )
            for row in rows
        ),
        key=lambda turn: turn.start,
    )
    speakers = sorted({turn.speaker for turn in turns})
    return DiarizationResponse(built=True, doc_id=doc_id, turns=turns, speakers=speakers)
