"""Topics endpoint — serve the precomputed topic hierarchy for the Tree page.

``raudio feature topics`` clusters chunks (isolated Toponymy worker) and writes
``topic_l*`` + ``doc_topic`` columns on ``chunks``; :func:`build_topic_tree`
folds those into a single-row ``topics.lance`` table whose ``hierarchy`` column
is Lance JSONB — the nested ``{name, children | value}`` tree the LayerChart
``<Treemap>`` renders. This one read-only route hands that tree to the frontend.

The topic *filter* lives on ``/api/search`` (``topic=`` matches any ``topic_l*``
layer, see :mod:`backend.search.service`); this route is only the hierarchy.

``topics.lance`` is read on demand (one tiny row) rather than at startup, so a
``raudio feature topics`` rebuild is picked up without restarting the server.
"""

import json
from typing import Any

import lance
from fastapi import APIRouter

from backend.deps import StateDep
from backend.schemas.topics import TopicsResponse
from raudio.features.topic_tree import NOISE_LABEL

router = APIRouter(prefix="/api/topics", tags=["topics"])

_NOT_BUILT = TopicsResponse(built=False, noise_label=NOISE_LABEL)


def _decode_jsonb(raw: Any) -> Any:
    """Lance JSONB hands back either the raw JSON string or an already-decoded value."""
    return json.loads(raw) if isinstance(raw, str) else raw


@router.get("")
def get_topics(state: StateDep) -> TopicsResponse:
    """The topic hierarchy for the treemap, or ``built: false`` if not generated yet."""
    path = state.db_path / "topics.lance"
    if not path.exists():
        return _NOT_BUILT
    rows = lance.dataset(str(path)).to_table().to_pylist()
    if not rows:
        return _NOT_BUILT
    # The columns are a writer-guaranteed contract (build_topic_tree always writes
    # all three non-null), so index directly — a schema drift should 500 loudly,
    # not silently return a built-but-empty payload the treemap can't render.
    row = rows[0]
    return TopicsResponse(
        built=True,
        layers=int(row["layers"]),
        n_chunks=int(row["n_chunks"]),
        hierarchy=_decode_jsonb(row["hierarchy"]),
        noise_label=NOISE_LABEL,
    )
