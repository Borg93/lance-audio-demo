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

router = APIRouter(prefix="/api/topics", tags=["topics"])

_NOT_BUILT: dict[str, Any] = {"built": False, "layers": 0, "n_chunks": 0, "hierarchy": None}


def _decode_jsonb(raw: Any) -> Any:
    """Lance JSONB hands back either the raw JSON string or an already-decoded value."""
    return json.loads(raw) if isinstance(raw, str) else raw


@router.get("")
def get_topics(state: StateDep) -> dict[str, Any]:
    """The topic hierarchy for the treemap, or ``built: false`` if not generated yet."""
    path = state.db_path / "topics.lance"
    if not path.exists():
        return _NOT_BUILT
    rows = lance.dataset(str(path)).to_table().to_pylist()
    if not rows:
        return _NOT_BUILT
    row = rows[0]
    return {
        "built": True,
        "layers": int(row.get("layers") or 0),
        "n_chunks": int(row.get("n_chunks") or 0),
        "hierarchy": _decode_jsonb(row.get("hierarchy")),
    }
