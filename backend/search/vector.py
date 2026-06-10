"""Single-column cosine vector search over the chunks table.

A failed search raises a domain :class:`ValidationError` (HTTP 400, stable
message) with the real Lance error logged, never interpolated.
"""

from __future__ import annotations

import logging
from typing import Any

from backend.core.exceptions import ValidationError
from backend.search.constants import (
    _PAYLOAD_COLUMNS,
    _VECTOR_MAX_NPROBES,
    _VECTOR_NPROBES,
    _VECTOR_REFINE_FACTOR,
)

logger = logging.getLogger(__name__)


def _vector_search(
    table,
    vec: Any,
    column: str,
    *,
    n: int,
    where: str | None,
    prefilter: bool = True,
) -> list[dict[str, Any]]:
    """Run a cosine vector search on ``column``; returns raw list of dicts.

    Returns ``[]`` when the embedding column doesn't exist yet (embeddings are
    attached post-ingest by ``embed-chunks``), so fusion modes degrade to the
    other rankings instead of erroring.
    """
    if vec is None or column not in table.schema.names:
        return []
    try:
        qb = (
            table.search(vec.tolist(), vector_column_name=column)
            .distance_type("cosine")
            .minimum_nprobes(_VECTOR_NPROBES)
            .maximum_nprobes(_VECTOR_MAX_NPROBES)
            .refine_factor(_VECTOR_REFINE_FACTOR)
            .select([*_PAYLOAD_COLUMNS, "_distance"])
            .limit(n)
        )
        if where:
            qb = qb.where(where, prefilter=prefilter)
        return qb.to_list()
    except Exception as e:
        logger.warning("vector search failed", exc_info=True)
        raise ValidationError("vector search failed") from e
