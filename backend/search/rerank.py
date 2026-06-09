"""Cross-encoder head rerank: re-score the top ``rerank_n`` hits, keep the tail.

Pure: no Lance, no exceptions. The reranker getter (passed in) is the only
dependency, and that getter owns its own 503 mapping (see ``backend.clients``).
Extracted verbatim from the former monolithic ``service.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from raudio.vllm.reranker import VLLMReranker


def _rerank_by_text(
    get_reranker: Callable[[], VLLMReranker],
    query: str,
    hits: list[dict[str, Any]],
    rerank_n: int,
    n: int,
) -> list[dict[str, Any]]:
    """Cross-encoder rerank the top ``rerank_n`` of ``hits``, then return ``n``
    results: the reranked head followed by the remaining first-stage hits.

    Reranking only the head bounds the (slow) cross-encoder cost while letting
    the result list be longer than the reranked window. The reranker is
    text-only — it scores ``query`` jointly with each candidate's transcript and
    ignores vectors/images, so with no query text (image-only search) or no hits
    it's a plain top-``n`` and the first-stage order is preserved.
    """
    if not query or not hits:
        return hits[:n]
    head = hits[:rerank_n]
    tail = hits[rerank_n:]
    scores = get_reranker().rerank(query, [h["text"] for h in head])
    head = [h for _, h in sorted(zip(scores, head, strict=False), key=lambda p: -p[0])]
    return (head + tail)[:n]
