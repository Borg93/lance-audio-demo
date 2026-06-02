"""System endpoints — the health badge and the documents gallery.

Both are pure DB-fact reads off ``StateDep``. ``health`` additionally pings the
two vLLM servers (best-effort, short timeout) so the frontend can show a status
badge without the API depending on them.
"""

from typing import Annotated, Any

from fastapi import APIRouter, Query

from backend.deps import StateDep

router = APIRouter(tags=["system"])


@router.get("/api/health")
def health(state: StateDep) -> dict[str, Any]:
    """Frontend status badge: pings vLLM embed/rerank, reports DB facts."""
    from raudio.vllm.embedding import DEFAULT_EMBED_URL
    from raudio.vllm.reranker import DEFAULT_RERANK_URL

    def _ping(url: str) -> dict[str, Any]:
        import httpx

        try:
            r = httpx.get(f"{url}/health", timeout=1.5)
            return {"ok": r.status_code == 200, "url": url}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "url": url, "error": str(e).split("\n")[0][:120]}

    return {
        "db": {
            "path": str(state.db_path),
            "tables": state.names,
            "chunks": state.chunks.count_rows(),
            "documents": state.docs_ds.count_rows() if state.docs_ds is not None else 0,
        },
        "embed": _ping(DEFAULT_EMBED_URL),
        "rerank": _ping(DEFAULT_RERANK_URL),
    }


@router.get("/api/documents")
def documents(
    state: StateDep,
    page: Annotated[int, Query(ge=1)] = 1,
    per_page: Annotated[int, Query(ge=1, le=100)] = 24,
) -> dict[str, Any]:
    if state.docs_ds is None:
        return {"total": 0, "page": 1, "docs": []}
    total = state.docs_ds.count_rows()
    offset = max(0, (page - 1) * per_page)
    tbl = state.docs_ds.to_table(
        columns=[
            "doc_id",
            "audio_path",
            "duration",
            "referenskod",
            "namn",
            "bildid",
            "extraid",
        ],
        limit=per_page,
        offset=offset,
    )
    return {"total": total, "page": page, "docs": tbl.to_pylist()}
