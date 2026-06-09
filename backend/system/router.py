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
        "embed": _ping(state.settings.embed_url),
        "rerank": _ping(state.settings.rerank_url),
    }


_COLUMN_EXCLUDE = {"alignments_json"}


@router.get("/api/columns")
def columns(state: StateDep) -> list[dict[str, str]]:
    """Filterable scalar columns of the ``chunks`` table (name + friendly type).

    Lets the UI show *what* can go in a WHERE filter. Vector / blob / list /
    embedding columns are omitted — they can't appear in a SQL filter anyway.
    """
    import pyarrow as pa

    out: list[dict[str, str]] = []
    for field in state.chunks.schema:
        name = field.name
        if name in _COLUMN_EXCLUDE or name.endswith("_embedding"):
            continue
        t = field.type
        if pa.types.is_integer(t) or pa.types.is_floating(t):
            kind = "number"
        elif pa.types.is_boolean(t):
            kind = "boolean"
        elif pa.types.is_temporal(t):
            kind = "time"
        elif pa.types.is_string(t) or pa.types.is_large_string(t):
            kind = "text"
        else:
            continue  # list / struct / binary / vector — not filterable in SQL
        out.append({"name": name, "type": kind})
    return out


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
