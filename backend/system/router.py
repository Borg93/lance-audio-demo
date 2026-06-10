"""System endpoints — the health badge and the documents gallery.

Both are pure DB-fact reads off ``StateDep``. ``health`` additionally pings the
two vLLM servers (best-effort, short timeout) so the frontend can show a status
badge without the API depending on them.
"""

from typing import Annotated

import httpx
import pyarrow as pa
from fastapi import APIRouter, Query

from backend.deps import StateDep
from backend.schemas.system import (
    ColumnKind,
    DbFacts,
    DocumentsResponse,
    DocumentSummary,
    FilterColumn,
    HealthResponse,
    VllmPing,
)

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/health")
def health(state: StateDep) -> HealthResponse:
    """Frontend status badge: pings vLLM embed/rerank, reports DB facts."""
    # One pooled client per process (state.http); the module fallback only fires
    # for bare AppState constructions in unit tests.
    http = state.http if state.http is not None else httpx

    def _ping(url: str) -> VllmPing:
        try:
            r = http.get(f"{url}/health", timeout=1.5)
            return VllmPing(ok=r.status_code == 200, url=url)
        except Exception as e:  # noqa: BLE001
            # Keep the exception TYPE — httpx messages alone (e.g. a bare host)
            # don't identify whether it was a timeout, refusal, or DNS failure.
            first_line = str(e).split("\n")[0][:100]
            return VllmPing(ok=False, url=url, error=f"{type(e).__name__}: {first_line}")

    return HealthResponse(
        db=DbFacts(
            path=str(state.db_path),
            tables=state.names,
            chunks=state.chunks.count_rows(),
            documents=state.docs_ds.count_rows() if state.docs_ds is not None else 0,
        ),
        embed=_ping(state.settings.embed_url),
        rerank=_ping(state.settings.rerank_url),
    )


_COLUMN_EXCLUDE = {"alignments_json"}


@router.get("/columns")
def columns(state: StateDep) -> list[FilterColumn]:
    """Filterable scalar columns of the ``chunks`` table (name + friendly kind).

    Lets the UI show *what* can go in a WHERE filter. Vector / blob / list /
    embedding columns are omitted — they can't appear in a SQL filter anyway.
    """
    out: list[FilterColumn] = []
    for field in state.chunks.schema:
        name = field.name
        if name in _COLUMN_EXCLUDE or name.endswith("_embedding"):
            continue
        t = field.type
        if pa.types.is_integer(t) or pa.types.is_floating(t):
            kind = ColumnKind.number
        elif pa.types.is_boolean(t):
            kind = ColumnKind.boolean
        elif pa.types.is_temporal(t):
            kind = ColumnKind.time
        elif pa.types.is_string(t) or pa.types.is_large_string(t):
            kind = ColumnKind.text
        else:
            continue  # list / struct / binary / vector — not filterable in SQL
        out.append(FilterColumn(name=name, type=kind))
    return out


@router.get("/documents")
def documents(
    state: StateDep,
    page: Annotated[int, Query(ge=1)] = 1,
    per_page: Annotated[int, Query(ge=1, le=100)] = 24,
) -> DocumentsResponse:
    if state.docs_ds is None:
        return DocumentsResponse(total=0, page=1)
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
    return DocumentsResponse(
        total=total,
        page=page,
        docs=[DocumentSummary.model_validate(row) for row in tbl.to_pylist()],
    )
