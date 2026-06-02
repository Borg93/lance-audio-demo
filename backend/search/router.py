"""Search endpoints — GET (query string) and POST (multipart, image upload).

Both build a :class:`SearchSpec`, short-circuit empty input, and delegate to the
framework-free :func:`run_search`, pulling the Lance handles off ``StateDep`` and
the lazy client getters off the factory deps. GET stays sync (threadpooled); POST
is async to await the upload, then offloads the blocking vLLM + Lance work.

No ``from __future__ import annotations`` here: FastAPI introspects these
signatures at runtime, so the annotations stay real objects.
"""

from typing import Annotated, Any

from fastapi import APIRouter, File, Form, Query, UploadFile
from starlette.concurrency import run_in_threadpool

from backend.deps import EmbedderFactoryDep, RerankerFactoryDep, StateDep
from backend.search.service import run_search
from backend.search.spec import SearchMode, SearchSpec

router = APIRouter(tags=["search"])


@router.get("/api/search")
def search_get(
    state: StateDep,
    get_embedder: EmbedderFactoryDep,
    get_reranker: RerankerFactoryDep,
    q: Annotated[str, Query()] = "",
    n: Annotated[int, Query()] = 20,
    mode: Annotated[SearchMode, Query()] = SearchMode.FTS,
    rerank: Annotated[bool, Query()] = False,
    rerank_n: Annotated[int, Query()] = 20,
    language: Annotated[str | None, Query()] = None,
    namn: Annotated[str | None, Query()] = None,
    referenskod: Annotated[str | None, Query()] = None,
    extraid: Annotated[str | None, Query()] = None,
    fuzziness: Annotated[int, Query()] = 0,
    phrase: Annotated[bool, Query()] = False,
    weight: Annotated[float | None, Query()] = None,
    q_vec: Annotated[str, Query()] = "",
    where: Annotated[str | None, Query()] = None,
    prefilter: Annotated[bool, Query()] = True,
) -> list[dict[str, Any]]:
    spec = SearchSpec(
        q=q.strip(),
        n=n,
        mode=mode,
        rerank=rerank,
        rerank_n=rerank_n,
        language=language,
        namn=namn,
        referenskod=referenskod,
        extraid=extraid,
        fuzziness=fuzziness,
        phrase=phrase,
        weight=weight,
        q_vec=q_vec,
        where=where,
        prefilter=prefilter,
    )
    if not spec.q:
        return []
    return run_search(
        state.chunks, state.chunk_frames_tbl, get_embedder, get_reranker, spec, image_bytes=None
    )


@router.post("/api/search")
async def search_post(
    state: StateDep,
    get_embedder: EmbedderFactoryDep,
    get_reranker: RerankerFactoryDep,
    image: Annotated[UploadFile | None, File()] = None,
    q: Annotated[str, Form()] = "",
    n: Annotated[int, Form()] = 20,
    mode: Annotated[SearchMode, Form()] = SearchMode.HYBRID,
    rerank: Annotated[bool, Form()] = False,
    rerank_n: Annotated[int, Form()] = 20,
    weight: Annotated[float | None, Form()] = None,
    language: Annotated[str | None, Form()] = None,
    namn: Annotated[str | None, Form()] = None,
    referenskod: Annotated[str | None, Form()] = None,
    extraid: Annotated[str | None, Form()] = None,
    q_vec: Annotated[str, Form()] = "",
    where: Annotated[str | None, Form()] = None,
    prefilter: Annotated[bool, Form()] = True,
) -> list[dict[str, Any]]:
    spec = SearchSpec(
        q=q.strip(),
        n=n,
        mode=mode,
        rerank=rerank,
        rerank_n=rerank_n,
        language=language,
        namn=namn,
        referenskod=referenskod,
        extraid=extraid,
        fuzziness=0,
        phrase=False,
        weight=weight,
        q_vec=q_vec,
        where=where,
        prefilter=prefilter,
    )
    image_bytes = await image.read() if image is not None else None
    if not spec.q and not image_bytes:
        return []
    # run_search makes blocking vLLM (httpx) + Lance calls — keep the event loop free.
    return await run_in_threadpool(
        run_search,
        state.chunks,
        state.chunk_frames_tbl,
        get_embedder,
        get_reranker,
        spec,
        image_bytes=image_bytes,
    )
