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
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from backend.deps import EmbedderFactoryDep, RerankerFactoryDep, StateDep
from backend.search.service import run_search
from backend.search.spec import SearchMode, SearchSpec

router = APIRouter(tags=["search"])


class DocTranscriptChunk(BaseModel):
    speech_id: int
    chunk_id: int
    start: float
    end: float
    text: str
    alignments: list[dict[str, Any]]


class DocTranscriptResponse(BaseModel):
    doc_id: str
    chunks: list[DocTranscriptChunk]


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
    topic: Annotated[str | None, Query()] = None,
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
        topic=topic,
        fuzziness=fuzziness,
        phrase=phrase,
        weight=weight,
        q_vec=q_vec,
        where=where,
        prefilter=prefilter,
    )
    if not spec.q and not spec.q_vec and not spec.topic:
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
    topic: Annotated[str | None, Form()] = None,
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
        topic=topic,
        fuzziness=0,
        phrase=False,
        weight=weight,
        q_vec=q_vec,
        where=where,
        prefilter=prefilter,
    )
    image_bytes = await image.read() if image is not None else None
    if not spec.q and not spec.q_vec and not image_bytes and not spec.topic:
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


@router.get("/api/chunk-alignments/{doc_id}/{speech_id}/{chunk_id}")
def chunk_alignments(state: StateDep, doc_id: str, speech_id: int, chunk_id: int) -> dict[str, Any]:
    """Per-word alignments for one chunk — lazy-fetched by the player when a hit is
    opened. Search results omit the multi-KB ``alignments_json`` blob (it was ~93%
    of the payload and only the selected hit renders it), so the player fetches the
    real array here on demand. Sync handler → threadpool (the Lance read is blocking).
    """
    from raudio.retrieval.search import parse_alignments_json

    safe_doc = doc_id.replace("'", "''")
    where = f"doc_id = '{safe_doc}' AND speech_id = {speech_id} AND chunk_id = {chunk_id}"
    rows = state.chunks_ds.to_table(columns=["alignments_json"], filter=where).to_pylist()
    alignments = parse_alignments_json(rows[0]["alignments_json"]) if rows else []
    return {"alignments": alignments}


@router.get("/api/doc-transcript/{doc_id}")
def doc_transcript(state: StateDep, doc_id: str) -> DocTranscriptResponse:
    """Whole-document, chunk-segmented transcript — ordered by start time. One lazy
    fetch drives both a clickable chunk timeline and (flattened) the player's
    karaoke track. Reuses ``parse_alignments_json`` so the per-chunk ``alignments``
    mirror the per-word shape the player already renders; chunks with no timing keep
    ``alignments == []`` (video still plays, no karaoke). Sync handler → threadpool
    (the Lance read is blocking). No vector/_score column is projected, so the plain
    dataset scan can't hit the FTS/vector _score restriction.
    """
    from raudio.retrieval.search import parse_alignments_json

    safe_doc = doc_id.replace("'", "''")
    where = f"doc_id = '{safe_doc}'"
    rows = state.chunks_ds.to_table(
        columns=["speech_id", "chunk_id", "start", "end", "text", "alignments_json"],
        filter=where,
    ).to_pylist()
    # Order by the contract field (start time) directly — `speech_id` is the
    # source-assigned id, not a sequential index, so (speech_id, chunk_id) is only
    # a proxy for time order. `start` is non-null in CHUNK_SCHEMA; (speech_id,
    # chunk_id) is a stable tiebreaker for any chunks that share a start.
    rows.sort(key=lambda r: (r["start"], r["speech_id"], r["chunk_id"]))
    chunks = [
        DocTranscriptChunk(
            speech_id=r["speech_id"],
            chunk_id=r["chunk_id"],
            start=r["start"],
            end=r["end"],
            text=r["text"],
            alignments=parse_alignments_json(r["alignments_json"]),
        )
        for r in rows
    ]
    return DocTranscriptResponse(doc_id=doc_id, chunks=chunks)
