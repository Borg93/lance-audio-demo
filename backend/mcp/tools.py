"""The curated MCP tool set — five hand-shaped tools over the existing services.

Deliberately NOT an auto-export of the REST API: each tool wraps the same
in-process service a router uses, but returns compact, prose-ready payloads
with hard caps, because an LLM context window is the consumer. The docstrings
double as the tool descriptions the model reads — keep them instructive about
WHEN to use each tool, not just what it does.

Domain errors (404/422/503 in REST) are re-raised as :class:`ToolError` so
hosts get a structured tool failure instead of a stack trace.
"""

from __future__ import annotations

from typing import Any, Literal

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import ValidationError as PydanticValidationError

from backend.clients import ensure_embedder, ensure_reranker
from backend.core.exceptions import DomainError
from backend.graph import router as graph
from backend.search.service import run_search
from backend.search.spec import SearchMode, SearchSpec
from backend.state import AppState
from backend.topics.router import get_topics

_MAX_HITS = 25
_MAX_TEXT_CHARS = 600
_KG_ROW_CAP = 50


def _score_of(hit: dict[str, Any]) -> float | None:
    """One comparable relevance number (higher = better) across search modes."""
    if (rrf := hit.get("_relevance_score")) is not None:
        return float(rrf)
    if (bm25 := hit.get("_score")) is not None:
        return float(bm25)
    if (dist := hit.get("_distance")) is not None:
        return 2.0 - float(dist)  # cosine distance in [0, 2], inverted
    return None


def _compact_hit(hit: dict[str, Any]) -> dict[str, Any]:
    return {
        "doc_id": hit["doc_id"],
        "video": hit.get("namn"),
        "start_s": hit["start"],
        "end_s": hit["end"],
        "text": (hit.get("text") or "")[:_MAX_TEXT_CHARS],
        "score": _score_of(hit),
    }


def _sql_quote(value: str) -> str:
    return value.replace("'", "''")


def register_tools(mcp: FastMCP, state: AppState) -> None:
    """Register the tool set, closing over the app's shared :class:`AppState`."""

    @mcp.tool
    def search_chunks(
        query: str,
        mode: Literal["fts", "semantic", "hybrid"] = "hybrid",
        n: int = 10,
        language: str | None = None,
        video_name: str | None = None,
        topic: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search the transcript corpus (Swedish parliamentary/archive videos).

        Use this FIRST for any content question. ``mode``: "fts" = exact
        keyword (BM25, best for names and quoted phrases), "semantic" = meaning
        (best for paraphrased questions), "hybrid" = both fused (good default).
        Each hit is a transcript chunk with its video id and time span — follow
        up with ``get_transcript_window`` for surrounding context. Filters:
        ``language`` (ISO code, e.g. "sv"), ``video_name`` (substring of the
        video title), ``topic`` (an exact topic name from ``list_topics``).
        """
        try:
            spec = SearchSpec(
                q=query,
                mode=SearchMode(mode),
                n=min(n, _MAX_HITS),
                language=language,
                namn=video_name,
                topic=topic,
            )
        except PydanticValidationError as exc:
            raise ToolError(f"invalid search arguments: {exc}") from exc
        try:
            hits = run_search(
                state.chunks,
                state.chunk_frames_tbl,
                get_embedder=lambda: ensure_embedder(state),
                get_reranker=lambda: ensure_reranker(state),
                spec=spec,
                image_bytes=None,
            )
        except DomainError as exc:
            raise ToolError(str(exc.detail)) from exc
        return [_compact_hit(h) for h in hits]

    @mcp.tool
    def get_transcript_window(
        doc_id: str,
        center_s: float,
        window_s: float = 60.0,
    ) -> dict[str, Any]:
        """The transcript around one moment of one video — context expansion.

        Use after ``search_chunks`` or ``find_similar_voices``: pass the hit's
        ``doc_id`` and its ``start_s`` as ``center_s`` to read what was said
        around the match (±``window_s`` seconds, capped at ±300).
        """
        half = min(max(window_s, 1.0), 300.0)
        lo, hi = center_s - half, center_s + half
        flt = f"doc_id = '{_sql_quote(doc_id)}' AND \"end\" >= {lo} AND start <= {hi}"
        rows = state.chunks_ds.to_table(
            columns=["namn", "start", "end", "text"], filter=flt
        ).to_pylist()
        if not rows:
            raise ToolError(f"no transcript in that window — unknown doc_id {doc_id!r}?")
        rows.sort(key=lambda r: r["start"])
        return {
            "doc_id": doc_id,
            "video": rows[0].get("namn"),
            "window_s": [lo, hi],
            "segments": [
                {"start_s": r["start"], "end_s": r["end"], "text": r["text"]} for r in rows
            ],
            "text": " ".join(r["text"] for r in rows),
        }

    @mcp.tool
    def find_similar_voices(doc_id: str, t: float, n: int = 8) -> dict[str, Any]:
        """Find other moments where the SAME-SOUNDING voice speaks (other videos).

        Anchor = whoever is speaking in video ``doc_id`` at ``t`` seconds.
        Ranks diarized speaker turns across the corpus by voiceprint
        similarity. Use to follow a person across recordings; note it matches
        the VOICE, not the textual content.
        """
        # Imported lazily: the voice package pulls heavier deps that the other
        # tools (and tool listing) shouldn't pay for on import.
        from backend.voice.service import similar_voices

        try:
            resp = similar_voices(
                state.speaker_embeddings_tbl,
                state.speakers_tbl,
                state.chunks_ds,
                state.chunk_frames_tbl,
                doc_id=doc_id,
                turn_id=None,
                speaker=None,
                t=t,
                n=min(n, _MAX_HITS),
                exclude_same_doc=True,
            )
        except DomainError as exc:
            raise ToolError(str(exc.detail)) from exc
        data = resp.model_dump()
        return {
            "anchor": data["query"],
            "hits": [
                {
                    "doc_id": h["doc_id"],
                    "video": h.get("namn"),
                    "speaker": h.get("speaker_label"),
                    "turn_start_s": h.get("turn_start"),
                    "turn_end_s": h.get("turn_end"),
                    "similarity": h.get("turn_score"),
                    "text": (h.get("text") or "")[:_MAX_TEXT_CHARS],
                }
                for h in data["hits"]
            ],
        }

    @mcp.tool
    def query_knowledge_graph(cypher: str) -> list[dict[str, Any]]:
        """Run a read-only Cypher query over the corpus knowledge graph.

        Schema — nodes: ``Entity(entity_id, name, entity_type, mention_count)``
        and ``Chunk(chunk_id, doc_id)``; relationships:
        ``(Entity)-[:MENTIONS]->(Chunk)`` (where an entity is spoken about) and
        ``(Entity)-[:RELATIONSHIP {description}]->(Entity)``. Use for WHO/WHAT
        connection questions ("which entities co-occur with X?"); use
        ``search_chunks`` for content questions. Results are capped at 50 rows.
        Example: ``MATCH (e:Entity)-[r:RELATIONSHIP]->(o:Entity) WHERE e.name
        CONTAINS 'klimat' RETURN e.name, r.description, o.name``.
        """
        # Reuses the graph router's version-cached CypherEngine on purpose —
        # one engine per kg_entities version, shared with /api/graph.
        res = graph._resources(state.db_path)
        if res is None:
            raise ToolError("knowledge graph not built — run the kg pipeline first")
        try:
            capped = graph._enforce_limit(cypher, _KG_ROW_CAP)
            return graph._run_rows(res, capped)
        except Exception as exc:
            raise ToolError(f"cypher query failed: {exc}") from exc

    @mcp.tool
    def list_topics() -> dict[str, Any]:
        """The corpus topic hierarchy (Swedish topic names, nested, with chunk
        counts). Use to discover what the corpus covers and to get exact topic
        names for ``search_chunks(topic=...)``.
        """
        resp = get_topics(state)
        if not resp.built:
            raise ToolError("topics not built — run `raudio feature topics` first")
        return {
            "layers": resp.layers,
            "n_chunks": resp.n_chunks,
            "hierarchy": resp.hierarchy,
        }
