"""Smoke tests for the MCP tool surface against the local Lance DB.

Same contract as ``test_backend_smoke.py``: these need the real (gitignored)
dataset, so CI without it skips cleanly. Tools are driven through FastMCP's
in-memory client — no HTTP, no running server — which exercises exactly what
an MCP host calls. GPU-dependent paths (semantic/hybrid embeddings) are not
exercised; ``search_chunks`` is tested in fts mode.

Run:  uv run pytest tests/test_backend_mcp.py -v
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

DB_PATH = Path(__file__).resolve().parent.parent / "transcripts_v2.lance"

pytestmark = [
    pytest.mark.skipif(
        not (DB_PATH / "chunks.lance").exists(),
        reason="local transcripts_v2.lance dataset not present",
    ),
    pytest.mark.anyio,
]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(scope="module")
def mcp() -> Any:
    from backend.mcp.server import build_mcp
    from backend.state import open_resources

    return build_mcp(open_resources(DB_PATH))


async def test_lists_the_curated_tools(mcp: Any) -> None:
    from fastmcp import Client

    async with Client(mcp) as client:
        names = {t.name for t in await client.list_tools()}
    assert {
        "search_chunks",
        "get_transcript_window",
        "find_similar_voices",
        "query_knowledge_graph",
        "list_topics",
    } <= names


async def test_search_chunks_fts_returns_compact_hits(mcp: Any) -> None:
    from fastmcp import Client

    async with Client(mcp) as client:
        result = await client.call_tool("search_chunks", {"query": "jag", "mode": "fts", "n": 3})
    hits = result.data
    assert isinstance(hits, list)
    assert len(hits) <= 3
    if hits:
        assert {"doc_id", "video", "start_s", "end_s", "text", "score"} <= hits[0].keys()


async def test_transcript_window_expands_a_hit(mcp: Any) -> None:
    from fastmcp import Client

    async with Client(mcp) as client:
        hits = (
            await client.call_tool("search_chunks", {"query": "jag", "mode": "fts", "n": 1})
        ).data
        if not hits:
            pytest.skip("no fts hits in local dataset")
        result = await client.call_tool(
            "get_transcript_window",
            {"doc_id": hits[0]["doc_id"], "center_s": hits[0]["start_s"], "window_s": 30},
        )
    window = result.data
    assert window["doc_id"] == hits[0]["doc_id"]
    assert window["segments"], "window around a hit must contain at least the hit itself"
    assert hits[0]["text"][:40] in window["text"]


async def test_transcript_window_unknown_doc_is_a_tool_error(mcp: Any) -> None:
    from fastmcp import Client
    from fastmcp.exceptions import ToolError

    async with Client(mcp) as client:
        with pytest.raises(ToolError):
            await client.call_tool(
                "get_transcript_window", {"doc_id": "no-such-doc", "center_s": 10.0}
            )


async def test_list_topics_or_clean_error(mcp: Any) -> None:
    from fastmcp import Client
    from fastmcp.exceptions import ToolError

    async with Client(mcp) as client:
        try:
            data = (await client.call_tool("list_topics", {})).data
        except ToolError as exc:
            assert "not built" in str(exc)
            return
    assert data["n_chunks"] > 0
    assert data["hierarchy"]["children"]


async def test_knowledge_graph_query_or_clean_error(mcp: Any) -> None:
    from fastmcp import Client
    from fastmcp.exceptions import ToolError

    async with Client(mcp) as client:
        try:
            rows = (
                await client.call_tool(
                    "query_knowledge_graph",
                    {"cypher": "MATCH (e:Entity) RETURN e.name LIMIT 3"},
                )
            ).data
        except ToolError as exc:
            assert "not built" in str(exc) or "failed" in str(exc)
            return
    assert isinstance(rows, list)
    assert len(rows) <= 3
