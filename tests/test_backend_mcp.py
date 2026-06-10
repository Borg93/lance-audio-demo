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
        "show_search_results",
        "show_clip",
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
    joined = " ".join(s["text"] for s in window["segments"])
    assert hits[0]["text"][:40] in joined


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


async def test_show_search_results_renders_app_with_llm_summary(mcp: Any) -> None:
    from fastmcp import Client

    async with Client(mcp) as client:
        tools = {t.name: t for t in await client.list_tools()}
        # App tools carry the MCP Apps linkage so hosts know to render a UI.
        assert (tools["show_search_results"].meta or {}).get("ui"), "missing _meta.ui"
        result = await client.call_tool(
            "show_search_results", {"query": "jag", "mode": "fts", "n": 5}
        )
    summary = result.content[0].text  # type: ignore[union-attr]
    assert "interactive table" in summary or "No hits" in summary
    if "doc_id=" in summary:
        # The structured side is the serialized Prefab component tree.
        assert result.structured_content, "missing Prefab structured content"
        assert "DataTable" in str(result.structured_content)


async def test_show_clip_returns_player_payload(mcp: Any) -> None:

    from fastmcp import Client

    async with Client(mcp) as client:
        tools = {t.name: t for t in await client.list_tools()}
        ui_meta = (tools["show_clip"].meta or {}).get("ui") or {}
        assert ui_meta.get("resourceUri") == "ui://raudio/clip.html"
        # The linked ui:// resource must be fetchable (that's what hosts render).
        html = await client.read_resource("ui://raudio/clip.html")
        assert "<video" in html[0].text  # type: ignore[union-attr]

        hits = (
            await client.call_tool("search_chunks", {"query": "jag", "mode": "fts", "n": 1})
        ).data
        if not hits:
            pytest.skip("no fts hits in local dataset")
        result = await client.call_tool(
            "show_clip", {"doc_id": hits[0]["doc_id"], "start_s": hits[0]["start_s"]}
        )
    # The viewer's payload rides in structuredContent; the model gets one line.
    clip = result.structured_content
    assert clip["doc_id"] == hits[0]["doc_id"]
    assert clip["media_url"].endswith(f"/api/media/{hits[0]['doc_id']}")
    assert clip["segments"], "clip payload must carry transcript segments"
    assert "Showing" in result.content[0].text  # type: ignore[union-attr]


async def test_unknown_topic_is_a_loud_tool_error(mcp: Any) -> None:
    from fastmcp import Client
    from fastmcp.exceptions import ToolError

    async with Client(mcp) as client:
        with pytest.raises(ToolError, match=r"unknown topic|topics not built"):
            await client.call_tool(
                "search_chunks", {"query": "skatt", "mode": "fts", "topic": "Arbetsmarknad"}
            )


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
