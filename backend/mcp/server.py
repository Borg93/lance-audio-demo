"""FastMCP server factory — built per-app, mounted at ``/mcp``.

``build_mcp`` returns the bare :class:`FastMCP` (what tests drive through the
in-memory client); ``build_mcp_app`` wraps it in the streamable-HTTP ASGI app
the factory mounts. The MCP session manager lives in that sub-app's lifespan,
so the app factory MUST combine it with its own lifespan (see
:func:`backend.app.create_app`) — without it every ``/mcp`` request 500s.

Connect an MCP host to ``http://<host>:<port>/mcp/`` (streamable HTTP), e.g.::

    claude mcp add raudio --transport http http://localhost:8000/mcp/
"""

from __future__ import annotations

from fastmcp import FastMCP
from fastmcp.server.http import StarletteWithLifespan

from backend.mcp.apps import register_app_tools
from backend.mcp.tools import register_tools
from backend.state import AppState

_INSTRUCTIONS = """\
Tools over a corpus of Swedish archive/parliamentary videos with ASR
transcripts, topic clusters, speaker voiceprints, and a knowledge graph.
Content questions: start with `search_chunks`, then expand promising hits with
`get_transcript_window`. Person-by-voice questions: `find_similar_voices`.
Entity/connection questions: `query_knowledge_graph`. Corpus overview / exact
topic names: `list_topics`. Transcripts and topic names are Swedish — query in
Swedish for keyword (fts) search; semantic mode tolerates other languages.
Cite findings as (doc_id, start_s) so they can be deep-linked to the video.
When the user wants to BROWSE or WATCH rather than read your summary, use the
interactive tools: `show_search_results` (sortable results table) and
`show_clip` (video player + transcript at a moment)."""


def build_mcp(state: AppState) -> FastMCP:
    """The MCP server with all tools registered against this app's state."""
    mcp = FastMCP(name="raudio", instructions=_INSTRUCTIONS)
    register_tools(mcp, state)
    register_app_tools(mcp, state)
    return mcp


def build_mcp_app(state: AppState) -> StarletteWithLifespan:
    """The mountable streamable-HTTP ASGI app (carries its own lifespan)."""
    return build_mcp(state).http_app(path="/")
