"""MCP (Model Context Protocol) surface over the raudio corpus.

A FastMCP server mounted at ``/mcp`` by the app factory, exposing a small
curated tool set (search, transcript windows, voice similarity, knowledge
graph, topics) so MCP hosts — Claude Desktop/Code, ChatGPT, or our own chat
agent — can RAG over the corpus through the same in-process services the REST
routers use.
"""

from backend.mcp.server import build_mcp, build_mcp_app

__all__ = ["build_mcp", "build_mcp_app"]
