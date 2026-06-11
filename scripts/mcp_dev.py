"""Standalone entrypoint for FastMCP tooling — dev preview, inspector, CLI.

Exposes the same MCP server the backend mounts at ``/mcp/``, but as a bare
``mcp`` instance the ``fastmcp`` CLI can load directly (no FastAPI, no
uvicorn). The backend on :8000 keeps running independently — pick free ports:

    # Browser preview of the APP tools (results table, clip viewer) with an
    # MCP message inspector — no MCP host needed:
    uv run fastmcp dev apps scripts/mcp_dev.py --mcp-port 8010 --dev-port 8080

    # Poke at the data tools from the terminal:
    uv run fastmcp list scripts/mcp_dev.py
    uv run fastmcp call scripts/mcp_dev.py search_chunks query=skatt mode=fts n=3

Note: this opens its own read-only Lance handles to RAUDIO_DB (default
``transcripts_v2.lance``) — safe alongside the running backend.
"""

from backend.core.config import get_settings
from backend.mcp.server import build_mcp
from backend.state import open_resources

mcp = build_mcp(open_resources(get_settings().db_path))
