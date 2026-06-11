"""MCP Apps — interactive UIs the host renders inside the conversation.

Two presentation tools on top of the data tools (the MCP Apps extension /
FastMCP 3.2+ ``app=`` support):

* ``show_search_results`` — Prefab app: the hits as a sortable, searchable
  DataTable with a metrics row. The LLM still receives a text summary it can
  reason over (``ToolResult(content=..., structured_content=app)``).
* ``show_clip`` — custom HTML app: the actual video (HTTP-range streamed from
  this backend's ``/api/media/{doc_id}``) seeked to the moment, with the
  surrounding transcript rendered alongside; clicking a segment seeks the
  player. Whether the sandboxed iframe may load media from this backend's
  origin is host-dependent — the CSP declares it, and the transcript half
  works regardless (it ships in the tool result, no network needed).

Both reuse the same service helpers as the data tools (``compact_search`` /
``transcript_window``) so the app and data surfaces can't drift.
"""

from __future__ import annotations

from typing import Any, Literal

from fastmcp import FastMCP
from fastmcp.apps import AppConfig, ResourceCSP
from fastmcp.tools import ToolResult
from prefab_ui.app import PrefabApp
from prefab_ui.components import Column, DataTable, DataTableColumn, Metric, Row

from backend.core.config import get_settings
from backend.mcp.tools import compact_search, transcript_window
from backend.state import AppState

_CLIP_URI = "ui://raudio/clip.html"
_EXT_APPS_CDN = "https://unpkg.com"
_SUMMARY_HITS = 5
_TABLE_TEXT_CHARS = 220


def _fmt_time(seconds: float) -> str:
    # Always h:mm:ss: the table's Start column sorts alphanumerically, and
    # variable-width strings mis-order hour-scale vs minute-scale stamps.
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}"


def _media_base() -> str:
    """The origin the host's browser sandbox must reach to stream media —
    overridable via RAUDIO_MEDIA_BASE_URL for non-loopback deployments."""
    settings = get_settings()
    if settings.media_base_url:
        return settings.media_base_url.rstrip("/")
    host = "127.0.0.1" if settings.host == "0.0.0.0" else settings.host
    return f"http://{host}:{settings.port}"


def _results_app(hits: list[dict[str, Any]], query: str, mode: str) -> PrefabApp:
    # list[Any]: DataTable's rows accept dicts OR ExpandableRow; list is
    # invariant, so a narrower element type fails assignability.
    rows: list[Any] = [
        {
            "video": h["video"] or h["doc_id"],
            "start": _fmt_time(h["start_s"]),
            "text": (h["text"][:_TABLE_TEXT_CHARS] + "…")
            if len(h["text"]) > _TABLE_TEXT_CHARS
            else h["text"],
            "score": round(h["score"], 3) if h["score"] is not None else "",
        }
        for h in hits
    ]
    with PrefabApp() as app, Column(gap=4, css_class="p-6"):
        with Row(gap=6):
            Metric(label="Query", value=query)
            Metric(label="Mode", value=mode)
            Metric(label="Hits", value=str(len(hits)))
        DataTable(
            columns=[
                DataTableColumn(key="video", header="Video", sortable=True),
                DataTableColumn(key="start", header="Start", sortable=True),
                DataTableColumn(key="text", header="Transcript"),
                DataTableColumn(key="score", header="Score", sortable=True),
            ],
            rows=rows,
            search=True,
        )
    return app


def _summary(hits: list[dict[str, Any]], query: str) -> str:
    """What the MODEL sees — enough to cite and to chain into other tools."""
    if not hits:
        return f"No hits for {query!r}. The user sees an empty results table."
    lines = [f"Rendered an interactive table of {len(hits)} hits for {query!r}. Top hits:"]
    lines += [
        f"{i}. [{h['video'] or h['doc_id']}] doc_id={h['doc_id']} @ {h['start_s']:.0f}s: "
        f"{h['text'][:160]}"
        for i, h in enumerate(hits[:_SUMMARY_HITS], 1)
    ]
    return "\n".join(lines)


# The clip viewer: plain HTML + the official ext-apps SDK from CDN (the
# pattern from FastMCP's "Custom HTML Apps" docs). It reads one JSON text
# block from the tool result and builds the player + transcript from it.
_CLIP_HTML = """\
<!DOCTYPE html>
<html>
<head>
  <meta name="color-scheme" content="light dark">
  <style>
    body { margin: 0; padding: 16px; font-family: system-ui, sans-serif; }
    video { width: 100%; max-height: 360px; border-radius: 8px; background: #000; }
    h2 { font-size: 15px; margin: 12px 0 6px; }
    .seg { padding: 6px 8px; border-radius: 6px; cursor: pointer; display: flex; gap: 10px; }
    .seg:hover { background: rgba(127,127,127,.15); }
    .seg.hot { background: rgba(59,130,246,.18); }
    .t { color: #888; font-variant-numeric: tabular-nums; flex-shrink: 0; }
    #err { color: #b91c1c; font-size: 13px; }
  </style>
</head>
<body>
  <video id="player" controls preload="metadata"></video>
  <div id="err"></div>
  <h2 id="title"></h2>
  <div id="segs"></div>
  <script type="module">
    import { App } from "https://unpkg.com/@modelcontextprotocol/ext-apps@1.0.1/app-with-deps";

    const app = new App({ name: "raudio clip", version: "1.0.0" });
    const fmt = (s) => { const m = Math.floor(s / 60); return m + ":" + String(Math.floor(s % 60)).padStart(2, "0"); };

    app.ontoolresult = ({ content, structuredContent }) => {
      const block = content?.find((c) => c.type === "text" && c.text.startsWith("{"));
      const clip = structuredContent ?? (block ? JSON.parse(block.text) : null);
      if (!clip) return;
      const player = document.getElementById("player");
      player.src = clip.media_url;
      player.currentTime = clip.start_s;
      player.onerror = () => {
        document.getElementById("err").textContent =
          "Video could not load in this host's sandbox — transcript below still works. " +
          "Open locally: " + clip.media_url;
      };
      document.getElementById("title").textContent = clip.video || clip.doc_id;
      const segs = document.getElementById("segs");
      segs.replaceChildren(...clip.segments.map((seg) => {
        const div = document.createElement("div");
        div.className = "seg" + (seg.start_s <= clip.start_s && clip.start_s < seg.end_s ? " hot" : "");
        const t = document.createElement("span");
        t.className = "t";
        t.textContent = fmt(seg.start_s);
        const x = document.createElement("span");
        x.textContent = seg.text;
        div.append(t, x);
        div.onclick = () => { player.currentTime = seg.start_s; player.play(); };
        return div;
      }));
    };

    await app.connect();
  </script>
</body>
</html>
"""


def register_app_tools(mcp: FastMCP, state: AppState) -> None:
    """Register the presentation tools, closing over the shared state."""
    media_base = _media_base()

    @mcp.tool(app=True)
    def show_search_results(
        query: str,
        mode: Literal["fts", "semantic", "hybrid"] = "hybrid",
        n: int = 15,
        language: str | None = None,
        video_name: str | None = None,
        topic: str | None = None,
    ) -> ToolResult:
        """Search Riksarkivet's moving-image archive (rörlig bild: film/video/
        audio recordings) AND show the hits as an interactive table the user
        can sort, search, and read — use when the user wants to BROWSE results
        rather than have you summarize them. Same arguments as
        ``search_chunks``. You receive a text summary of the top hits (with
        doc_id and start_s) so you can keep reasoning or open one with
        ``show_clip``.
        """
        hits = compact_search(
            state,
            query=query,
            mode=mode,
            n=n,
            language=language,
            video_name=video_name,
            topic=topic,
        )
        return ToolResult(
            content=_summary(hits, query),
            structured_content=_results_app(hits, query, mode),
        )

    @mcp.resource(
        _CLIP_URI,
        app=AppConfig(
            csp=ResourceCSP(
                resource_domains=[_EXT_APPS_CDN, media_base],
                connect_domains=[media_base],
            )
        ),
    )
    def clip_view() -> str:
        """The clip viewer UI (video player + clickable transcript)."""
        return _CLIP_HTML

    @mcp.tool(app=AppConfig(resource_uri=_CLIP_URI))
    def show_clip(doc_id: str, start_s: float, window_s: float = 45.0) -> ToolResult:
        """Show the actual archive VIDEO at one moment, with the surrounding
        transcript beside it (clicking a transcript line seeks the player).
        Use after a search when the user wants to watch/hear a recording from
        the rörlig bild archive: pass the hit's ``doc_id`` and ``start_s``.
        Requires the raudio backend to be reachable from the user's machine
        (it streams the media).
        """
        window = transcript_window(state, doc_id=doc_id, center_s=start_s, window_s=window_s)
        clip = {
            "doc_id": doc_id,
            "video": window["video"],
            "media_url": f"{media_base}/api/media/{doc_id}",
            "start_s": start_s,
            "segments": window["segments"],
        }
        # The viewer reads structuredContent; the model gets one cheap line
        # instead of the raw player payload.
        return ToolResult(
            content=(
                f"Showing {window['video'] or doc_id} at {_fmt_time(start_s)} — "
                f"{len(window['segments'])} transcript segments beside the player."
            ),
            structured_content=clip,
        )
