# MCP — the archive as a tool surface for LLM agents

> **REMOVED (services split, 2026-07-21):** the in-process MCP mount lived on
> the monolith (`backend/mcp`, `:8000/mcp/`) and was dropped when the backend
> split into per-domain services. Re-home it behind a service (or a standalone
> MCP gateway over the three `/api` domains) if the tool surface returns. The
> rest of this doc describes the removed design for reference.

The backend mounts a [Model Context Protocol](https://modelcontextprotocol.io)
server at **`http://<host>:8000/mcp/`** (streamable HTTP), built with
[FastMCP 3](https://gofastmcp.com). Any MCP host — VS Code Copilot, Claude
Desktop/Code, Goose — can search the corpus, read transcripts, follow voices,
query the knowledge graph, and render **interactive UIs (MCP Apps)** inside
the conversation: a sortable results table and a video clip player with a
word-level karaoke transcript.

The server identifies itself as covering **Riksarkivet's moving-image archive
("rörlig bild")** — digitized film/video/audio: press conferences, press
releases, seminars, government recordings. Every tool description leads with
that identity and explicitly defers written/scanned documents to the
`riksarkivet`/ra-mcp document server, so a host running both routes
spoken-content questions here.

---

## Architecture

```
backend/mcp/
├── server.py   build_mcp(state) → FastMCP   |  build_mcp_app(state) → ASGI sub-app
├── tools.py    5 curated data tools + shared service helpers
└── apps.py     2 MCP App tools (Prefab table, custom HTML clip viewer)
```

- `backend/app.py` mounts the sub-app at `/mcp` and **must** combine its
  lifespan with the API's own via `fastmcp.utilities.lifespan.combine_lifespans`
  — the MCP session manager lives in that lifespan; without it every `/mcp`
  request 500s.
- Tools are **closures over the shared `AppState`** (same Lance handles and
  vLLM clients the REST routers use). No second data path.
- The app tools and data tools share the same service helpers
  (`compact_search`, `transcript_window`), so the two surfaces can't drift on
  filters or hit shape.
- This is deliberately **not** an auto-export of the REST API: each tool
  returns compact, prose-ready payloads with hard caps (25 hits, 600 chars of
  text, 50 KG rows) because an LLM context window is the consumer. Docstrings
  double as the tool descriptions the model reads.
- Domain errors surface as `ToolError` (a structured tool failure the model
  can react to), never stack traces. Model-supplied `doc_id`s are validated
  before they reach any Lance SQL.

## Tool surface

| Tool | Kind | Use |
|---|---|---|
| `search_chunks` | data | Raw transcript hits for the **model** to reason over / chain. FTS, semantic, or hybrid; filters: `language`, `video_name`, `topic` (exact names from `list_topics`). |
| `get_transcript_window` | data | ±N seconds of timed segments around one moment — context expansion after a hit. |
| `find_similar_voices` | data | Voiceprint search: who *sounds like* the speaker at (doc, t), across recordings. |
| `query_knowledge_graph` | data | Read-only Cypher over `Entity`/`Chunk` nodes, `MENTIONS`/`RELATIONSHIP` edges. |
| `list_topics` | data | The topic hierarchy + exact topic names for filtered search. |
| `show_search_results` | **app** | Search **and** render the hits as an interactive table — the default for searches the user asked for. |
| `show_clip` | **app** | Render the video player + synced transcript at a moment. |

## MCP Apps (the interactive UIs)

Both app tools return a `ToolResult` whose `structured_content` the host
renders in a sandboxed iframe, while `content` carries a compact text summary
**for the model** — the user gets the UI, the model gets something it can
keep reasoning over.

**`show_search_results`** — a [Prefab](https://prefab.prefect.io) app:
metrics row + `DataTable` (sortable, client-side search). Each row expands
(`ExpandableRow`) to the full hit text plus two buttons that **hand the next
step back to the chat** via the `SendMessage` action — "▶ Show this clip"
pushes `Call show_clip(doc_id='…', start_s=…)` into the conversation as if
the user typed it, and the model makes the call. The button text spells out
the exact call signature because models guess argument names from prose.

**`show_clip`** — a custom HTML app (`ui://raudio/clip.html`, the
[ext-apps](https://github.com/modelcontextprotocol/ext-apps) SDK from CDN):
an HTML5 player seeked to the hit, beside the transcript window. The
transcript mirrors the frontend's `transcript-highlighter`: `timeupdate` +
`seeked` drive a **word-level karaoke cursor** (binary search over per-word
`[start, end, text]` triples from `alignments_json`; `include_words=True` is
clip-app-only — the LLM data tool stays compact), clicking a line seeks the
player, and the active segment auto-scrolls. The resource's CSP allow-lists
the ext-apps CDN and the media origin.

### Sound in VS Code: the media-clip endpoint

VS Code's webview Chromium ships **without the AAC decoder**
([microsoft/vscode#167685](https://github.com/microsoft/vscode/issues/167685)),
so the archive's H.264+AAC files play silently inside MCP apps. The fix:
`show_clip` points the player at

```
GET /api/media-clip/{doc_id}?lo=<s>&hi=<s>
```

which cuts the transcript window out of the source with ffmpeg (reading
seekably over the backend's own Range-streamed `/api/media`, since media
lives in Lance blobs) and re-encodes to **H.264 + MP3** — fully playable in
webviews. Excerpts are small complete MP4s (Range-seekable), disk-cached in
`$TMPDIR/raudio-clip-cache` with oldest-first eviction (~2–4 s first build,
instant after). Clip time starts at 0 = document-time `lo`; the viewer maps
clip-local ↔ document time via `media_offset_s`, so timestamps and karaoke
stay in real recording time. The viewer footer links the full original
recording.

## Connecting a host

The workspace ships `.vscode/mcp.json` (gitignored) with:

```json
{ "servers": { "raudio": { "type": "http", "url": "http://localhost:8000/mcp/" } } }
```

- **VS Code**: open the repo → accept the discovered server (or
  `MCP: List Servers → raudio → Start`). Copilot Chat in **Agent** mode; tick
  the raudio tools under the wrench icon. *Gotcha:* a chat request carries at
  most ~128 tools — untick large servers (ra-mcp ≈ 80 tools) or VS Code
  auto-deactivates tools and the model reports them "disabled".
- **Claude Code / Desktop**: `claude mcp add raudio --transport http
  http://localhost:8000/mcp/`
- **Remote hosts**: set `MEDIA_MEDIA_BASE_URL` (validated `AnyHttpUrl`) to an
  origin the *user's browser* can reach — it lands in the clip app's CSP and
  media URLs. Unset = `http://127.0.0.1:8000`.

**After any backend restart, restart the server entry in the host too** —
streamable-HTTP sessions live in backend memory, and hosts cache the tool
list. (On this box the backend is supervised: kill `:8000` and it respawns
with the code on disk in ~2 s; never start a second instance.)

## Testing

```bash
# In-memory protocol tests (no server, no HTTP) — skip without the local dataset
uv run pytest tests/test_backend_mcp.py -v

# Browser preview of the APP tools + live MCP message inspector (no host needed)
uv run fastmcp dev apps scripts/mcp_dev.py --mcp-port 8010 --dev-port 8080

# Poke the LIVE endpoint from the terminal
uv run fastmcp list http://localhost:8000/mcp/ --auth none
uv run fastmcp call http://localhost:8000/mcp/ search_chunks query=skatt mode=fts n=3 --auth none
```

`scripts/mcp_dev.py` exposes the same server as a bare `mcp` instance the
`fastmcp` CLI can load; it opens its own read-only Lance handles, safe
alongside the running backend.
