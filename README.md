# lance-audio-demo

Searchable archive viewer for Swedish press-conference video transcripts.
Single Lance database, FastAPI backend, SvelteKit frontend, optional
multimodal search via Qwen3-VL embeddings.

> Built around [`easytranscriber`](https://github.com/kb-labb/easytranscriber)
> output (alignment JSON + MP4 source). The pipeline ingests both into a
> single self-contained [Lance](https://lancedb.com) dataset, then serves
> search + playback through a typed HTTP API.

> 📐 **New to the codebase?** Read **[GUIDE.md](GUIDE.md)** — architecture, data
> flow, design rationale, and the developer workflow. Running task list:
> **[TODO.md](TODO.md)**.

---

## What this repo does

```
input/sv/*.mp4                 ← source videos
        │
   transcribe                  → output/sv/alignments/*.json    (easytranscriber)
        │
   thumbnail                   → thumbnails/{stem}.jpg           (one per doc)
        │
   ingest-full                 → transcripts.lance/chunks        (FTS + metadata)
                                 transcripts.lance/documents     (media blobs)
        │
   embed-chunks                → chunks.text_embedding           (Qwen3-VL → 2048 dims)
   extract-chunk-frames        → transcripts.lance/chunk_frames  (NEW table, append-only)
                                   ↳ keyed (doc_id, speech_id, chunk_id, frame_idx) — N frames/chunk,
                                     frame_idx=0 is the representative frame
                                   ↳ frame_blob (Blob V2 inline ~50 KB JPEG)
                                   ↳ frame_mime, frame_width, frame_height
   embed-chunk-frames          → chunk_frames.frame_embedding    (Qwen3-VL on each frame,
                                                                  via dataset.add_columns)
        │
   make backend                → FastAPI on :8000 (/api/*)
   make frontend               → SvelteKit + Bun proxy on :3000
```

**Why a separate `chunk_frames` table?** Lance 4.0's `merge_insert` crashes its
encoder when filling blob columns post-hoc on a wide schema with multiple
extension types (`lance.blob.v2` + `lance.json` + fixed-size-list embedding).
The Lance file format 2.2 docs explicitly recommend the two-step
"append + add_columns" pattern for data evolution; we follow it exactly:
`extract-chunk-frames` writes a new fragment per batch, and
`embed-chunk-frames` adds the `frame_embedding` column via
`dataset.add_columns(...)` — no JOIN, no `merge_insert`.

Search modes the API supports:

| `mode` | what it matches | requires |
|---|---|---|
| `fts` | BM25 over chunk text (Tantivy + Swedish stemmer) | `chunks.text` |
| `semantic` | cosine over `chunks.text_embedding` | `embed-chunks` run |
| `hybrid` | FTS + semantic, RRF-fused (Lance native) | both |
| `visual` | cosine over `chunk_frames.frame_embedding` (joined back to chunks), query is text *or* image | frames + embeddings |
| `all` | union of FTS + semantic + visual, RRF-fused | everything |

Optional `rerank=true` swaps the default RRF for a Qwen3-VL-Reranker cross-encoder over the top candidates.

---

## Repo layout

```
lance-audio-demo/
├── backend/                   FastAPI package: app.py (create_app), state.py, deps.py,
│                              clients.py, search/ media/ system/ routers. Serves
│                              /api/search, /api/media, /api/thumbnail, /api/chunk-frame,
│                              /api/health, Range streaming
├── frontend/                  SvelteKit + Tailwind v4 viewer (main UI)
│   ├── src/                   routes + components
│   └── server.ts              Bun static-file server + /api/* proxy
├── demo/                      Secondary SvelteKit app (transformers.js audio demo)
├── src/raudio/                Python ingestion + search core
│   ├── cli/                   typer CLI: ingest, feature, extract-…, serve
│   ├── model/                 PyArrow schemas (schema.py) + Pydantic DTOs (datamodel.py)
│   ├── asr/                   in-process Whisper/wav2vec2 (transcribe.py, detect_language.py)
│   ├── ingest/                JSON → Lance writer (ingest.py, audio.py)
│   ├── media/                 ffmpeg frames (frames.py), download.py, thumbnails.py
│   ├── vllm/                  HTTP clients to remote vLLM: embedding.py, reranker.py,
│   │                          caption.py, summarize.py, image.py, base.py (transport)
│   ├── features/              data-evolution engine.py + columns.py (FEATURES registry)
│   └── retrieval/             FTS + query helpers (search.py) + qwen3_vl_reranker.jinja
├── Makefile                   end-to-end developer commands
├── pyproject.toml             uv-managed Python deps (+ [multimodal] extra)
└── transcripts.lance/         Lance dataset (gitignored — local only)
```

---

## Quickstart

### 0. System prerequisites

- Python 3.11 (managed via `uv`)
- `ffmpeg` on `$PATH`
- NVIDIA GPU + Docker for the multimodal vLLM servers (optional — text FTS works without it)

```bash
make bootstrap     # uv venv + install all Python deps
```

### 1. Ingest (one-time)

Place transcripts under `output/sv/alignments/*.json` and source videos under `input/sv/*.mp4`.

```bash
make pipeline      # transcribe + thumbnail + ingest-full
# OR if alignments already exist:
make ingest-full
```

This populates `transcripts.lance/` with two tables: `documents` (one row per video, with thumbnail + media URI) and `chunks` (one row per ~30s transcript chunk).

### 2. Run the viewer

Three terminals:

```bash
# T1: FastAPI backend
make backend                       # → http://127.0.0.1:8000

# T2: SvelteKit frontend (production build + Bun proxy)
make frontend                      # → http://localhost:3000
# OR Vite HMR for dev:
make frontend-dev                  # → http://localhost:5173
```

You can now search by keyword (FTS) and play any chunk in the right pane.

### 3. Add semantic search (optional)

Requires GPU. Spin up the vLLM embed server, then embed all chunks once:

```bash
# T3: long-running vLLM HTTP server (Qwen3-VL-Embedding-2B)
make embed-server-docker            # → http://127.0.0.1:8001

# Once-off — populates chunks.text_embedding + builds IVF_PQ index
make embed-chunks                   # ~25 min on a 5090 for 145k chunks
```

After this, the UI's "meaning" / "both" toggle is live and `mode=semantic|hybrid` work via the API.

### 4. Add visual search (optional)

Requires GPU + the embed server. Two stages:

```bash
make extract-chunk-frames          # ffmpeg, CPU-bound, ~30 min for 145k chunks
make embed-chunk-frames            # Qwen3-VL image embeddings → frame_embedding
```

Drag-drop an image onto the search bar to query frames; or set mode to `all` for cross-modal text+image fusion.

### 5. Reranking (optional)

```bash
make rerank-server-docker          # Qwen3-VL-Reranker-2B on :8002
```

With both servers up, toggle "rerank" in the UI to engage the cross-encoder over the top candidates. ~200–500 ms latency cost, large recall improvement.

---

## How the search runs end-to-end

### Plain FTS query

```
browser  →  GET /api/search?q=alkohol&mode=fts
frontend (Bun :3000)  →  proxy  →  FastAPI :8000
backend  →  chunks.search().full_text_search("alkohol").limit(20)
                ↳ Tantivy BM25 index (Swedish stemmer)
backend  →  json hits  →  frontend renders list
```

### Semantic / hybrid query

```
browser  →  GET /api/search?q=klimat&mode=hybrid
backend  →  vLLM /v1/embeddings (chat shape, system="Represent the user's input.")
              ↳ Qwen3-VL embedding (full 2048-d, no truncation)
backend  →  chunks.query().full_text_search(q).nearest_to(vec).rerank(RRFReranker())
              ↳ Lance native FTS+vector hybrid with RRF fusion
backend  →  json hits  →  frontend
```

### Visual query (image upload)

```
browser  →  POST /api/search  multipart  (image, mode=visual)
backend  →  PIL center-crop to 448×448, base64 → vLLM /v1/embeddings (image_url)
backend  →  chunks.query()
              .nearest_to(image_vec, vector_column_name="frame_embedding")
              .distance_type("cosine").limit(n)
backend  →  json hits including chunk-frame URLs
frontend →  /api/chunk-frame/{doc_id}/{speech_id}/{chunk_id}?frame_idx=0  (Lance Blob V2 fetch)
```

### Playback

```
HitCard click  →  PlayerPane mounts <video>
<video src="/api/media/{doc_id}">
backend  →  documents table → media_blob (Blob V2 External URI → MP4 on disk)
            ↳ Range request maps to BlobFile.seek(start) + read(length)
```

---

## Inspect the dataset directly

```bash
# chunks table (text + metadata)
uv run python -c "
import lancedb
t = lancedb.connect('./transcripts.lance').open_table('chunks')
print('chunks rows         :', t.count_rows())
print('text_embedding NULL :', t.count_rows('text_embedding IS NULL'))
"

# chunk_frames table (per-chunk JPEG + frame embedding, NEW since Phase 2 v2)
uv run python -c "
import lance, pathlib
p = pathlib.Path('./transcripts.lance/chunk_frames.lance')
if not p.exists():
    print('chunk_frames not yet created — run extract-chunk-frames first.')
else:
    ds = lance.dataset(str(p))
    print('chunk_frames rows :', ds.count_rows())
    print('schema cols       :', ds.schema.names)
    print('has frame_embedding:', 'frame_embedding' in ds.schema.names)
"
```

(Note: Lance 4.0 panics on `IS NULL` against `lance.blob.v2` columns. We avoid
the issue entirely by making `chunk_frames` append-only — no nullable blob
column to query against.)

---

## API cheat sheet

| Endpoint | What it does |
|---|---|
| `GET /api/health` | Pings vLLM embed + rerank, reports DB path / table list / row counts |
| `GET /api/search?q=…&mode=fts\|semantic\|hybrid&n=20` | Text search |
| `POST /api/search` (multipart `image=…&mode=visual`) | Image / cross-modal search |
| `GET /api/documents?page=1&per_page=24` | Paginated browse |
| `GET /api/thumbnail/{doc_id}` | Document thumbnail (Inline Blob) |
| `GET /api/chunk-frame/{doc_id}/{speech_id}/{chunk_id}?frame_idx=N` | Chunk frame (Inline Blob; `frame_idx=0` is the representative frame) |
| `GET /api/media/{doc_id}` | Stream the MP4 (External Blob, Range-friendly) |

The viewer's status badge (top-right of the navbar) hits `/api/health` every 10 s — green = embed/rerank reachable, red = down.

---

## GPU layout (3-GPU box)

| GPU | Service |
|---|---|
| 0 | (free / `make pipeline GPU=0` for transcribe) |
| 1 | `make rerank-server-docker RERANK_GPU=1` |
| 2 | `make embed-server-docker EMBED_GPU=2` |

For a single-GPU box, run sequentially: transcribe first, then start the vLLM servers with `--gpu-memory-utilization 0.40` each.

---

## Development tips

- `make dev` runs backend + frontend together (in two tmux panes if available).
- `make frontend-dev` for Vite HMR while iterating on Svelte components.
- Logs go to `logs/` (rotating, gitignored). Process logs use `tee` so you can watch live with `tail -f logs/<file>`.
- `make vllm-stop` stops both Docker vLLM containers.
- Re-embedding is incremental: every `embed-*` command has `--only-null` (default) and only processes rows where the target column is empty. Safe to Ctrl-C and resume.

---

## Author

[Borg93](https://github.com/Borg93)
