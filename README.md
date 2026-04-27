# lance-audio-demo

Searchable archive viewer for Swedish press-conference video transcripts.
Single Lance database, FastAPI backend, SvelteKit frontend, optional
multimodal search via Qwen3-VL embeddings.

> Built around [`easytranscriber`](https://github.com/kb-labb/easytranscriber)
> output (alignment JSON + MP4 source). The pipeline ingests both into a
> single self-contained [Lance](https://lancedb.com) dataset, then serves
> search + playback through a typed HTTP API.

---

## What this repo does

```
input/sv/*.mp4                 ← source videos
        │
   transcribe                  → output/sv/alignments/*.json   (easytranscriber)
        │
   thumbnail                   → thumbnails/{stem}.jpg          (one per doc)
        │
   ingest-full                 → transcripts.lance/             (Lance: chunks + documents)
        │
   embed-chunks                → chunks.text_embedding          (Qwen3-VL → 1024 dims)
   extract-chunk-frames        → chunks.frame_blob              (ffmpeg @ chunk.start)
   embed-chunk-frames          → chunks.frame_embedding         (Qwen3-VL on each frame)
        │
   make backend                → FastAPI on :8000 (/api/*)
   make frontend               → SvelteKit + Bun proxy on :3000
```

Search modes the API supports:

| `mode` | what it matches | requires |
|---|---|---|
| `fts` | BM25 over chunk text (Tantivy + Swedish stemmer) | `chunks.text` |
| `semantic` | cosine over `chunks.text_embedding` | `embed-chunks` run |
| `hybrid` | FTS + semantic, RRF-fused (Lance native) | both |
| `visual` | cosine over `chunks.frame_embedding`, query is text *or* image | frames + embeddings |
| `all` | union of FTS + semantic + visual, RRF-fused | everything |

Optional `rerank=true` swaps the default RRF for a Qwen3-VL-Reranker cross-encoder over the top candidates.

---

## Repo layout

```
lance-audio-demo/
├── backend/app.py             FastAPI: /api/search, /api/media, /api/thumbnail,
│                              /api/chunk-frame, /api/health, Range streaming
├── frontend/                  SvelteKit + Tailwind v4 viewer (main UI)
│   ├── src/                   routes + components
│   └── server.ts              Bun static-file server + /api/* proxy
├── demo/                      Secondary SvelteKit app (transformers.js audio demo)
├── src/raudio/                Python ingestion + search core
│   ├── cli.py                 typer CLI: ingest, embed-chunks, extract-…, serve
│   ├── ingest.py              JSON → Lance writer (chunks + documents tables)
│   ├── schema.py              PyArrow schemas (text_embedding, frame_blob, …)
│   ├── embeddings.py          vLLM HTTP client (text + image) + Qwen reranker
│   ├── frames.py              ffmpeg per-chunk frame extractor
│   └── search.py              FTS + vector + hybrid query helpers
├── tools/                     Jinja templates, NVIDIA toolkit installer
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
# T3: long-running vLLM HTTP server (Qwen3-VL-Embedding-8B)
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
make rerank-server-docker          # Qwen3-VL-Reranker-8B on :8002
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
              ↳ Qwen3-VL embedding (4096 → MRL-truncated to 1024)
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
frontend →  /api/chunk-frame/{doc_id}/{speech_id}/{chunk_id}  (Lance Blob V2 fetch)
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
uv run python -c "
import lancedb
t = lancedb.connect('./transcripts.lance').open_table('chunks')
print('rows           :', t.count_rows())
print('text emb null  :', t.count_rows('text_embedding IS NULL'))
print('frame mime null:', t.count_rows('frame_mime IS NULL'))
print('frame emb null :', t.count_rows('frame_embedding IS NULL'))
"
```

(`frame_mime` is the safe NULL sentinel — Lance 4.0 panics on `IS NULL` against `lance.blob.v2` columns.)

---

## API cheat sheet

| Endpoint | What it does |
|---|---|
| `GET /api/health` | Pings vLLM embed + rerank, reports DB path / table list / row counts |
| `GET /api/search?q=…&mode=fts\|semantic\|hybrid&n=20` | Text search |
| `POST /api/search` (multipart `image=…&mode=visual`) | Image / cross-modal search |
| `GET /api/documents?page=1&per_page=24` | Paginated browse |
| `GET /api/thumbnail/{doc_id}` | Document thumbnail (Inline Blob) |
| `GET /api/chunk-frame/{doc_id}/{speech_id}/{chunk_id}` | Chunk frame (Inline Blob) |
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
