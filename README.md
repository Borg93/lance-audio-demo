# lance-audio-demo

Searchable archive viewer for Swedish press-conference video transcripts.
Single Lance database, FastAPI backend, SvelteKit frontend, optional
multimodal search via out-of-process Qwen3-VL embeddings and a Qwen3-VL
cross-encoder reranker.

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
                                 transcripts.lance/documents     (media + thumbnail blobs)
        │
   embed-chunks                → chunks.text_embedding           (Qwen3-VL → 2048-d)
   extract-chunk-frames        → transcripts.lance/chunk_frames  (separate table, append-only)
                                   ↳ keyed (doc_id, speech_id, chunk_id, frame_idx) — N frames/chunk,
                                     frame_idx=0 is the representative frame
                                   ↳ frame_blob (Blob V2 Inline ~50 KB JPEG)
                                   ↳ frame_mime, frame_width, frame_height
   embed-chunk-frames          → chunk_frames.frame_embedding    (Qwen3-VL on each frame,
                                                                  via dataset.add_columns)
        │
   make backend                → FastAPI on :8000 (/api/*)
   make frontend               → SvelteKit + Bun proxy on :3000
```

**Why a separate `chunk_frames` table?** Lance 4.0's `merge_insert` crashes its
encoder when filling blob columns post-hoc on a wide schema with multiple
extension types (`lance.blob.v2` + `lance.json` + a fixed-size-list embedding).
The Lance file-format 2.2 docs recommend the two-step "append + add_columns"
pattern for data evolution, and we follow it exactly: `extract-chunk-frames`
writes a new fragment per batch into a separate table, and `embed-chunk-frames`
adds the `frame_embedding` column via `dataset.add_columns(...)` — no JOIN, no
`merge_insert`.

### Search modes

The API exposes five modes (`backend/search/spec.py:SearchMode`):

| `mode` | what it matches | requires |
|---|---|---|
| `fts` | BM25 keyword search over `chunks.text` (Tantivy + Swedish stemmer) | `chunks.text` |
| `semantic` | cosine over `chunks.text_embedding` (text-vector) | `embed-chunks` run |
| `visual` | cosine over `chunk_frames.frame_embedding` (frame-vector), joined back to chunks; query is text *or* image | frames + embeddings |
| `hybrid` | `fts` + `semantic`, fused | both |
| `all` | `fts` + `semantic` + `visual`, fused (3-way) | everything |

The frontend chooses the mode automatically: the Keyword / Vector / Hybrid
selector maps to `fts` / `semantic` / `hybrid`; attaching an image switches to
`visual` (image only) or `all` (image + text).

**Fusion.** Reciprocal-rank fusion (RRF, k=60) is the default and the only
option that scales past two legs — each leg returns a ranked list and a
candidate scores the sum of `1/(k+rank)` over the lists it appears in. The
3-way `all` mode *always* uses equal-weight RRF. The 2-way `hybrid` mode uses
RRF by default, but switches to a `LinearCombinationReranker(weight)` blend
(`final = weight·vectorScore + (1−weight)·ftsScore`) when the Balance slider
sets `weight`. There is no per-leg weight for `all` yet.

**Cross-encoder rerank** (`rerank=true`, optional). This does *not* replace the
fusion step. After fusion, the Qwen3-VL cross-encoder re-scores only the top
`rerank_n` results (the "head", default 20); the rest keep first-stage order.
It is **text-only** — it scores the user's combined text intent (`q` + `q_vec`)
against each candidate's transcript and never sees the image or the vectors.
On image-only `visual` search there is no query text, so rerank is a no-op.

---

## Repo layout

```
lance-audio-demo/
├── backend/                   FastAPI package: app.py (create_app), state.py, deps.py,
│   ├── search/                spec.py (SearchSpec), service.py (run_search), router.py
│   ├── media/                 blobs.py + router.py (thumbnail, chunk-frame, Range-streamed media)
│   └── system/                router.py (health, columns, documents)
├── frontend/                  SvelteKit + Svelte 5 + Tailwind v4 viewer (main UI)
│   ├── src/                   routes + components + bits-ui (shadcn-style) ui/ kit
│   └── server.ts              Bun static-file server + /api/* proxy
├── demo/                      Secondary SvelteKit app (transformers.js audio demo)
├── src/raudio/                Python ingestion + search core
│   ├── cli/                   typer CLI: ingest, feature, extract-chunk-frames, compact, serve, …
│   ├── model/                 PyArrow schemas (schema.py) + Pydantic DTOs (datamodel.py)
│   ├── asr/                   in-process Whisper/wav2vec2 (transcribe.py, detect_language.py)
│   ├── ingest/                JSON → Lance writer (ingest.py, audio.py)
│   ├── media/                 ffmpeg frames (frames.py), download.py, thumbnails.py
│   ├── vllm/                  HTTP clients to remote vLLM servers: embedding.py, reranker.py,
│   │                          caption.py, summarize.py, image.py, base.py (transport)
│   ├── features/              data-evolution engine.py + columns.py (FEATURES registry)
│   └── retrieval/             FTS + query helpers (search.py) + qwen3_vl_reranker.jinja
├── Makefile                   end-to-end developer commands
├── pyproject.toml             uv-managed Python deps (+ [multimodal] extra)
└── transcripts.lance/         Lance dataset (gitignored — local only)
```

Import paths follow the package layout: `raudio.vllm.embedding`,
`raudio.vllm.reranker`, `backend.search.service`. The embedding and reranker
clients live under `raudio.vllm`, not `raudio.retrieval`.

---

## Quickstart

### 0. System prerequisites

- Python 3.11 (managed via `uv`)
- `ffmpeg` on `$PATH`
- NVIDIA GPU for the multimodal vLLM servers (optional — keyword FTS works without it)

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

This populates `transcripts.lance/` with two tables: `documents` (one row per
video, with thumbnail + media URI) and `chunks` (one row per transcript chunk).

### 2. Run the viewer

Two terminals:

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

Requires a GPU. Start the vLLM embedding server, then embed all chunks once:

```bash
# T3: long-running vLLM HTTP server (Qwen3-VL-Embedding-2B, port 8001)
make embed-server-docker            # → http://127.0.0.1:8001
# (or `make embed-server` to run it via uvx instead of Docker)

# Once-off — populates chunks.text_embedding + builds the IVF_PQ index.
# Wraps `raudio feature text_embedding`.
make embed-chunks                   # ~25 min on a 5090 for 145k chunks
```

After this, the UI's Vector / Hybrid modes are live and `mode=semantic|hybrid`
work via the API.

### 4. Add visual search (optional)

Requires a GPU + the embedding server. Two stages:

```bash
make extract-chunk-frames          # ffmpeg, CPU-bound, ~30 min for 145k chunks
make embed-chunk-frames            # Qwen3-VL image embeddings → frame_embedding
                                   # (wraps `raudio feature frame_embedding`)
```

Drag-drop an image onto the search bar to query frames (`mode=visual`); add text
as well and the request becomes `mode=all` (3-way text + image fusion).

### 5. Reranking (optional)

```bash
make rerank-server-docker          # Qwen3-VL-Reranker-2B on :8002
# (or `make rerank-server` for the uvx variant)
```

With the rerank server up, toggle "Rerank" in the Settings popover to engage the
cross-encoder over the top `rerank_n` results. ~200–500 ms latency cost.

---

## How search runs end-to-end

### Keyword (FTS) query

```
browser  →  GET /api/search?q=alkohol&mode=fts
frontend (Bun :3000)  →  proxy  →  FastAPI :8000
backend  →  chunks.search(MatchQuery("alkohol", "text")).select(...).limit(n)
                ↳ Tantivy BM25 index (Swedish stemmer); PhraseQuery when phrase=true
backend  →  json hits  →  frontend renders list
```

### Semantic / hybrid query

```
browser  →  GET /api/search?q=klimat&mode=hybrid
backend  →  vLLM /v1/embeddings (Qwen-VL chat shape, system="Represent the user's input.")
              ↳ Qwen3-VL embedding (full 2048-d, no Matryoshka truncation), L2-normalized
backend  →  chunks.search(query_type="hybrid")
                  .vector(text_vec).text(MatchQuery(q, "text"))
                  .rerank(RRFReranker() | LinearCombinationReranker(weight))
                  .nprobes(20).refine_factor(3).limit(n)
              ↳ Lance-native FTS + text-vector hybrid, fused by RRF (or the Balance slider)
backend  →  json hits  →  frontend
```

### Visual query (image upload)

```
browser  →  POST /api/search  multipart  (image, mode=visual)
backend  →  PIL center-crop + resize to 392×392, base64 → vLLM /v1/embeddings (image_url)
backend  →  chunk_frames.search(image_vec, vector_column_name="frame_embedding")
                  .distance_type("cosine").nprobes(20).refine_factor(3).limit(n)
              ↳ rank frames, collapse to one hit per chunk, join back to `chunks` for the payload
backend  →  json hits  →  frontend
frontend →  GET /api/chunk-frame/{doc_id}/{speech_id}/{chunk_id}?frame_idx=0  (Blob V2 fetch)
```

### Playback

```
HitCard click  →  PlayerPane mounts <video>
<video src="/api/media/{doc_id}">
backend  →  documents table → media_blob (Blob V2 External URI → MP4 on disk/HF/S3)
            ↳ HTTP Range request maps to a seek + bounded read on the BlobFile
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

# chunk_frames table (per-chunk JPEG + frame embedding)
uv run python -c "
import lance, pathlib
p = pathlib.Path('./transcripts.lance/chunk_frames.lance')
if not p.exists():
    print('chunk_frames not yet created — run extract-chunk-frames first.')
else:
    ds = lance.dataset(str(p))
    print('chunk_frames rows  :', ds.count_rows())
    print('schema cols        :', ds.schema.names)
    print('has frame_embedding:', 'frame_embedding' in ds.schema.names)
"
```

(Note: Lance 4.0 panics on `IS NULL` against `lance.blob.v2` columns. We avoid
this by making `chunk_frames` append-only — there is no nullable blob column to
query against.)

---

## API cheat sheet

| Endpoint | What it does |
|---|---|
| `GET /api/health` | Reports DB path / table list / row counts; pings the vLLM embed + rerank servers |
| `GET /api/search?q=…&mode=fts\|semantic\|hybrid&n=20` | Text search (query string) |
| `POST /api/search` (multipart, `image=…`, `mode=visual\|all`) | Image / cross-modal search |
| `GET /api/columns` | Filterable scalar columns of `chunks` (name + friendly type) |
| `GET /api/documents?page=1&per_page=24` | Paginated browse |
| `GET /api/thumbnail/{doc_id}` | Document thumbnail (Blob V2 Inline) |
| `GET /api/chunk-frame/{doc_id}/{speech_id}/{chunk_id}?frame_idx=N` | Chunk frame (Blob V2 Inline; `frame_idx=0` is the representative frame) |
| `GET /api/media/{doc_id}` | Stream the MP4 (Blob V2 External, HTTP Range supported) |

`SearchSpec` fields: `q` (FTS/keyword text), `q_vec` (separate text for the
vector leg; falls back to `q`), `n` (results, default 20, clamped 1..200),
`mode`, `rerank` (bool), `rerank_n` (rerank head size, default 20, clamped
1..200), `weight` (0..1, hybrid only; `None` = RRF), `fuzziness` (0..2),
`phrase`, the structured filters `language` / `namn` / `referenskod` /
`extraid`, `where` (raw SQL WHERE, ANDed with the structured filters), and
`prefilter` (default `True`).

The service core (`backend/search/service.py:run_search`) is framework-free: it
takes the Lance handles plus the two vLLM client getters as callables, and the
routers wire it to the request via dependency injection.

The viewer's status badge (top-right) polls `/api/health` — green = embed/rerank
reachable, red = down.

---

## Frontend at a glance

SvelteKit + Svelte 5 runes + a bits-ui shadcn-style `ui/` kit:

- a compact search toolbar (mode selector + query + Search; Hybrid reveals a
  second "Vector" query box for `q_vec`),
- a ⚙ Settings popover (Results to return, Rerank top + toggle, Balance slider,
  Match style),
- a Filters popover (a structured column·operator·value builder that
  auto-applies, plus column show/hide and a raw-SQL advanced field),
- results as list / grid / table (the table has thumbnails, query highlighting,
  and a column chooser), and
- a Help (?) popover with an Examples tab and a "How search works" flow diagram.

---

## GPU layout (3-GPU box)

| GPU | Service |
|---|---|
| 0 | (free / `make pipeline GPU=0` for transcribe) |
| 1 | `make rerank-server-docker RERANK_GPU=1` |
| 2 | `make embed-server-docker EMBED_GPU=2` |

Both 2B models also co-locate on a single GPU at `--gpu-memory-utilization 0.45`
each. On the Blackwell box, drop the `--with kernels` flag from the uvx server
commands (the HF kernels API changed); the bundled FlashAttention 2 works.

---

## Development tips

- `make dev` runs backend + frontend together (in two tmux panes if available).
- `make frontend-dev` for Vite HMR while iterating on Svelte components.
- Logs go to `logs/` (rotating, gitignored). Process logs use `tee`, so watch
  them live with `tail -f logs/<file>`.
- `make vllm-stop` stops both Docker vLLM containers.
- Feature builds are incremental: `raudio feature <name>` defaults to
  `--only-null`, processing only rows where the target column is empty. Safe to
  Ctrl-C and resume; pass `--all` to drop and rebuild.

---

## Limitations

- **Image search is frame similarity, not face/identity recognition.** It finds
  visually similar video frames; it does not identify who is in them.
- **No speaker diarization** — there is no link between who is on screen and who
  is speaking.
- **The reranker is text-only** — it scores transcript text against the query
  and ignores the image and the vectors.
- **3-way `all` fusion is equal-weight RRF** — there is no image-vs-text weight
  yet (the Balance slider only applies to 2-way `hybrid`).

---

## Author

[Borg93](https://github.com/Borg93)
</content>
</invoke>
