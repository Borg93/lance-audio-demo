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
> **[TODO.md](docs/TODO.md)**. Forward-looking architecture bets:
> **[WHATS_LEFT.md](docs/WHATS_LEFT.md)**.

---

## What this repo does

```
input/sv/*.mp4                 ← source videos
        │
   transcribe                  → output/sv/alignments/*.json    (easytranscriber)
        │
   thumbnail                   → thumbnails/{stem}.jpg           (one per doc)
        │
   ingest-full                 → <DB>/chunks                     (FTS + metadata)
                                 <DB>/documents                  (media + thumbnail blobs)
        │   (<DB> = $(DB), Makefile default ./transcripts.lance — but the LIVE,
        │    served dataset on this machine is transcripts_v2.lance; the default
        │    ./transcripts.lance is an EMPTY manifest with no tables)
        │
   embed-chunks                → chunks.text_embedding           (Qwen3-VL → 2048-d)
   extract-chunk-frames        → <DB>/chunk_frames               (separate table, append-only)
                                   ↳ keyed (doc_id, speech_id, chunk_id, frame_idx) — N frames/chunk,
                                     frame_idx=0 is the representative frame
                                   ↳ frame_blob (Blob V2 Inline ~50 KB JPEG)
                                   ↳ frame_mime, frame_width, frame_height
   embed-chunk-frames          → chunk_frames.frame_embedding    (Qwen3-VL on each frame,
                                                                  via dataset.add_columns)
        │   chunks ALSO carries (so it has TWO vector columns):
        │     ↳ text_embedding (2048) + frame_embedding (2048, chunk-level image
        │       vector, frame_idx=0, for the image atlas)
        │     ↳ atlas_x/atlas_y/atlas_cluster        (text-space EVōC projection)
        │     ↳ atlas_img_x/atlas_img_y/atlas_img_cluster (visual-space EVōC projection)
        │
   make captions               → chunk_frames.caption            (Gemma 4 Swedish captions,
   (caption + caption_embedding)                                  via your VLM on :8003)
                                 chunk_frames.caption_embedding   (Qwen3-VL → 2048-d, scene search)
                                   ↳ powers mode=scene (caption-vector) + mode=scene_fts (caption BM25)
                                   ↳ NOT built on the live DB yet → scene/scene_fts return EMPTY
        │
   make services-up            → viewer:8101 search:8102 annotator:8103 (/api/*)
   make frontend               → viewer zone + per-domain proxy on :5274
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

The API exposes seven modes (`backend/search/spec.py:SearchMode`):

| `mode` | what it matches | requires |
|---|---|---|
| `fts` | BM25 keyword search over `chunks.text` (Tantivy + Swedish stemmer) | `chunks.text` |
| `semantic` | cosine over `chunks.text_embedding` (text-vector) | `embed-chunks` run |
| `visual` | cosine over `chunk_frames.frame_embedding` (frame-vector), joined back to chunks; query is text *or* image | frames + embeddings |
| `scene` | cosine over `chunk_frames.caption_embedding` (Swedish-caption text-vector), joined back to chunks | `make captions` run |
| `scene_fts` | BM25 keyword search over `chunk_frames.caption` (Swedish captions), joined back to chunks | `make captions` run |
| `hybrid` | `fts` + `semantic`, fused | both |
| `all` | `fts` + `semantic` + `visual` + `scene`, fused (up to 4-way) | everything |

The frontend chooses the mode automatically: the Keyword / Vector / Hybrid
selector maps to `fts` / `semantic` / `hybrid`; attaching an image switches to
`visual` (image only) or `all` (image + text).

**Fusion.** Reciprocal-rank fusion (RRF, k=60) is the default and the only
option that scales past two legs — each leg returns a ranked list and a
candidate scores the sum of `1/(k+rank)` over the lists it appears in. The
`all` mode fuses up to four independent legs — FTS(text) + text-vector
(`text_embedding`) + frame-vector (`frame_embedding`; the image vector if an
image is attached, otherwise the text vector) + caption/scene-vector
(`caption_embedding`) — always with equal-weight RRF (legs are unioned by
chunk, not chained). The 2-way `hybrid` mode uses RRF by default, but switches
to a `LinearCombinationReranker(weight)` blend (`final = weight·vectorScore +
(1−weight)·ftsScore`) when the Balance slider sets `weight`. There is no
per-leg weight for `all` yet.

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
├── services/                  Per-domain FastAPI microservices (lance-ns services/ shape)
│   ├── common/                shared kernel: lancekit, core (RFC 9457 handlers), schemas, state, deps
│   ├── viewer/                read plane :8101 — media/blobs, transcripts, atlas, graph, topics, voice
│   ├── search/                retrieval :8102 — FTS/vector/hybrid, encoders, rerank
│   └── annotator/             write plane :8103 — annotations (merge_insert+409), assist, jobs
├── frontend/                  turborepo: apps/{media,annotator} + packages/@lance/{engine,labeling,ui,api}
│   ├── apps/media/            viewer zone (SvelteKit) + server.ts (per-domain /api + /annotate proxy)
│   └── apps/annotator/        annotator zone (kit.paths.base=/annotate)
├── src/rmedia/                Python ingestion + search core (Ray/vLLM pipeline)
│   ├── cli/                   typer CLI: ingest, feature, extract-chunk-frames, compact, serve, …
│   ├── model/                 PyArrow schemas (schema.py) + Pydantic DTOs (datamodel.py)
│   ├── asr/                   in-process Whisper/wav2vec2 (transcribe.py, detect_language.py)
│   ├── ingest/                JSON → Lance writer (ingest.py, audio.py)
│   ├── media/                 ffmpeg frames (frames.py), download.py, thumbnails.py
│   ├── vllm/                  HTTP clients to remote vLLM servers: embedding.py, reranker.py,
│   │                          caption.py, summarize.py, image.py, base.py (transport)
│   ├── features/              data-evolution engine.py + columns.py (FEATURES registry)
│   │                          + projection.py (EVōC atlas fit for the 'atlas'/'atlas_visual' features)
│   └── retrieval/             FTS + query helpers (search.py) + qwen3_vl_reranker.jinja
├── Makefile                   end-to-end developer commands
├── pyproject.toml             uv-managed Python deps (+ [multimodal] and [atlas] extras)
└── transcripts.lance/         Lance dataset (gitignored — local only; the populated
                               one on this machine is transcripts_v2.lance)
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

This populates the dataset at `$(DB)` (Makefile `DB ?= ./transcripts.lance`)
with two tables: `documents` (one row per video, with thumbnail + media URI) and
`chunks` (one row per transcript chunk).

> **Heads-up on this machine:** the live, served dataset is
> `transcripts_v2.lance` (chunks 145,175 rows; documents 1,154; chunk_frames
> 145,175). The Makefile default `./transcripts.lance` is an **empty manifest
> with no tables** — point `DB=transcripts_v2.lance` (or `make services-up
> DB=transcripts_v2.lance`) at the populated one.

### 2. Run the viewer

Two terminals:

```bash
# T1: the three services
make services-up                   # → viewer:8101 search:8102 annotator:8103

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
# Wraps `rmedia feature text_embedding`.
make embed-chunks                   # ~25 min on a 5090 for 145k chunks
```

After this, the UI's Vector / Hybrid modes are live and `mode=semantic|hybrid`
work via the API.

### 4. Add visual search (optional)

Requires a GPU + the embedding server. Two stages:

```bash
make extract-chunk-frames          # ffmpeg, CPU-bound, ~30 min for 145k chunks
make embed-chunk-frames            # Qwen3-VL image embeddings → frame_embedding
                                   # (wraps `rmedia feature frame_embedding`)
```

Drag-drop an image onto the search bar to query frames (`mode=visual`); add text
as well and the request becomes `mode=all` (text + image + scene fusion).

### 5. Add scene search (optional)

Requires a generative VLM (Gemma 4) you run yourself on `:8003` (`CAPTION_URL`)
plus the embedding server. `raudio` is only a *client* of the caption model. Two
stages, both resumable:

```bash
make caption-chunk-frames          # POST frames → Gemma → chunk_frames.caption
                                   # (Swedish captions; wraps `rmedia feature caption`)
make embed-captions                # Qwen3-VL embeds captions → caption_embedding + IVF_PQ
                                   # (wraps `rmedia feature caption_embedding`)
# OR both in one go:
make captions
```

This lights up `mode=scene` (caption-vector) and `mode=scene_fts` (caption BM25),
and adds the scene leg to `mode=all`. Until captions are built, both scene modes
return empty. (On the live DB this is the genuinely-open piece —
`frame_embedding` is already built end-to-end and indexed.)

### 6. Add speaker diarization (optional)

Per-video **who-spoke-when** via `pyannote/speaker-diarization-community-1`. Runs
**offline, in-process** (no vLLM server, no isolated worker — `pyannote.audio`
already lives in the main venv), GPU-accelerated if available (~90 s/video) and
crash-resumable. Needs a cached HF token (`hf auth login`) and the community-1
model terms accepted on the Hub.

```bash
make speaker-turns                 # diarize each video → speaker_turns.lance
                                   # (wraps `rmedia extract-speaker-turns --audio-root input/sv`)
```

Writes a new `speaker_turns.lance` table (`doc_id, turn_id, speaker_label,
start, end` in absolute video seconds), separate from `chunks` for the same
reason as `chunk_frames`. It is append-only, one video at a time, and
`--only-null` (the default) skips already-diarized videos; pass `--all` to redo
everything. There is no embedding/vector column, so **no vector reindex is
needed**. Optional hygiene at corpus scale: a scalar BTREE index on
`speaker_turns.doc_id` speeds per-video lookup, and `rmedia compact --table
speaker_turns` consolidates the per-video append fragments.

**`rmedia serve` has no auto-reload — RESTART the backend after building** so it
opens `speaker_turns.lance` and serves `GET /api/diarization/{doc_id}`. The
player then shows a **Speakers** tab (per-speaker lanes + a playhead synced to
the video, click-to-seek); until the table is built it reads "Diarization not
built for this video." Labels (`SPEAKER_00…`) are anonymous and stable only
within one video.

### 7. Reranking (optional)

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
frontend (Bun :5274)  →  proxy  →  search :8102  (/api/search → search service)
search   →  chunks.search(MatchQuery("alkohol", "text")).select(...).limit(n)
                ↳ Tantivy BM25 index (Swedish stemmer); PhraseQuery when phrase=true
search   →  json hits  →  frontend renders list
```

### Semantic / hybrid query

```
browser  →  GET /api/search?q=klimat&mode=hybrid
backend  →  vLLM /v1/embeddings (Qwen-VL chat shape, system="Represent the user's input.")
              ↳ Qwen3-VL embedding (full 2048-d, no Matryoshka truncation), L2-normalized
backend  →  chunks.search(query_type="hybrid", vector_column_name="text_embedding")
                  .vector(text_vec).text(MatchQuery(q, "text"))
                  .rerank(RRFReranker() | LinearCombinationReranker(weight))
                  .nprobes(20).refine_factor(3).limit(n)
              ↳ Lance-native FTS + text-vector hybrid, fused by RRF (or the Balance slider)
              ↳ vector_column_name is required — chunks has TWO vector columns
                (text_embedding + a chunk-level frame_embedding)
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
# Use the populated dataset — ./transcripts.lance is empty on this machine.
DB=transcripts_v2.lance

# chunks table (text + metadata)
uv run python -c "
import lancedb
t = lancedb.connect('$DB').open_table('chunks')
print('chunks rows         :', t.count_rows())
print('text_embedding NULL :', t.count_rows('text_embedding IS NULL'))
"

# chunk_frames table (per-chunk JPEG + frame/caption embeddings)
uv run python -c "
import lance, pathlib
p = pathlib.Path('$DB/chunk_frames.lance')
if not p.exists():
    print('chunk_frames not yet created — run extract-chunk-frames first.')
else:
    ds = lance.dataset(str(p))
    print('chunk_frames rows   :', ds.count_rows())
    print('schema cols         :', ds.schema.names)
    print('has frame_embedding :', 'frame_embedding' in ds.schema.names)
    print('has caption         :', 'caption' in ds.schema.names)            # scene_fts
    print('has caption_embedding:', 'caption_embedding' in ds.schema.names)  # scene
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
| `GET /api/search?q=…&mode=fts\|semantic\|hybrid\|scene\|scene_fts&n=20` | Text search (query string) |
| `POST /api/search` (multipart, `image=…`, `mode=visual\|all`) | Image / cross-modal search |
| `GET /api/columns` | Filterable scalar columns of `chunks` (name + friendly type) |
| `GET /api/documents?page=1&per_page=24` | Paginated browse |
| `GET /api/thumbnail/{doc_id}` | Document thumbnail (Blob V2 Inline) |
| `GET /api/chunk-frame/{doc_id}/{speech_id}/{chunk_id}?frame_idx=N` | Chunk frame (Blob V2 Inline; `frame_idx=0` is the representative frame) |
| `GET /api/media/{doc_id}` | Stream the MP4 (Blob V2 External, HTTP Range supported) |
| `GET /api/media-clip/{doc_id}?lo=…&hi=…` | Windowed excerpt re-encoded to H.264+MP3 (ffmpeg, disk-cached) — sound inside AAC-less webview hosts like VS Code |
| `GET /api/atlas/status?space=text\|visual` | Which projection spaces are built + the space's projected row count |
| `GET /api/atlas/points?space=text\|visual` | Compact coord/cluster/language/namn arrays + doc-id keys for the scatter map |
| `GET /api/atlas/chunk/{doc_id}/{speech_id}/{chunk_id}` | Full hit for one chunk (lazy-fetched on hover/select) |
| `POST /api/atlas/chunks` (`{"keys": [[doc_id, speech_id, chunk_id], …]}`) | Full hits for a lasso/box selection (capped 1000 keys) |

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
- Feature builds are incremental: `rmedia feature <name>` defaults to
  `--only-null`, processing only rows where the target column is empty. Safe to
  Ctrl-C and resume; pass `--all` to drop and rebuild.

---

## Limitations

- **Image search is frame similarity, not face/identity recognition.** It finds
  visually similar video frames; it does not identify who is in them.
- **Speakers are anonymous per-video** — per-video speaker diarization exists
  (who-spoke-when; the **Speakers** tab in the player, `speaker_turns.lance`,
  `GET /api/diarization`), but labels (`SPEAKER_00…`) are stable only *within*
  one video. There is no cross-video speaker identity and no link between who is
  on screen and who is speaking.
- **The reranker is text-only** — it scores transcript text against the query
  and ignores the image and the vectors.
- **`all` fusion (up to 4 legs) is equal-weight RRF** — there is no per-leg
  weight yet (the Balance slider only applies to 2-way `hybrid`).

---

## Author

[Borg93](https://github.com/Borg93)
</content>
</invoke>
