# TODO

Living checklist for `raudio`. Update as items land.

> **How to read this:** ✅ done · ⏳ in progress · ❌ blocked · 📋 backlog.
> Each pending item points to the file(s) and command needed to pick it up.
>
> New here? Read [GUIDE.md](GUIDE.md) first (architecture, data flow, design
> rationale, dev workflow), then [README.md](README.md) for the quickstart.

> **Contents:** [Active blockers](#active-blockers-do-these-next) ·
> [Visual search wiring](#visual--cross-modal-search-wiring) ·
> [UX backlog](#ux-backlog) · [Hygiene](#cleanup--hygiene) ·
> [Code-quality backlog](#code-quality-backlog) ·
> [Search perf](#search-performance) · [vLLM perf](#vllm-performance) ·
> [Closed](#closed-for-context--commit-log)

---

## Active blockers (do these next)

The text/FTS, semantic (`text_embedding`), and **visual** (`frame_embedding`)
pipelines are all done and serving on the live DB. The one cross-modal leg still
returning empty is **scene** (`caption_embedding`) — the frames have no Swedish
captions yet. See [Open work: captions](#open-work-captions-scene-search).

### ✅ 1. `extract-chunk-frames` ran end to end (RESOLVED)

Architecture writes per-chunk frames into a separate append-only
`chunk_frames.lance` table (commit `3954ee5`) — never `merge_insert` into the
wide `chunks` schema (see [docs/INVESTIGATION.md](docs/INVESTIGATION.md) §A1).
The code path (`raudio extract-chunk-frames`, `src/raudio/cli/media.py`) ran on
the full corpus: the live DB's `chunk_frames` table holds **145,175 rows**
(`doc_id, speech_id, chunk_id, frame_idx, frame_blob, frame_mime, frame_width,
frame_height, frame_embedding`).

```bash
make extract-chunk-frames EXTRACT_JOBS=24
```

Verify on the live DB:

```bash
uv run python -c "
import lance
ds = lance.dataset('./transcripts_v2.lance/chunk_frames.lance')
print('rows:', ds.count_rows(), '— cols:', ds.schema.names)
"
```

### ✅ 2. `embed-chunk-frames` built `frame_embedding` end to end (RESOLVED)

Frame embedding is built via Lance data evolution — `raudio feature
frame_embedding` (wrapped by `make embed-chunk-frames`), which calls
`add_columns(...)` to write `chunk_frames.frame_embedding` and then builds the
cosine IVF index. This **completed on the live DB**: `chunk_frames.frame_embedding`
is **145,175 / 145,175 rows populated (zero nulls)** and indexed by the IVF index
`frame_embedding_idx`. The earlier vLLM Qwen3-VL image-embed crash (deepstack
buffer mismatch) is no longer blocking — the image-embed path produced the full
column.

Current pin is `vllm==0.22.0` / `Qwen3-VL-Embedding-2B` with the server pixel
budget pinned via `--mm-processor-kwargs '{"min_pixels": 153664, "max_pixels":
153664}'` (Makefile), matched by the client-side crop (`_IMAGE_SIDE` in
`src/raudio/vllm/image.py`) — see [docs/INVESTIGATION.md](docs/INVESTIGATION.md)
Part B for the history.

`visual` and the frame leg of `all` are now live: `backend/search/service.py::
_frame_search` queries `chunk_frames.frame_embedding` and joins back to `chunks`
by `(doc_id, speech_id, chunk_id)`.

---

## Visual / cross-modal search wiring

### ✅ Backend visual-search path reads `chunk_frames` (done)

`backend/search/service.py::_frame_search` runs the `visual` / `all` frame leg
against `chunk_frames.frame_embedding`, then joins back to `chunks` by
`(doc_id, speech_id, chunk_id)` for text/timestamps/metadata, preserving the
frame-distance ranking. `/api/chunk-frame` reads only the `chunk_frames` table.

**Live end to end:** with `chunk_frames.frame_embedding` populated + indexed
(145,175 rows), the visual happy path is exercised — not just the graceful-empty
fallback. The `caption_embedding` (scene) leg still degrades to `[]` until
captions are built (see [Open work: captions](#open-work-captions-scene-search)).

### 📋 `/api/health` should report `chunk_frames` state

`/api/health` (`backend/system/router.py::health`) currently reports DB path,
table list, and `chunks` / `documents` row counts, plus embed/rerank pings.
`backend/state.py` already computes `has_embeddings` for `chunk_frames` (only
logs it). Surface a `chunk_frames` row count + `has_embeddings` boolean in the
health payload and render it in `frontend/src/lib/components/status-badge.svelte`
(which today shows only tables/chunks/documents). ~6 LOC each side.

---

## Open work: captions (scene search)

The genuinely-missing cross-modal piece. The `scene` and `scene_fts` search
modes are wired in code but return **empty** on the live DB because
`chunk_frames` has no `caption` / `caption_embedding` column yet — its columns
are `doc_id, speech_id, chunk_id, frame_idx, frame_blob, frame_mime,
frame_width, frame_height, frame_embedding`. The frame JPEGs and their image
vectors (`frame_embedding`) already exist; what's left is to *describe* each
frame in Swedish and embed that description.

### 📋 Run `make captions` to build `caption` + `caption_embedding`
Two-stage, resumable, reuses the existing frames (never re-extracts):

```bash
make captions   # = caption-chunk-frames + embed-captions
```

- **`caption-chunk-frames`** (`raudio feature caption`) generates a Swedish
  caption per frame from `frame_blob` into `chunk_frames.caption`. **Needs the
  Gemma server (a generative VLM, Gemma 4) running on `:8003`** (`CAPTION_URL` in
  the Makefile) — *not* the embed/rerank servers.
- **`embed-captions`** (`raudio feature caption_embedding`) embeds
  `chunk_frames.caption` into the shared 2048-d space as `caption_embedding`
  (+ IVF index) using the embed server on `:8001`.

Once both columns exist, `scene` (cosine over `caption_embedding`), `scene_fts`
(BM25 over `caption`), and the caption leg of `all` light up with no further
backend changes — `backend/search/service.py::_frame_search` already queries
`caption_embedding` and the `scene_fts` BM25 path is in place.

---

## UX backlog

### 📋 Persist active filters across a hard reload
Active filters live in component state only; a full page reload drops them.
`active-filters.svelte` renders removable pills (`namn`, `referenskod`,
`extraid`, `language`, raw `where`) and is correct — it does *not* wipe state on
mount. A nicety: mirror those fields to `localStorage` (or the URL query string)
so the last filter set survives a refresh.

### 📋 Hit-card thumbnail: use the exact frame instead of the doc thumbnail
`hit-card.svelte` renders the doc thumbnail (`/api/thumbnail/{doc_id}`). Once
`chunk_frames` is populated we have the exact frame per hit
(`/api/chunk-frame/{doc_id}/{speech_id}/{chunk_id}`) — switch search-mode hits
to that and drop the extra request per card.

### 📋 Debounce + auto-search on the search input
`search-bar.svelte` submits on Enter / button. For fast typers an optional
~300 ms debounced auto-search on non-empty input would feel snappier. Combine
with the existing loading state in `+page.svelte`.

---

## Cleanup / hygiene

### ✅ No dead `frame_*` columns on `chunks`
The `chunks` table carries exactly **two** vector columns and both are live:
`text_embedding` (transcript text, indexed `text_embedding_idx`) and
`frame_embedding` (a CHUNK-LEVEL 2048-d image vector that backs the visual
image-atlas — populated on the live DB, *not* dead). The per-frame blob/mime/
size data lives in the separate `chunk_frames` table, not on `chunks`. There is
nothing to drop here — do **not** run `drop_columns(["frame_embedding", ...])`
against `chunks`, it would delete the visual-atlas vector.

### 📋 `make compact` after multi-stage writes
`extract-chunk-frames` lands many small fragments and `feature frame_embedding`
adds a column. Run `make compact` (or `raudio compact`) afterward to consolidate
fragments and rebuild the IVF_PQ indexes. Optional at this dataset size — just
slightly faster scans.

### 📋 Resolve the stray `images_per.jpg`
Test image in the repo root (gitignored). Delete it or move it to
`tests/fixtures/` for use in unit tests.

### 📋 Fix stale Makefile help text
The `extract-chunk-frames` target's `##` help still reads "into
`chunks.frame_blob`" — frames now go into the `chunk_frames` table. One-line fix.

---

## Search performance

Vector / multimodal search is acceptable but not free. Items ordered roughly by
impact-per-effort.

### ✅ IVF_PQ recall knobs (`nprobes` + `refine_factor`) (done)
All vector legs (`semantic`, the hybrid vector leg, and `_frame_search`) pass
`nprobes=20` + `refine_factor=3` (`_VECTOR_NPROBES` / `_VECTOR_REFINE_FACTOR` in
`backend/search/service.py`). This was the "feels broken / re-query reflex"
recall fix from INVESTIGATION §A3.

### 📋 Stop fetching `alignments_json` in the search projection
`alignments_json` is a multi-KB JSONB blob per chunk that the result list never
renders (only the player pane needs it). The hit projection (`_HIT_COLUMNS` /
`_PAYLOAD_COLUMNS` in `service.py`) currently includes it. Drop it from the
search projection and add a `GET /api/chunk-alignments/{doc_id}/{speech_id}/
{chunk_id}` endpoint that returns it on demand (the player pane already re-fetches
per hit). Estimated 30–60% win on result-set serialization for large `all`/hybrid
queries.

### 📋 Query-vector LRU cache
The vLLM client is cached on `app.state` (`backend/clients.py`) ✅, but every
query still pays the embed round-trip. Add an `lru_cache(maxsize=512)` on
`embed_text(query)` keyed by the exact query string so repeated searches (same
query, different filters) skip the ~50 ms vLLM RTT. Images stay uncached (each
upload is unique).

### 📋 Run the legs of `hybrid` / `all` concurrently
`mode=all` issues up to four legs sequentially before RRF (`run_search`'s `all`
branch): FTS (`text`) + text-vector (`text_embedding`) + frame-vector
(`frame_embedding`; the image vec if an image is attached, else the text vec) +
caption/scene-vector (`caption_embedding`). `hybrid` is fused natively by Lance
so it's already one call. The `all` legs are independent and unioned by chunk via
`_rrf_fuse` (k=60, 0-indexed rank, score = sum of `1/(60+rank)`) — overlap them
with `asyncio.gather` + `run_in_executor` (Lance is sync). Pairs with the
async-client item under [vLLM perf](#vllm-performance).

### 📋 Default the cross-encoder rerank off in the UI
The cross-encoder (`Qwen3-VL-Reranker-2B`) adds ~200–500 ms when toggled on.
`SearchSpec.rerank` already defaults `False` and the rerank window is bounded to
the top `rerank_n` (default 20) — verify the Settings popover
(`search-settings.svelte`) defaults the toggle off and labels it "best quality,
slower."

### 📋 (Stretch) Try `IVF_HNSW_SQ` for the frame-embedding index
Better recall at the cost of memory; might let `nprobes` stay low and end up
faster overall on `frame_embedding`. Now actionable — `frame_embedding` is built
and indexed (currently IVF_PQ `frame_embedding_idx`); worth a one-shot benchmark
against an `IVF_HNSW_SQ` rebuild.

```python
ds.create_index("frame_embedding", index_type="IVF_HNSW_SQ",
                num_partitions=256, replace=True)
```

### Quick benchmarking recipe

```bash
uv run python -c "
import time, lancedb, numpy as np
t = lancedb.connect('./transcripts_v2.lance').open_table('chunks')  # populated DB (default ./transcripts.lance is empty)
q = np.random.randn(2048).astype('float32')
t.query().nearest_to(q).limit(20).to_list()  # warmup
n, total = 50, 0
for _ in range(n):
    s = time.perf_counter()
    t.query().nearest_to(q).distance_type('cosine').nprobes(20).refine_factor(3).limit(20).to_list()
    total += time.perf_counter() - s
print(f'avg {total/n*1000:.1f} ms / query')
"
```

---

## vLLM performance

A single `POST /v1/embeddings` against `Qwen3-VL-Embedding-2B` takes ~100–300 ms
(plus ~5 ms localhost RTT). A hybrid search fires it once per query, so combined
with the index scan it's the bulk of the visible latency.

### 📋 Make the per-query vLLM client async
`POST /api/search` already offloads the blocking work via `run_in_threadpool`
(`backend/search/router.py`), so it no longer stalls the event loop. The next
step is a true async embed call: switch `VLLMEmbeddingClient`'s per-query path
(`src/raudio/vllm/embedding.py`) to `httpx.AsyncClient` and await it at the
event-loop level, freeing the worker for other connections. Keep the
`ThreadPoolExecutor` batch path for the CLI `feature` (batch-embed) case.

### 📋 Confirm `--enable-prefix-caching` is active
The embed chat-template sends the same system instruction every query, so prefix
caching reuses the KV cache for that prefix (~10 ms/call). Default-on in modern
vLLM — confirm in the startup log:

```
INFO  …  enable_prefix_caching=True  …
```

### 📋 Use vLLM's `/metrics` to find the real bottleneck

```bash
curl -s http://127.0.0.1:8001/metrics | grep -E "vllm_(time_to_first_token|e2e_request_latency|gpu_cache_usage)"
```

- `vllm_e2e_request_latency_seconds_*` — end-to-end per request
- `vllm_time_to_first_token_seconds_*` — TTFT (KV-cache miss dominated)
- `vllm_gpu_cache_usage_perc` — > 0 if prefix caching helps
- `vllm_request_queue_time_seconds_*` — ~0 for a single-user demo

### 📋 Watch the shared-GPU memory budget (both servers co-locate)
By default **both** servers run on the **same** GPU: `VLLM_GPU ?= 0`,
`EMBED_GPU ?= $(VLLM_GPU)`, `RERANK_GPU ?= $(VLLM_GPU)`, each capped at
`EMBED_MEM_FRAC ?= 0.45` / `RERANK_MEM_FRAC ?= 0.45` (both 2B models fit a 96 GB
card, ~88 GB combined). Start them **sequentially** to avoid the memory-profiling
race. Confirm the split (or move one to another GPU via `EMBED_GPU=` /
`RERANK_GPU=`) with:

```bash
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
```

### 📋 (Stretch) FP8 quantization
If/when an FP8 checkpoint of `Qwen3-VL-Embedding-2B` ships, swap it in and pass
`--quantization fp8` — roughly 2× throughput on sm_120 vs bf16, no quality loss
for pooled-embedding use. Not actionable today.

### 📋 (Stretch) `--async-scheduling`
Overlaps scheduling overhead with decoding. Off in the current container start;
try it after the items above are validated (it can interact poorly with
structured output / sampling penalties, neither of which we use).

---

## Backup plan (moot — the Lance frame path succeeded)

This was the fallback if the Lance/vLLM frame path kept misbehaving: store frames
on plain disk at `./frames/{doc_id}/{speech_id}-{chunk_id}.jpg`, serve via
`FileResponse`, and put embeddings in a separate `chunk_frame_embeddings.lance`
table. **No longer needed** — `chunk_frames.frame_embedding` built end to end in
Lance (145,175 rows, indexed), so frames live in the `chunk_frames` table as
designed. Kept here only as historical context.

---

## Code-quality backlog

Larger or behavior-shifting changes intentionally deferred from the skills-audit
pass (the safe mechanical wins are in [Closed](#closed-for-context--commit-log)).

### Backend (`backend/`)

#### 📋 Domain-error hierarchy + exception handlers instead of inline `HTTPException`
Search/blob paths catch broad `Exception` and translate to `HTTPException`
inline (`backend/search/service.py`, `backend/media/router.py`). A small
`DomainError` hierarchy + registered handlers (per the fastapi skill) would
centralize this. ⚠️ Do **not** narrow the `health()` ping catch to
`except httpx.HTTPError` — `httpx` is lazy-imported there behind the
`[multimodal]` extra, so an `ImportError` would escape as a 500. Keep it broad.

#### 📋 CORS `allow_origins=["*"]` is hard-coded
Fine for the local demo (API-only behind the Bun proxy); tighten via settings if
ever exposed.

### CLI / Python (`src/raudio/`)

#### 📋 FTS-language defaults disagree (latent correctness bug on a Swedish corpus)
`ingest` defaults `--fts-language English` (`src/raudio/cli/ingest.py:75`) but
`reindex-fts` defaults `Swedish` (`:143`). On this Swedish corpus the English
stemmer silently mis-stems inflected forms (`ministern`, `vägen`, `ansåg`). Pick
one default (almost certainly `Swedish`) so a plain `raudio ingest` produces a
usable index without the flag.

#### 📋 `_Ctx` global state → Typer's context object
`src/raudio/cli/_app.py` shares `--db`/`--table` via mutable class attributes on
`_Ctx`. The idiomatic pattern is `ctx.obj` / `typer.Context`. Low-risk but
touches command signatures across the `cli/` package.

#### 📋 `print()` → `logging` in library code
`media/thumbnails.py`, `media/download.py`, `asr/detect_language.py` print
progress directly. Library modules should log (or return data) and let the CLI
render — matters when the backend imports them.

#### 📋 Minor typing / dedup
`frames._extract_one` takes an untyped `args: tuple`; `detect_language` probe
closures lack annotations; the reranker prefix/suffix constants in
`vllm/reranker.py` and `retrieval/qwen3_vl_reranker.jinja` want a one-line
cross-reference comment (they must stay in sync). All low priority.

### Frontend & demo

#### 📋 Add eslint + prettier to `frontend/` and `demo/`
`svelte-check` + `tsc` are the type gate today; eslint/prettier would round out
the `writing-typescript` toolchain.

#### 📋 `demo/`: type the Web Worker message protocol
`worker.ts` ↔ `+page.svelte`/`RealtimePanel`/`BatchPanel` share one
`postMessage` channel disambiguated only by an implicit `jobId` convention. A
shared discriminated-union message type would remove the ad-hoc `as` casts.
(Secondary app — low urgency.)

#### 📋 `demo/`: tighten `tsconfig`, drop unused dep, decide the `/search` stub
Add `noUncheckedIndexedAccess` + `exactOptionalPropertyTypes` (as in
`frontend/`); `d3-scale` looks unused (remove from `package.json` + reinstall);
the `/search` route is a permanent `WorkInProgress` placeholder — build it or
remove route + sidebar entry. Also: `formatBytes` is duplicated across
`ProgressItem`, `TranscriptHistory`, and `BatchPanel` — extract one helper.

---

## Closed (for context / commit log)

### ✅ Architecture / domain-package refactor
`src/raudio/` split into domain packages: `model/` (schemas + DTOs), `asr/`
(local Whisper/wav2vec2 stage), `ingest/`, `media/` (ffmpeg frames), `vllm/`
(out-of-process model-server clients: `embedding`, `reranker`, `caption`,
`summarize`, `base`, `image`), `features/` (`engine.py` + the `columns.py`
`FEATURES` registry), `retrieval/` (`search.py`), and a `cli/` package (one
module per command group). The 944-line `cli.py` split landed here. The backend
likewise split into `backend/{search,media,system}/` routers + a framework-free
`run_search` core wired through DI (`deps.py`, `state.py`, `clients.py`).

### ✅ Search UX + configurable retrieval (done)
- Mode `Select` + query + Search toolbar; Hybrid reveals a second "Vector" box
  (`q_vec`, a distinct query string for the vector leg, falling back to `q`).
- ⚙ Settings popover (`search-settings.svelte`): "Results to return" (`n`,
  1–200, default 20), "Rerank top" (`rerank_n`) + rerank toggle, Balance slider
  (hybrid-only `weight`; unset → RRF), Match style (phrase / fuzziness).
- Filters popover (`filter-popover.svelte`): structured column·operator·value
  builder that auto-applies, column show/hide, and a raw-SQL `where` advanced
  field — backed by `GET /api/columns` (filterable scalar columns of `chunks`).
- Results as list / grid / table; the table has a thumbnail, query highlight, and
  a column chooser.
- Help (?) popover with an Examples tab and a "How search works" flow diagram
  (`help-popover.svelte`, `search-flow.svelte`).
- Cross-encoder rerank applies to `fts` / `semantic` / `hybrid` / `all` (text-
  only, on the top-`rerank_n` head); a no-op for image-only `visual`.
- Dark-mode + `Select` fixes; `SearchSpec` is a Pydantic v2 model (unknown
  `mode` → 422 at the route boundary).

### ✅ Skills-audit cleanup pass
Applied `writing-python` / `fastapi` / `svelte` / `writing-typescript`. Gates
green: ruff, ty, `pytest`, `raudio --help`, frontend `bun run check` + build.
- **Tooling:** `[tool.ruff]`, `[tool.ty]`, `[tool.pytest.ini_options]`, and a
  `dev` dependency group in `pyproject.toml`.
- **Tests:** `tests/test_units.py` (timecode, `_build_where_clause`, `_rrf_fuse`,
  query-term extraction) + `tests/test_backend_smoke.py` (dataset-gated
  end-to-end: FTS, health, documents, thumbnail, Range streaming, 503).
- **Bugs fixed:** `typer.Exit("msg")` passing a string as the exit code → `_die()`.
- **FastAPI:** `POST /api/search` offloads blocking vLLM/Lance via
  `run_in_threadpool`; embedding client cached on `app.state`; `db.list_tables()`
  replaces the deprecated `table_names()`.
- Dead UI components removed (Card / Badge / Checkbox); demo `AudioVisualizer`
  removed; `noUncheckedIndexedAccess` + `exactOptionalPropertyTypes` enabled in
  `frontend/`.
- Added [GUIDE.md](GUIDE.md) and the `docs/` set (INVESTIGATION, EMBEDDINGS,
  PIPELINE, STORAGE, TESTING).

### Earlier
- ✅ Frontend migrated to SvelteKit + Tailwind v4 + Bun proxy; secondary `demo/`
  app (transformers.js audio) added.
- ✅ Status badge reports vLLM embed/rerank reachability + Lance dataset facts.
- ✅ `make compact` target + `raudio compact` command.
- ✅ `make embed-chunks` (`raudio feature text_embedding`) ran clean
  (~145 k chunks, IVF_PQ built).
- ✅ Schema redesign: `chunk_frames` as a separate Lance table (commit `3954ee5`).
</content>
</invoke>
