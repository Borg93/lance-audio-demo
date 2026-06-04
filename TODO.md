# TODO

Living checklist for `raudio`. Update as items land.

> **How to read this:** ✅ done · ⏳ in progress · ❌ blocked · 📋 backlog.
> Each pending item points to the file(s) and command needed to pick it up.
>
> New here? Read [GUIDE.md](GUIDE.md) first (architecture, data flow, design
> rationale, dev workflow), then [README.md](README.md) for the quickstart.

> **Contents:** [Active blockers](#active-blockers-do-these-next) ·
> [Visual search wiring](#visual--cross-modal-search-wiring) ·
> [Future bets](#future--bigger-bets) ·
> [UX backlog](#ux-backlog) · [Hygiene](#cleanup--hygiene) ·
> [Code-quality backlog](#code-quality-backlog) ·
> [Search perf](#search-performance) · [vLLM perf](#vllm-performance) ·
> [Closed](#closed-for-context--commit-log)

---

## Active blockers (do these next)

**None.** Every retrieval leg is built and serving on the live DB: text/FTS,
semantic (`text_embedding`), **visual** (`frame_embedding`), and **scene**
(`caption` + `caption_embedding` — both keyword `scene_fts` and vector `scene`).
The **embedding atlas** and **topic modeling + Tree** are live too (see the ✅
sections below). Remaining work is the backlog + the
[future bets](#future--bigger-bets) (voice/speaker search, the Studio desktop merge).

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
fallback. The `caption_embedding` (scene) leg is **also live now** (captions
built — see [✅ Captions + scene search](#-captions--scene-search-done)).

### 🟡 (parked) `/api/health` could surface `chunk_frames` count + `has_embeddings`
Cosmetic for a single-user demo: `chunk_frames` already appears in
`health.db.tables`, and frame fetches self-heal via the 404 fallback
(`features.framesUnavailable`). ~6 LOC each side if ever wanted.

---

## ✅ Captions + scene search (done)

`chunk_frames` now carries `caption` (a Swedish caption of the representative
frame, generated from `frame_blob` via the **Gemma-4** server on `:8003`) and
`caption_embedding` (2048-d, embed server `:8001`) — both crash-resumable and
reusing the existing frames (never re-extracted). `scene` (cosine over
`caption_embedding`), `scene_fts` (BM25 over `caption`), and the caption leg of
`all` are live; captions also render in the search list, table, and player.
Built with `make captions` (`raudio feature caption` + `caption_embedding`).

---

## ✅ Embedding atlas + Topics + Tree (done)

### ✅ Embedding atlas (`/atlas`)
EVōC 2-D projection of `chunks` (`atlas_x/y` + `atlas_cluster`) over
`text_embedding`, plus `--space visual` (`frame_embedding` → `atlas_img_*`) and
`--space caption` (`caption_embedding` → `atlas_cap_*`). Rendered by a custom
**WebGPU** instanced-quad scatter (no WebGL2/Canvas2D), with lasso/box/legend
selection cross-filtered to Search. `raudio feature atlas [--space …]`, served
via `/api/atlas/*`.

### ✅ Topic modeling + Tree page (`/tree`)
`raudio feature topics` clusters chunks with Toponymy in an **isolated PEP723
worker** (so the main env never depends on `transformers<5`): per-chunk
`topic_l0/l1/l2` + per-video `doc_topic` (BITMAP-indexed), and the nested
hierarchy stored as Lance **JSONB** in `topics.lance`. The Tree page is a
LayerChart `<Treemap>` (Flat + Nested views, hide-noise toggle, drill-down) that
hands a topic to Search via `/?topic=`. Backed by `GET /api/topics` + a `topic=`
filter on `/api/search` (matches any `topic_l*` layer via the shared
`topic_layer_columns`). Caveat: ~64% of chunks are HDBSCAN noise at the finest
layer — retune `base-min-cluster-size` / layers later if the long tail matters.

### ✅ Atlas topic colours
The atlas can colour points by **Topic** (broadest `topic_l*` layer) or **Video
topic** (`doc_topic`), with a clickable named legend that selects → table → seed
search. Implemented as one DRY categorical channel shared with the language
colouring.

---

## Future / bigger bets

### 📋 Video-level concatenated text + summarization
Today there is **no per-video full text and no video-level summary** — a "video"
exists only as its ~125 scattered `chunks` rows. (`FEATURES["summary"]` exists but
is *chunk-level* and not even built; `documents` carries metadata/blobs but no
transcript.) Two clean per-video columns on the `documents` table, same
roll-up-by-`doc_id` pattern as `doc_topic`:
- **`documents.full_text`** — concatenate each video's chunk `text` ordered by
  `start`, grouped by `doc_id`. **Pure aggregation, no model**; enables full-video
  FTS and feeds the summary. (`FEATURES["doc_text"]`, `table="documents"`.)
- **`documents.doc_summary`** — an LLM summary of `full_text` (instruct LLM `:8004`
  or long-context Gemma-4 `:8003`). Likely needs a **map-reduce / hierarchical**
  pass (per-chunk summaries → reduce) since a full press conference overflows one
  prompt — the chunk-level `summary` feature is the natural first stage.
  (`FEATURES["doc_summary"]`, `table="documents"`.)

### 📋 Voice / speaker search (ECAPA x-vectors)
Add an **audio** embedding axis: a per-chunk 1024-d speaker embedding so you can
upload a voice clip and find chunks where a similar-sounding voice speaks (and
cluster the corpus *by speaker*). Encoder:
`marksverdhei/Qwen3-Voice-Embedding-12Hz-0.6B` — a standalone ECAPA-TDNN x-vector
extracted from Qwen3-TTS-0.6B-**Base** (Apache-2.0, ~6.3M params, 24 kHz mono →
1024-d). Maps cleanly onto the existing machinery:

- **Encoder** — a *local in-process* HF model (lives with `asr/`, **not**
  `vllm/`); run it in an isolated env / a `voice` extra (`trust_remote_code` + a
  possibly-pinned `transformers`, same discipline as the topics worker).
- **Audio slicing** — extend `media/frames.py`'s ffmpeg fast-seek to pull each
  chunk's `[start,end]` as 24 kHz mono wav (the audio analog of the JPEG frame
  extractor).
- **Feature** — `FEATURES["speaker_embedding"]`: slice → encode → write a 1024-d
  column + IVF (cosine) index, crash-resumable like the caption build.
- **Search** — a `voice` mode: multipart **audio upload** → encode → vector
  search over `speaker_embedding` (mirrors the `visual` image-upload path); plus
  a "more of this voice" action on a hit using its stored vector.
- **Atlas** — a `--space voice` projection → speaker clusters spatially.

Caveats: x-vectors assume one speaker per clip (overlap/applause muddy them);
need ~1.5–3 s+ of audio per clip; normalize → cosine. **De-risk first** by
loading the model in isolation and confirming same- vs different-speaker cosine
separates, before wiring the pipeline.

### 📋 Studio desktop merge (ranymizer + raudio + multimodal-webgpu-demo → Tauri)
Fold the three SvelteKit apps into one Tauri 2 **"Studio"** shell, where raudio
becomes the server-backed **Search sandbox**. Full architecture, contracts, and a
phased roadmap live in **[docs/STUDIO_MERGE.md](docs/STUDIO_MERGE.md)** (moved out
of the repo root). Not started; the biggest piece is the shell + sandbox registry
(phase P0).

---

## UX backlog

Audited 2026-06 against the code — mostly done or YAGNI for a single-user demo:

- ✅ **Hit-card shows the exact per-chunk frame** (`chunkFrameUrl`, gated by
  `features.framesUnavailable`; the doc thumbnail is just the poster behind it) — done.
- 🟡 **Persist filters across a hard reload** — dropped: no deep-link/sharing need,
  filters already auto-apply, and a re-search is cheap.
- 🟡 **Debounce / auto-search the text box** — dropped: each query hits vLLM + Lance
  over 145k rows, so Enter-to-search is correct; the cheap controls (filters,
  settings) already auto-apply.

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

### ✅ Removed stray `images_per.jpg` + fixed the Makefile help text (done)
Deleted the untracked repo-root test image, and corrected the
`extract-chunk-frames` `##` help to say it writes into the `chunk_frames` table.

---

## Search performance

The worthwhile work here is **✅ done** (recall knobs, read-path caching, scalar
indexes, compaction — below). Everything still marked 📋 was **audited 2026-06 and
is parked / YAGNI for a single-user local demo**: sub-millisecond serialization
wins (`alignments_json`), exact-repeat-only gains (query-vector LRU), or
concurrency/rare-mode-only payoffs (parallel `all` legs, `IVF_HNSW_SQ`). Revisit
only if a profiler or real concurrency makes them bite. The benchmarking recipe at
the end is kept as reference.

### ✅ IVF_PQ recall knobs (`nprobes` + `refine_factor`) (done)
All vector legs (`semantic`, the hybrid vector leg, and `_frame_search`) pass
`nprobes=20` + `refine_factor=3` (`_VECTOR_NPROBES` / `_VECTOR_REFINE_FACTOR` in
`backend/search/service.py`). This was the "feels broken / re-query reflex"
recall fix from INVESTIGATION §A3.

### ✅ Read-path caching: memoize `/atlas/points`, shared handle, startup warmup (done)
- **`GET /atlas/points` memoized** on `(space, dataset version)` — the 145k-row
  scan+factorize runs once per version, not per request (`backend/atlas/router.py`,
  `_POINTS_CACHE`). ~3× on repeat (0.59s → 0.19s warm).
- **`chunks` dataset handle cached** on `AppState` (`chunks_ds = chunks.to_lance()`
  once at startup; `backend/state.py`) — avoids re-seeding Lance's metadata/index
  cache per request; reused by every read path.
- **Startup warmup** (`backend/warmup.py` + the app lifespan) preloads every
  IVF/scalar/FTS index + the atlas-points payload so the first request isn't cold
  (`cache warmup done in ~1.4s`).

### ✅ Scalar indexes on the equality-filter columns (done)
`doc_id` + `audio_path` (chunks) and `doc_id` (chunk_frames) back the per-row
lookups (caption-attach + frame joins); **`extraid`** (chunks) was added for the
selective archival-id facet — `analyze_plan` confirms it pushes into a
`ScalarIndexQuery(extraid_idx)`. **`language` is deliberately NOT indexed** (corpus
is 100% `sv`, so an index prunes nothing); `namn`/`referenskod` are filtered with
`LIKE '%…%'`, which a BTREE can't accelerate. Built in `ingest.py`, rebuilt by
`raudio compact`. Verified `num_unindexed_rows=0` on every index (`index_stats`).

### ✅ `compact` keeps scalar indexes + honors `TABLE` (done)
`raudio compact` rebuilds the BTREE scalar indexes after `compact_files`
(compaction otherwise invalidates the row addresses they point at) and now compacts
`--table` (`TABLE=chunk_frames` for the frames table), not just `chunks`
(`src/raudio/cli/media.py`).

### 📋 Prune old dataset versions (disk, not latency)
`chunk_frames` retains **80** versions (5.1 G), `chunks` **21** (4.2 G) — every
feature/index write left an old version behind. `ds.cleanup_old_versions(older_than=…)`
reclaims the unreferenced files. **Irreversible** (drops time-travel/rollback) — run
with a conservative retention window after confirming no history is needed.

### 📋 Considered, marginal at this scale (145k rows, single local node)
Shared `lance.Session` across the 3 table handles (caches are per-table; modest
memory, not latency), `LANCE_IO_THREADS`/`LANCE_CPU_THREADS` bumps (local default
8 IO threads is fine for small projected scans), and `IVF_HNSW_SQ`/`IVF_RQ` over
the current `IVF_PQ`. Compacting `chunk_frames` (73 frags) is low-value — the frame
reads are index-driven, not scan-bound.

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

### ✅ Cross-encoder rerank defaults off in the UI (done)
`search-settings.svelte` declares `rerank = $bindable(false)` (off) as a labelled
Switch ("Rerank results"); `SearchSpec.rerank` defaults `False` and `rerank_n` is
only sent when rerank is on. Verified — nothing to do.

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

⚪ **Mostly ops/diagnostics + parked stretch items — not active code TODOs.**
`--enable-prefix-caching` is already set in the Makefile; the `/metrics` recipe,
the GPU-budget note, and the prefix-caching check are *diagnostic/ops references*;
the async client, FP8, and `--async-scheduling` are concurrency/throughput wins
that are **YAGNI at single-user scale**. Kept as context for if/when that changes:

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
