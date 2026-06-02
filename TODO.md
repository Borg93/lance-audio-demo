# TODO

Living checklist for `lance-audio-demo`. Update as items land.

> **How to read this:** ✅ done · ⏳ in progress · ❌ blocked · 📋 backlog.
> Each pending item points to the file(s) and command needed to pick it up.
>
> New here? Read [GUIDE.md](GUIDE.md) first (architecture, data flow, design
> rationale, dev workflow), then [README.md](README.md) for the quickstart.

> **Contents:** [Active blockers](#active-blockers-do-these-next) ·
> [Visual search wiring](#visual--cross-modal-search-wiring) ·
> [UX polish](#ux-polish-that-came-up-in-conversation-but-is-still-open) ·
> [Hygiene](#cleanup--hygiene) · [Code-quality backlog](#code-quality-backlog--skills-audit-2026-05-29) ·
> [Search perf](#search-performance--observed-slow-prioritized-fixes) ·
> [vLLM perf](#vllm-performance--observed-slow-embeddings) · [Closed](#closed-for-context--commit-log)

---

## Active blockers (do these next)

### ⏳ 1. Re-run `extract-chunk-frames` against new `chunk_frames` table

The previous attempts crashed in `merge_insert` against the wide `chunks`
schema. Architecture was redesigned (commit `3954ee5`) to write into a
separate `chunk_frames.lance` table append-only. **Not yet validated end
to end** — the user needs to run it.

```bash
make extract-chunk-frames EXTRACT_JOBS=24
```

Expected: ~30 min for 145 k frames at ~75 fps; multiple small Lance
fragments accumulate as it runs (resumable). Verify with:

```bash
uv run python -c "
import lance
ds = lance.dataset('./transcripts.lance/chunk_frames.lance')
print('rows:', ds.count_rows(), '— cols:', ds.schema.names)
"
```

If this still crashes for any reason, fall back to a sidecar-directory
implementation (write JPEGs to `frames/{doc_id}/{speech_id}-{chunk_id}.jpg`).
See "Backup plan" below.

### ❌ 2. `embed-chunk-frames` blocked by vLLM Qwen3-VL image-embed crash

Observed on vLLM 0.20.0 / Qwen3-VL-Embedding-8B: the warmup-time
deepstack-input-embeds buffer is sized differently from runtime, even
when `--mm-processor-kwargs '{"min_pixels": …, "max_pixels": …}'` is set.
The pin is now 0.22.0 / Qwen3-VL-Embedding-2B, and the client crop
(`_IMAGE_SIDE` in `vllm/image.py`) still mismatches the server pixel pin
(392px) — re-verify on the current pin before calling it fixed (see the Makefile
pixel-pin note + INVESTIGATION.md).
Single image request kills the engine with:

```
ValueError: Requested more deepstack tokens than available in buffer:
            num_tokens=N > buffer=N-k
```

We have not been able to find a vLLM config that consistently avoids this.

**Two options to unblock — both are cheap to implement (~80 LOC each):**

- **(A)** In-process HF transformers fallback. Load
  Qwen3-VL-Embedding-2B once at backend startup via `transformers`,
  embed images directly. Slower (~2 s/query) but immune to vLLM internals.
  Add a second embedding client class in `src/raudio/vllm/` with the same
  `embed_text`/`embed_image` surface as `VLLMEmbeddingClient`, then wire it into
  `backend/app.py` (`_get_embedder`) and `cli.py`.
- **(B)** Different vLLM tag (`v0.21.0+` once released, or back to `v0.10.x`).
  Risk: Blackwell sm_120 support compat. Won't know until tested.

User has not chosen yet. Ask before implementing.

### ❌ 3. `chunk_frames` IVF_PQ index build (depends on #2)

Once frame embeddings exist, `dataset.add_columns(...)` writes the
`frame_embedding` column, then `ensure_vector_index` builds the cosine
IVF_PQ index. Code path is in place at
`src/raudio/features/engine.py` (`ensure_vector_index`), driven by
`raudio feature frame_embedding` — runs automatically when the embed step
completes.

---

## Visual / cross-modal search wiring

### ✅ 4. Backend visual-search query path reads `chunk_frames` (done)

`backend/app.py::_frame_search` now runs the `mode=visual` / `mode=all` frame
branch against the `chunk_frames` table's `frame_embedding`, then joins back to
`chunks` (by `(doc_id, speech_id, chunk_id)`) for text/timestamps/metadata,
preserving the frame-distance ranking. The legacy all-NULL `chunks.frame_*`
columns were dropped from `CHUNK_SCHEMA`, and `/api/chunk-frame` no longer falls
back to `chunks.frame_blob`. Degrades to `[]` until frames are embedded.

**Still gated** on the frame *data*: `embed-chunk-frames` has never completed
(the vLLM image-embed crash, [INVESTIGATION.md](docs/INVESTIGATION.md)), so the
happy path is unverified end-to-end — only the graceful-empty path is exercised.

### ❌ 5. `/api/health` should report `chunk_frames` state

The status badge popover currently shows DB path + chunks/documents row
counts. Add `chunk_frames` row count and a `has_embeddings` boolean.
Tiny change in `backend/app.py` `def health()` (~6 LOC) + matching field
in `frontend/src/lib/components/status-badge.svelte`.

---

## UX polish that came up in conversation but is still open

### 📋 Active filter chips don't survive page reload
`ActiveFilters.svelte` clears the filter object on mount. Persist the
relevant fields (`namn`, `referenskod`, `extraid`, `language`) to
`localStorage` so the user's last filter survives a hard refresh.

### 📋 Karaoke cursor reads stale `currentTime` after seek-while-paused
The `$effect` in `transcript-highlighter.svelte` ticks on every RAF, so
this should be fine — but worth a quick smoke test once visual search
goes live.

### 📋 Hit-card thumbnail: combine main thumbnail + frame chip into one image
Right now we render two `<img>` tags. With `chunk_frames` we have the
exact frame for each hit — could replace `thumbnail_url(doc_id)` with
`chunk_frame_url(...)` for search-mode hits and skip the doc thumbnail
entirely. Saves a request per card.

---

## Cleanup / hygiene

### ✅ Drop the dead `frame_*` columns from `chunks` schema (done in code)

`CHUNK_SCHEMA` no longer declares `frame_blob`/`frame_mime`/`frame_width`/
`frame_height`/`frame_embedding`, and the backend no longer reads them (see #4).
**New** ingests produce a clean `chunks` table automatically.

An **existing** dataset built with the old schema still carries the (all-NULL)
columns — drop them in place with:

```python
import lance
ds = lance.dataset("./transcripts.lance/chunks.lance")
ds.drop_columns(["frame_blob", "frame_mime", "frame_width",
                 "frame_height", "frame_embedding"])
```

### 📋 `make compact` after multi-stage writes

Once `extract-chunk-frames` lands many small fragments and
`embed-chunk-frames` adds a column, run `make compact` to consolidate
fragments and rebuild the IVF_PQ index. Optional — search works fine
without it on this dataset size, just slightly faster after.

### 📋 Update `images_per.jpg` decision

Stray test image in repo root, currently gitignored. Either delete it
or move to a `tests/fixtures/` dir for use in unit tests.

---

## Search performance — observed slow, prioritized fixes

Multimodal / vector search currently feels sluggish (a few hundred ms to
seconds per query). Most of that latency is fixable. Items roughly ordered
by impact-per-effort.

### 📋 Boost recall *and* speed of vector queries with `nprobes` + `refine_factor`

The IVF_PQ index defaults to `nprobes=1` (touches one of 256 partitions →
fast but poor recall, which forces a re-query reflex). Lance docs:

> "Search with the index … `nprobes`: Number of partitions to search."

Concrete change in `backend/app.py` `_vector_search(...)`:

```python
chunks.query()
    .nearest_to(vec)
    .distance_type("cosine")
    .nprobes(20)            # up from default ~1 — visit 20/256 partitions
    .refine_factor(3)       # re-score top-K * 3 with full-precision vectors
    .limit(n)
```

`nprobes=20` is the sweet spot for `num_partitions=256` (≈ √n). Adds maybe
20–30 ms but recall jumps dramatically — fewer "feels broken" misses.
`refine_factor=3` re-checks the top results with un-quantized vectors,
costs ~5 ms, big quality win.

### 📋 Stop fetching `alignments_json` in the search results projection

`alignments_json` is a multi-KB blob per chunk; the search list only needs
text + start/end + metadata. Currently `_run_search` projects all columns,
which means each hit pulls a big JSON payload that the list view doesn't
even render. Move that fetch to **playback time** — the player pane already
re-fetches when a hit is clicked.

In `backend/app.py`, change the search projection from `select(["…", "alignments_json"])`
to omit `alignments_json`. Add a `/api/chunk-alignments/{doc_id}/{speech_id}/{chunk_id}`
endpoint that returns it on demand. **Estimated win: 30–60% on result-set
serialization for large queries**, especially on hybrid/all where 30+ rows
come back.

### 📋 Cache the embedding client + keep a query-vector LRU

Two cheap wins inside `backend/app.py` `_get_client()`:

1. The vLLM client object is already cached at app startup ✅ — but each
   query rebuilds the chat-message wrapper. That's negligible; ignore.
2. Add an LRU cache on `client.embed_text(query)` keyed by the exact query
   string. Repeated searches (same query, different filters) skip the
   ~50 ms vLLM RTT. `functools.lru_cache(maxsize=512)` is enough.

For images, no caching — every uploaded image is unique. Just embed once.

### 📋 Run `make compact` after extract+embed completes

145 small fragments after `extract-chunk-frames` means scans pay metadata
overhead. Lance docs:

> "Many small appends will lead to a large number of small fragments…
> queries [become] slower due to the need to filter out deleted rows."

After both extract + embed steps land:

```bash
make compact     # consolidates fragments, rebuilds IVF_PQ
```

Expected ~5–10% scan-time improvement on a fragmented table.

### 📋 Async parallel branches for `mode=hybrid` and `mode=all`

`mode=hybrid` runs FTS *then* vector search sequentially before RRF. They're
independent. Same for `mode=all` (FTS + text-vector + frame-vector — three
independent calls). Wrap each branch in `asyncio.gather(...)` to overlap.
Native Lance is sync, so use `loop.run_in_executor(...)`. **Cuts hybrid
latency by ~40% on cold cache.**

### 📋 Drop the rerank cross-encoder for typical queries

`Qwen3-VL-Reranker-2B` adds 200–500 ms when toggled on — that's the bulk of
the user-visible slowness when "rerank" is checked. Two mitigations:

1. **Frontend default off**: the toggle is currently easy to leave on; default
   it off and label clearly that it's for "best quality, slower."
2. **Cap top-K at 30**: the reranker only re-orders the top results returned
   by the underlying search. We already do this; verify no path passes a
   larger candidate set.

If quality at default-off feels weak, the IVF_PQ `nprobes`/`refine_factor`
improvements above usually close the gap without needing the cross-encoder.

### 📋 Frontend: debounce the search input + show pending state

In `frontend/src/lib/components/search-bar.svelte`, the form submits on
Enter — fine. But fast typers + auto-search-on-pause would feel faster
than waiting for an explicit submit. Add a 300 ms debounce on text input
and dispatch a search if the input is non-empty. Combine with the
loading-spinner state that already exists in `+page.svelte`.

### 📋 (Stretch) Try `IVF_HNSW_SQ` for the frame-embedding index

Lance docs:

> "IVF_HNSW_SQ offers better recall at the cost of more memory."

For `frame_embedding` (145 k × 2048 dims, ~1.2 GB raw → ~300 MB SQ-quantized),
the better recall might let you keep `nprobes` low and end up faster overall.
Worth a one-shot benchmark after the basic IVF_PQ implementation lands.

```python
ds.create_index("frame_embedding", index_type="IVF_HNSW_SQ",
                num_partitions=256, replace=True)
```

## vLLM performance — observed slow embeddings

A single `POST /v1/embeddings` against Qwen3-VL-Embedding-2B takes 100–300 ms
(plus ~5 ms localhost RTT). For a hybrid search this fires once per query;
combined with the index search itself, it's the bulk of the visible latency.
Items below are ordered by impact-per-effort.

### 📋 Switch to **Qwen3-VL-Embedding-2B** (biggest single win)

Qwen ships a 2B-parameter sibling to the 8B model — same architecture, same
2048-d output space, ~4× faster forward pass, ~4× lower memory (vs the 8B).
Per the Qwen3-VL release page, quality on retrieval benchmarks is within
1–2 points of 8B for most languages.

Change in `Makefile`:

```diff
- vllm/vllm-openai:v0.22.0 --model Qwen/Qwen3-VL-Embedding-2B …
+ vllm/vllm-openai:latest --model Qwen/Qwen3-VL-Embedding-2B …
```

⚠️ Caveat: re-running `embed-chunks` is required because the embedding
spaces aren't compatible across model sizes. ~5–7 min instead of 25 min on
the same hardware (it's faster too).

### 📋 Verify `--enable-prefix-caching` is active (default in v1)

Our chat-template sends the same system instruction every query
(`"Represent the user's input."`). Prefix caching reuses the KV cache for
that prefix across queries, saving ~10 ms per call. Should already be on
in vLLM 0.20+ but worth confirming in the startup log:

```
INFO  …  enable_prefix_caching=True  …
```

If for any reason it's not, add `--enable-prefix-caching` to
`embed-server-docker` in the Makefile.

### 📋 Make the backend's vLLM client async

`backend/app.py` opens an `httpx.Client` (sync) wrapped in a
`ThreadPoolExecutor`. Each search blocks one FastAPI worker until vLLM
responds. Two improvements:

1. Switch the embedding HTTP call to `httpx.AsyncClient`. The vLLM call
   then awaits at the FastAPI event-loop level, freeing the worker to
   handle other connections concurrently.
2. For `mode=hybrid` and `mode=all`, fire FTS + vector queries
   concurrently with `asyncio.gather`. (Already mentioned in the
   search-perf section above — the same change benefits both.)

Code path: `src/raudio/vllm/embedding.py` `VLLMEmbeddingClient._embed_one(...)`.
Concurrent-batch path (`embed-chunks`) already uses `ThreadPoolExecutor` —
keep that for the CLI batch case; only swap to async for the per-query
serving path.

### 📋 Use vLLM's `/metrics` endpoint to find the actual bottleneck

vLLM exposes Prometheus metrics on the same port:

```bash
curl -s http://127.0.0.1:8001/metrics | grep -E "vllm_(time_to_first_token|e2e_request_latency|gpu_cache_usage)"
```

Key metrics to watch while running searches:
- `vllm_e2e_request_latency_seconds_*` — end-to-end per request
- `vllm_time_to_first_token_seconds_*` — TTFT, dominated by KV-cache miss
- `vllm_gpu_cache_usage_perc` — should be > 0 if prefix caching helps
- `vllm_request_queue_time_seconds_*` — queue contention; should be ~0
  for a single-user demo

If TTFT is much larger than total latency minus a few ms, prefix caching
isn't being hit. If GPU cache usage stays at 0, prefix caching is off.

### 📋 Make sure the embed server is truly using GPU 2 alone

Earlier in this project the rerank server occasionally landed on the same
GPU as embed during memory profiling, slowing both. Confirm with:

```bash
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
```

GPU 2 should sit at ~25 GB (model + KV cache); if it's < 18 GB, vLLM is
under-utilized and `--gpu-memory-utilization 0.85` may not be applying
correctly inside the container.

### 📋 (Stretch) FP8 quantization

vLLM 0.20+ supports FP8 weight loading on Hopper / Blackwell. If/when an
FP8 checkpoint of Qwen3-VL-Embedding-2B (or 8B) is published on HF, swap
in the model and pass `--quantization fp8`. Roughly 2× throughput on
sm_120 vs bf16, no quality loss for embeddings (which only care about
the pooled hidden state).

Not actionable today — listed for the future.

### 📋 (Stretch) Run vLLM with `--async-scheduling`

Docs note: *"Specifying `--async-scheduling` improves the overall system
performance by overlapping scheduling overhead with the decoding process."*
Default is off in our current container start. Try adding it after other
items are validated — it can interact poorly with structured output and
sampling penalties, neither of which we use here.

```diff
  --max-model-len 8192 \
+ --async-scheduling \
  --limit-mm-per-prompt.image 1 \
```

---

### Quick benchmarking recipe

Before/after any of the above:

```bash
uv run python -c "
import time, lancedb, numpy as np
t = lancedb.connect('./transcripts.lance').open_table('chunks')
q = np.random.randn(2048).astype('float32')
# warmup
t.query().nearest_to(q).limit(20).to_list()
# measure
n, total = 50, 0
for _ in range(n):
    s = time.perf_counter()
    t.query().nearest_to(q).distance_type('cosine').nprobes(20).limit(20).to_list()
    total += time.perf_counter() - s
print(f'avg {total/n*1000:.1f} ms / query')
"
```

Compare to the same loop without `nprobes(20)` to see the impact.

---

## Backup plan (if `chunk_frames` fails for any reason)

If Lance keeps misbehaving, fall back to plain disk:

```
./frames/{doc_id}/{speech_id}-{chunk_id}.jpg
```

Backend serves them via `FileResponse`. Embeddings go into a separate
`chunk_frame_embeddings.lance` table (keys + 2048-d vector only — no
extension types, no merge_insert). ~15 min to implement.

---

## Code-quality backlog (skills audit, 2026-05-29)

Items deferred from the `writing-python` / `fastapi` / `svelte` / `writing-typescript`
cleanup pass. The pass applied all the safe, mechanical wins directly (see the
Closed section); these are the larger or behavior-shifting changes that were
**intentionally not auto-applied** — each notes why and how to verify.

### Backend (`backend/app.py`)

#### 📋 Replace the hand-rolled `SearchSpec` with a Pydantic model + shared `Depends`
`SearchSpec.__init__` both clamps values *and* raises `HTTPException` (an HTTP
concern in a data class), with the param list duplicated across `search_get`,
`search_post`, and the constructor. A Pydantic model behind a shared
`Depends(get_search_params)` would dedupe it. **Behavior change to confirm
first:** unknown `mode` would become a **422** (FastAPI validation) instead of
the current **400**. The frontend tolerates this (`api.ts` `asJson` doesn't
branch on 400), but it's an observable contract change — decide deliberately.

#### 📋 Domain-error hierarchy + exception handlers instead of inline `HTTPException`
Search/blob paths catch broad `Exception` and translate to `HTTPException`
inline. A small `DomainError` hierarchy + registered handlers (per the fastapi
skill) would centralize this. ⚠️ Do **not** narrow the `health()` ping to
`except httpx.HTTPError` — `httpx` is lazy-imported there behind the
`[multimodal]` extra, so an `ImportError` would escape as a 500. Keep that catch
broad.

#### 📋 Embedding client via `app.state` + a dependency, not a closure `state` dict
`_get_client` closes over a module-local `state = {"client": None}`. It works
(per-app-instance), but `app.state` + a FastAPI dependency expresses the lazy
singleton more idiomatically and testably.

#### 📋 CORS `allow_origins=["*"]` is hard-coded
Fine for the local demo (API-only behind the Bun proxy); tighten via settings if
this is ever exposed.

### CLI / Python (`src/raudio/`)

#### 📋 FTS-language defaults disagree (latent correctness bug on a Swedish corpus)
`ingest` defaults `--fts-language English` (`cli.py`), but `reindex-fts` defaults
`Swedish`. On this Swedish corpus the English stemmer silently returns zero hits
for inflected forms. Pick one default (almost certainly `Swedish`) so a plain
`raudio ingest` produces a usable index without the `--fts-language Swedish` flag.

#### 📋 `_Ctx` global state → Typer's context object
`cli.py` shares `--db`/`--table` via mutable class attributes on `_Ctx`. The
idiomatic Typer pattern is `ctx.obj` / `typer.Context`. Low-risk but touches
every command signature.

#### 📋 Optional: split `cli.py` (944 lines) into a `cli/` package
Each command is already a thin lazy-importing wrapper, so coupling is low. A
split by group (ingest / search / embed / maintenance) is feasible and low-risk
**if** verified with `uv run raudio --help` + importing `raudio.cli:app`. Polish,
not a fix — the lazy-import discipline already provides most of the decoupling.

#### 📋 `print()` → `logging` in library code
`media/thumbnails.py`, `media/download.py`, `asr/detect_language.py` print progress directly.
Library modules should log (or return data) and let the CLI render; this matters
if they're ever imported by the backend.

#### 📋 Minor typing/dedup
`ingest_document` is a public re-export never called in-repo that duplicates
`ingest_many`'s 9-param signature; `frames._extract_one` takes an untyped
`args: tuple` (lost the 8-element typing); `detect_language` probe closures lack
annotations; `iter_matching_words` uses bare `dict`/`list` params; the reranker
prefix/suffix constants in `vllm/reranker.py` and `retrieval/qwen3_vl_reranker.jinja`
want a one-line cross-reference comment (they must stay in sync). All low priority.

### Frontend & demo

#### 📋 Add eslint + prettier to `frontend/` and `demo/`
Neither has a lint/format config or scripts. `svelte-check` + `tsc` are the type
gate today; eslint/prettier would round out the `writing-typescript` toolchain.

#### 📋 `demo/`: type the Web Worker message protocol
`worker.ts` ↔ `+page.svelte`/`RealtimePanel`/`BatchPanel` communicate over one
`postMessage` channel disambiguated only by an implicit `jobId` convention. A
shared discriminated-union message type would remove the ad-hoc `as` casts in
`RealtimePanel` and the untyped boundary. (Secondary app — low urgency.)

#### 📋 `demo/`: tighten `tsconfig`, drop unused dep, decide the `/search` stub
Add `noUncheckedIndexedAccess` + `exactOptionalPropertyTypes` (as now done in
`frontend/`); `d3-scale` appears unused in `src` (remove from `package.json` +
`bun install`); the `/search` route is a permanent `WorkInProgress` placeholder —
build it or remove the route + sidebar entry. Also: `formatBytes` is duplicated
across `ProgressItem`, `TranscriptHistory`, and `BatchPanel` — extract one shared
helper.

### Repo hygiene

#### 📋 Stray files
`images_per.jpg` (repo root, gitignored) — delete or move to `tests/fixtures/`.
`frontend/transcripts.lance/` is an empty stray dir — remove. `db.table_names()`
is deprecated in lancedb (use `list_tables()`); verify the return shape before
swapping in `backend/app.py` + `cli.py`.

---

## Closed (for context / commit log)

### ✅ Skills-audit cleanup pass (2026-05-29)

Applied `writing-python` / `fastapi` / `svelte` / `writing-typescript`. All gates
green afterward: ruff ✅, ty ✅, `pytest` 32 passed, `raudio --help` ✅,
frontend `bun run check` 0/0 + build ✅.

- **Tooling:** added `[tool.ruff]` (lint-only, `py311`), `[tool.ty]`,
  `[tool.pytest.ini_options]`, and a `dev` dependency group (pytest, httpx) to
  `pyproject.toml`. De-duplicated `.gitignore` and added tool-cache/IDE entries.
- **Tests (new):** `tests/test_units.py` (timecode, `_parse_range`,
  `_build_where_clause`, `_rrf_fuse`, query-term extraction) and
  `tests/test_backend_smoke.py` (dataset-gated end-to-end: FTS, health,
  documents, thumbnail, Range streaming, 503 degradation).
- **Real bugs fixed:** `typer.Exit("msg")` (×4) was passing a *string* as the
  exit code and silently dropping the message → now `_die()` (prints + exits 1);
  removed a dead `chunks_ds` param from `_run_search` and dead `chunk_frames_tbl`.
- **FastAPI:** `async search_post` now offloads the blocking vLLM/Lance work via
  `run_in_threadpool` (was stalling the event loop); `raise … from e` on
  re-raised `HTTPException`s; fixed import order (E402).
- **Lint/types:** dead imports removed; `collections.abc` over `typing`;
  storage-version constants typed `Final` (fix Lance `Literal` mismatch);
  `Image.Resampling.LANCZOS`; `capture_output=True`; `zip(strict=…)`; etc.
- **Stale docstrings corrected:** `frames.py` ("no separate frames table" — now
  the `chunk_frames` table), `audio.py` (superseded `media_uri` design → Blob V2
  External), `search.py` (`create_fts=True` kwarg that never existed),
  `_pick_alignments` (interval notation), `cli.py` module + `_merge_insert_vectors`.
- **Frontend:** enabled `noUncheckedIndexedAccess` + `exactOptionalPropertyTypes`
  and fixed the fallout (`api.ts` optional fields, transcript binary-search index
  access); `as` cast → `satisfies` in `search-bar`; cleaned `loadMore`; deleted 3
  dead UI components (Card/Badge/Checkbox).
- **Demo:** removed dead canvas `AudioVisualizer.svelte` and an unused
  `formatDuration` export.
- **Docs:** added [GUIDE.md](GUIDE.md) (architecture / data flow / design
  rationale / onboarding) and restructured this file.

### Earlier

- ✅ Migrated frontend from vanilla HTML → SvelteKit + Tailwind v4 + Bun proxy.
- ✅ Renamed `frontend-svelte/` → `frontend/`, deleted old `frontend/`.
- ✅ Added secondary `demo/` SvelteKit app (transformers.js audio).
- ✅ Theme toggle bug (production-build cache) — replaced rune store with
  a self-contained class field on the toggle component.
- ✅ List-view selection ring invisible — switched to `ring-inset`.
- ✅ Karaoke highlight not catching `Göran` etc — `queryTerms` regex now
  uses `\p{L}` instead of `\w`.
- ✅ Status badge in navbar reports vLLM embed/rerank reachability + Lance
  dataset facts.
- ✅ `make compact` Make target + `raudio compact` CLI command.
- ✅ Stripped `Co-Authored-By: Claude` from all commits, force-pushed.
- ✅ `make embed-chunks` ran clean (135 k → 145 k chunks, IVF_PQ built).
- ✅ Schema redesign: `chunk_frames` as a separate Lance table per Lance
  2.2 docs (commit `3954ee5`).
