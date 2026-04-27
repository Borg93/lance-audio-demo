# TODO

Living checklist for `lance-audio-demo`. Update as items land.

> **How to read this:** ✅ done · ⏳ in progress · ❌ blocked · 📋 backlog.
> Each pending item points to the file(s) and command needed to pick it up.

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

### ❌ 2. `embed-chunk-frames` blocked by vLLM Qwen3-VL deepstack bug

Confirmed bug in vLLM 0.20.0 for Qwen3-VL-Embedding-8B: the warmup-time
deepstack-input-embeds buffer is sized differently from runtime, even
when `--mm-processor-kwargs '{"min_pixels": …, "max_pixels": …}'` is set.
Single image request kills the engine with:

```
ValueError: Requested more deepstack tokens than available in buffer:
            num_tokens=N > buffer=N-k
```

We have not been able to find a vLLM config that consistently avoids this.

**Two options to unblock — both are cheap to implement (~80 LOC each):**

- **(A)** In-process HF transformers fallback. Load
  Qwen3-VL-Embedding-8B once at backend startup via `transformers`,
  embed images directly. Slower (~2 s/query) but immune to vLLM internals.
  See `src/raudio/embeddings.py` — add `class HFClient(EmbeddingClient)`
  alongside `VLLMClient`. Backend already supports `--backend hf` switch
  via `make_client()` — just needs the implementation.
- **(B)** Different vLLM tag (`v0.21.0+` once released, or back to `v0.10.x`).
  Risk: Blackwell sm_120 support compat. Won't know until tested.

User has not chosen yet. Ask before implementing.

### ❌ 3. `chunk_frames` IVF_PQ index build (depends on #2)

Once frame embeddings exist, `dataset.add_columns(...)` writes the
`frame_embedding` column, then `_ensure_vector_index` builds the cosine
IVF_PQ index. Code path is in place at
`src/raudio/cli.py` (`cmd_embed_chunk_frames`) — runs automatically when
the embed step completes.

---

## Visual / cross-modal search wiring

### ⏳ 4. Backend visual-search query path needs to read `chunk_frames`

Currently in `backend/app.py`, `_run_search(...)` for `mode=visual` does:

```python
_vector_search(chunks, vec, "frame_embedding", spec.n, where)
```

This still queries the (all-NULL) `chunks.frame_embedding` column. Needs
to be re-pointed at `chunk_frames_ds` and JOIN back with `chunks` for
text/timestamps/metadata in the response.

Suggested approach (≈30 LOC):
1. If `chunk_frames_ds` is opened and has `frame_embedding`, run vector
   search against it → list of (doc_id, speech_id, chunk_id) keys.
2. Build a SQL `IN (…)` filter on chunks for those keys, fetch the
   chunk text + alignments + metadata.
3. Compose hits in the same shape the frontend already expects.

`/api/chunk-frame` is already wired correctly — only the search side
needs the redirect.

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

### 📋 Drop the dead `frame_*` columns from `chunks` schema

After `chunk_frames` is in production use, the columns `frame_blob`,
`frame_mime`, `frame_width`, `frame_height`, `frame_embedding` on the
`chunks` table become dead weight (still all NULL). They cost nothing
on disk but add schema noise.

To remove safely:

```python
import lance
ds = lance.dataset("./transcripts.lance/chunks.lance")
ds.drop_columns(["frame_blob", "frame_mime", "frame_width",
                 "frame_height", "frame_embedding"])
```

Don't do this until visual search has been exercised against
`chunk_frames` for a while — easier to keep the legacy fallback path in
the backend until then.

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

`Qwen3-VL-Reranker-8B` adds 200–500 ms when toggled on — that's the bulk of
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

For `frame_embedding` (145 k × 1024 dims, ~600 MB raw → ~150 MB SQ-quantized),
the better recall might let you keep `nprobes` low and end up faster overall.
Worth a one-shot benchmark after the basic IVF_PQ implementation lands.

```python
ds.create_index("frame_embedding", index_type="IVF_HNSW_SQ",
                num_partitions=256, replace=True)
```

### Quick benchmarking recipe

Before/after any of the above:

```bash
uv run python -c "
import time, lancedb, numpy as np
t = lancedb.connect('./transcripts.lance').open_table('chunks')
q = np.random.randn(1024).astype('float32')
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
`chunk_frame_embeddings.lance` table (keys + 1024-d vector only — no
extension types, no merge_insert). ~15 min to implement.

---

## Closed (for context / commit log)

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
