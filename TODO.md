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
