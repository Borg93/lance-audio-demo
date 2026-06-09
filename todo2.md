# todo2 — FiftyOne-inspired curation & exploration roadmap

This doc plans the next wave of **raudio** features, inspired by what FiftyOne gives a
dataset-exploration tool, but scoped tightly to *this* corpus (145,175 × 30 s Swedish
press-conference chunks across 1,154 videos) and *this* stack: Lance-backed read path,
FastAPI backend, SvelteKit 5 frontend. Swedish data, English chrome.

The unifying lever is the one we already use for the atlas: **one embedding column per
modality, selected by a `space`** parameter.

| modality            | embedding column                                   | status      |
| ------------------- | -------------------------------------------------- | ----------- |
| `text`              | `chunks.text_embedding` (IVF_PQ index)             | live        |
| `visual` (image)    | `chunk_frames.frame_embedding` (joined onto chunks)| live        |
| `caption`           | `chunk_frames.caption_embedding` (joined onto chunks)| live      |
| `voice` (audio)     | `chunks.speaker_embedding` (ECAPA)                 | **planned** |
| `document`          | `documents.doc_embedding`                          | **planned** |

Every feature below is **modality-aware via the exact `--space` plumbing the atlas already
has** (`FeatureRunOptions.space` Literal + per-space output-column namespacing + the
`_run_atlas`-style router fn). We ship `text|visual|caption` now; `voice`/`document` light
up the day their columns land — usually a one-line addition to a `space → column` map, not
a new code path. No other modalities (no 3D / geo / point-cloud).

House conventions assumed throughout: Python type hints + Pydantic + ruff/ty clean +
Annotated FastAPI params + sync handlers threadpooling the blocking Lance work + the
native-LanceDB scan idiom; TS strict types + zod at the boundary + Svelte 5 runes +
auto-apply-on-change + guided UI over raw SQL. New *derived* columns go through the
`FEATURES` registry; new endpoints go in a focused router.

---

## Roadmap / suggested sequencing

Ordered by **value-per-effort**. The first two are the high-value / low-effort pair and
should ship together.

| # | feature | effort | impact | depends on | status |
| - | ------- | :----: | ------ | ---------- | :----: |
| 1 | **Group-by-video** (result-layer grouping by `doc_id`) | **S** | High — collapses the "one video floods the page" noise; pure frontend reshape | uniform `Hit` shape (`doc_id`+`namn` already projected) | 📋 |
| 2 | **Uniqueness + near-dup collapse** (`raudio feature uniqueness`) | **M** | High — directly fixes adjacent-chunk redundancy at the *retrieval* level | `projection.py` scan/attach helpers; `--space` plumbing; existing embedding cols | 📋 |
| 3 | **More-like-this** (similarity sort from a hit) | **M** | High — "explore from here"; zero new data, zero GPU, reuses kNN helpers | `_vector_search`/`_frame_search`; stored embedding cols | 📋 |
| 4 | **Stats / histograms** (aggregation panel + live facet counts) | **M** | Medium-high — turns the opaque filter builder into faceted, count-annotated discovery | `chunks_ds` scan; `/api/columns` classifier; LayerChart | 📋 |
| 5 | **Tags + saved views** (curation backbone) | **M** | Medium — first real curation loop; but introduces *mutable* state | new SQLite store; `_attach_captions` join pattern; atlas `_rowid` take | 📋 |

> **Highest-leverage pair:** **#1 group-by-video** and **#2 uniqueness/dedup** together fix
> the redundancy problem we've already hit — adjacent 30 s clips of one press conference are
> near-identical and currently flood the top of semantic/scene results with ~5 copies of one
> moment. Group-by-video is a *presentation* fix (cheap, zero risk); uniqueness/dedup is the
> *retrieval* fix (a one-time offline build, near-zero search-time cost). Ship #1 first
> (pure frontend), then #2.

---

## 1. Group-by-video (dynamic groups)

### Goal
Group the flat result list by video (`doc_id`), rendering a collapsible per-video header
with its hits underneath, toggled by a "Group by video" control in the results view bar.
Optionally add a backend **per-video cap** so one dominant video can't swallow the page.

### Backend
**Core path needs no backend.** Every hit already carries the grouping key + label:
`_HIT_COLUMNS`/`_PAYLOAD_COLUMNS` in `backend/search/service.py` project `doc_id` and `namn`
on all seven modes, and `run_search` returns a flat `list[dict]` (uniform Hit shape).
Grouping is derivable client-side.

**Optional per-video cap** (the dedup/diversify pairing — small, well-scoped):
- `backend/search/spec.py`: add `per_doc_cap: int | None = None` with a `@field_validator`
  clamping to `[1, 50]` (mirrors `_clamp_n`). Do **not** add a `group` enum to the backend —
  grouping is presentation; the backend only needs the cap.
- `backend/search/service.py`: a pure helper `_cap_per_doc(hits, cap)` that walks the
  already-ranked hits, keeps `dict[doc_id, count]`, drops hits past the cap (rank order
  preserved). Apply it on the ranked list **just before** `_postprocess_hits` in each
  branch — *not inside* `_postprocess_hits` (which also runs the caption scan and the
  browse/topic-only branch). Cleanest: a single `_finalize(hits, spec, chunk_frames)` that
  does cap-then-postprocess. Over-fetch (`spec.n`, `spec.n * 3` in the fusion branches) keeps
  a capped page useful.
- `backend/search/router.py`: thread `per_doc_cap` through both `search_get`
  (`Annotated[int|None, Query()]`) and `search_post` (`Form()`), like `extraid`/`weight`.

`loadMore` re-runs at higher `n` and re-caps each fetch; the `allLoaded` heuristic
(`hits.length < requested`) still works since the cap only shrinks the list.

### Frontend
- `frontend/src/routes/+page.svelte` owns `hits` and already derives
  `docCount = new Set(hits.map(h => h.doc_id)).size`. Add `groupByVideo = $state(false)`
  (persist to localStorage like `gridCols`/`tableCols`) and a `$derived` `groups` that
  buckets `hits` into a `Map` preserving first-seen order (best-ranked video floats up).
  Label = `hits[0].namn ?? audio_path ?? doc_id`.
- New thin presentational `hit-group-list.svelte`: maps groups → a collapsible
  `<details>/<summary>` header (namn + count badge + thumbnail) and reuses `<HitCard>` per
  member. Wire into the list/grid branch:
  `{#if groupByVideo}<HitGroupList .../>{:else}<HitList .../>{/if}`. Collapse state =
  a persistent `SvelteSet<string>` of collapsed `doc_id`s (so collapses survive load-more).
- Toggle = one more icon button in the existing view-bar `<div class="ml-auto …">` next to
  list/grid/table/map (Layers icon, `variant={… ? 'secondary' : 'ghost'}`).
- `frontend/src/lib/api.ts`: **only if the cap ships** — add `perDocCap?: number | undefined`
  (explicit `| undefined`, `exactOptionalPropertyTypes` is on) and append in both GET/POST
  branches. The grouping toggle never touches `api.ts`.
- Optional cap UI: a "Max per video" Select in `search-settings.svelte`.

Keep grouping to **list + grid**; leave the table ungrouped for v1 (the `doc_id` column
already exists for manual sorting). Don't tangle the Map view's cross-filter store.

### Data-model impact
**None.** Reuses `doc_id` (ingest identity, on every chunk) and `namn` (already projected).
Cheapest possible feature on the data axis — zero bytes at rest. `per_doc_cap` is a request
param, not stored state.

### Consequences & impacts
- **Perf:** trivial — bucketing ≤ 200 hits into a Map is microseconds; no extra network.
- **UX:** big readability win on the press-conference corpus; complements the existing
  "X chunks across Y documents" line. The cap changes *what* is returned (drops lower-ranked
  chunks of a dominant video), so it must be an explicit, discoverable toggle, never silent.
- **Scale honesty:** presentation polish, not a scaling lever — no server-side group
  pagination, no group-count endpoint. Grouping is over the *current page* only.

### Risks
- The grouped count per video reflects only the loaded page (`n`), **not** the video's true
  chunk total — label honestly ("7 in results"), don't imply completeness.
- A backend cap applied inside `_postprocess_hits` would wrongly hit the caption scan and the
  browse/topic-only branch — apply per-branch before postprocess (or via `_finalize`).
- Collapse state must survive a hits refresh — persistent `SvelteSet`, not a fresh `$derived`.
- Don't extend grouping into the Map view (it already overloads the bottom table with
  selection-vs-search hits).

### Effort
**S** (core frontend-only). Cap is a small separate follow-up.

### Modality notes
`doc_id`/`namn` are modality-independent, so the same grouping code works across all seven
search modes unchanged. For the planned **document** modality, `doc_id` *is* the group, so
group-by-video degenerates to one-hit-per-group there — harmless, no special branch.

### Open questions
- Clickable group header → open the whole video in the player (synth a `start=0` hit like
  `openDoc()`), or pure visual divider?
- Default group order: best-member-rank (first-seen, recommended) vs group size?
- Ship the `per_doc_cap` now, or defer to the uniqueness feature it pairs with? Tight-scope
  ethos argues frontend-only first.
- Group header label disambiguation when `namn` is long/duplicated (append
  `referenskod`/`audio_path` stem, like the atlas)?

---

## 2. Uniqueness + near-duplicate collapse (`raudio feature uniqueness`)

### Goal
Add `raudio feature uniqueness --space text|visual|caption` writing two per-chunk scalar
columns to `chunks.lance`:
- `uniqueness` (float32) = `1 − mean cosine` to the *k* nearest neighbours in the chosen
  embedding space,
- `dup_group` (int32) = connected-component id where edges are cosine ≥ ε.

At search time: a **"Collapse near-duplicates"** switch (collapses each `dup_group` to its
top-ranked hit) and a **"Sort by"** Select (`relevance` | `unique`).

### Backend
**The one non-obvious decision (grounded, measured):** do **not** compute this with a
per-row IVF query loop. Only `text_embedding` has an IVF_PQ index (~23 ms/query → ~56 min for
145k); `frame_embedding`/`caption_embedding` have **no** vector index (~258 ms brute-force →
~10 h). The right engine is the **in-memory blocked all-pairs pass** that `projection.py`
already establishes: load the whole 145,175 × 2048 float32 matrix (~1.19 GB, ~1.6 s) and do a
blocked normalized matmul + `argpartition` top-k (~3.4 min CPU NumPy; seconds on GPU).
`dup_group` is a union-find over the ε-thresholded kNN edges from that *same* pass.

New column-builder `src/raudio/features/uniqueness.py` (mirrors `projection.py`'s two-pass
global-fit shape — every row's value depends on all rows, so it **cannot** be an
`add_columns` UDF):
```
compute_uniqueness_columns(chunks_path, *, source_column, uniq_col, group_col,
                           k=10, eps=0.92, overwrite, batch_rows, progress) -> int
```
- Reuse `projection._load_embedding_matrix(ds, source_column, batch_rows)` — already returns
  `(row_ids, (N,2048) float32)` and raises on NULLs (exactly the precondition we want).
- L2-normalize once; blocked loop computing `block @ Mn.T` (B×N cosine);
  `np.argpartition(-block, k+1)` for the k+1 nearest (drop self);
  `uniqueness = 1 − mean(top-k cosines excluding self)`; collect edges where `sim ≥ eps` into
  a union-find → `dup_group`.
- Attach both via `projection._attach_column_by_row_id` (float32 / int32) — **metadata-only**
  `add_columns`, no fragment rewrite, safe next to `text_embedding` on the wide schema.
- Optional torch GPU path behind `try/except` (NumPy fallback is the always-works path).

`FEATURES` registry + router (`src/raudio/features/columns.py`):
- Add `UNIQUENESS_COLUMN`/`DUP_GROUP_COLUMN`. Per-space namespacing like atlas:
  `text → (uniqueness, dup_group)`, `visual → (uniqueness_img, dup_group_img)`,
  `caption → (uniqueness_cap, dup_group_cap)`, mapping to
  `text_embedding`/`frame_embedding`/`caption_embedding`.
- `_run_uniqueness(db_path, opts, progress)` routes on `opts.space` (same switch as
  `_run_atlas`), picks source + output columns, calls the builder. No index (scalar columns).
- Register `FEATURES["uniqueness"]` (`table="chunks"`). The current
  `FeatureRunOptions.space` Literal is `["text","visual","caption"]`; **voice is out of scope**
  (`speaker_embedding` does not exist yet — verified). It lights up by adding `"voice"` to the
  Literal + the source-column map the day ECAPA lands.

CLI (`src/raudio/cli/features.py`): `--space` and the `FEATURES` loop already exist — just
extend help text. **Recommended:** surface `--k`/`--eps` as typer options threaded into
`FeatureRunOptions` (honours the no-hardcoding ethos so ε can be tuned).

Search (`backend/search/spec.py`): add `dedup: bool = False` and
`sort: Literal["relevance","unique"] = "relevance"`. Wire GET `Query` + POST `Form` params in
`backend/search/router.py` (both handlers).

Search core (`backend/search/service.py`):
- **`_optional_columns(chunks)`** appends `uniqueness`/`dup_group` to the projection lists
  **only when present** (guard on `chunks.schema.names`). Selecting a missing Lance column
  **errors** — this guard is mandatory (the #1 footgun).
- `_collapse_dup_groups(hits)`: walk ranked hits, keep first occurrence per `dup_group`
  (best-scoring representative), drop the rest; **NULL `dup_group` is always kept**. Apply in
  `run_search` right before the final `_postprocess_hits`, after rerank/fusion, when
  `spec.dedup`.
- `_sort_by_uniqueness(hits)`: stable sort by `-uniqueness` (NULLs last) when
  `spec.sort=="unique"`, applied after dedup. Both are pure O(n) list ops over already-fetched
  dicts — no extra Lance round-trip. Because `_frames_to_chunk_hits` joins back via the
  projection list, the columns flow through for visual/scene too once added.

### Frontend
- `api.ts`: `HitSchema` gains `uniqueness: z.number().nullable().optional()` and
  `dup_group: z.number().int().nullable().optional()` (same optional pattern as `caption`).
  `SearchSpec` gains `dedup?: boolean | undefined` and
  `sort?: 'relevance' | 'unique' | undefined`; wire both into GET/POST builders, only when set.
- `search-settings.svelte`: two `$bindable` props `dedup`/`sortBy`. A "Collapse
  near-duplicates" `Switch` — copy: *"Show one result per near-identical moment (adjacent 30 s
  clips of the same video collapse to the best-scoring one)."* Disable/grey with a hint when
  the columns aren't built (reuse a feature-flag check like `framesUnavailable`). A "Sort by"
  `Select`: Relevance / Most unique.
- `search-bar.svelte`: thread `dedup`/`sortBy` as `$state`, bind into `<SearchSettings>`, add
  to `buildSpec()` (mode-agnostic).
- `hit-card.svelte`: a uniqueness badge in both `tile` and `row` layouts —
  `{#if hit.uniqueness != null}` render a small pill (snowflake/star icon +
  `{Math.round(hit.uniqueness*100)}`).

### Data-model impact
Two new per-chunk scalar columns **per space** on `chunks.lance` (~145k × 8 bytes ≈ **1.2 MB**
each — negligible), attached via metadata-only `add_columns` (no fragment rewrite, no
`merge_insert`). **Global fit** (depends on all rows) → **not** incrementally fill-NULL-able
like `text_embedding`; `--all`/overwrite rebuilds, same property as the atlas columns. A later
ingest leaves new rows NULL until a full rebuild — the search code treats the columns as
optional/nullable, so it degrades gracefully (un-scored rows never collapse, sort last).
`dup_group` ids are **not stable across rebuilds** (independent fit, like `atlas_cluster`) —
fine, used only within one result list. Build cost: ~1.6 s load + ~3.4 min CPU pass per space
(seconds on GPU), ~1.19 GB transient RAM. No vector index (scalar columns).

### Consequences & impacts
- **Value (honest):** directly fixes the adjacent-near-dup flood. **Dedup-collapse is the
  high-leverage half;** uniqueness-sort is a nice "show me the unusual clips" lens. Genuinely
  useful, not over-engineering, *because* the expensive part is a one-time offline build and
  the search-time part is a cheap O(n) list op.
- **Perf:** search-time cost ≈ zero (n ≤ 200); +2 small numeric fields per hit.
- **UX honesty:** collapsing reduces visible count below `n` — either over-fetch candidates
  before collapsing, or show *"showing N of M after collapsing duplicates."* `dup_group` is
  **semantic** (embedding-space) dedup, not time-adjacency, so it also collapses repeated
  boilerplate across different videos (identical intro stings) — usually desirable; state it
  in the UI copy.

### Risks
- Projecting a missing Lance column **errors** — the optional-column guard is mandatory or
  every search 400s on an un-built DB (**highest-risk footgun**).
- The per-row IVF kNN loop is a trap (~56 min indexed / ~10 h un-indexed). The blocked-matmul
  global fit (~3.4 min CPU) is the only viable engine — a naive loop concludes "too slow."
- `eps` is corpus-specific and unvalidated — ship `--eps` and do one tuning pass over real
  adjacent-chunk cosine similarities, or collapse will be wrong.
- Collapse shrinks the page below `n` — over-fetch or honest "N of M" messaging.
- Global-fit columns aren't incrementally updatable — new chunks stay NULL until rebuild
  (fine for a static corpus, but **not** a live/streaming feature).
- `voice` has no backing column yet — scope to `text|visual|caption` or it 400s.
- `dup_group` ids are non-deterministic across rebuilds — safe only because used within one
  result list, never persisted as references.

### Effort
**M.**

### Modality notes
`space=text|visual|caption` maps **exactly** onto the atlas `--space` machinery — the cleanest
possible fit. Output columns are namespaced (`uniqueness`/`_img`/`_cap`,
`dup_group`/`_img`/`_cap`) identically to `atlas_`/`atlas_img_`/`atlas_cap_`, so the three
coexist and never clash. Voice activates by adding `"voice"` to the Literal + source map the
day `speaker_embedding` lands — zero other changes. Search-time dedup/sort are
modality-agnostic (read whichever columns are present); a per-space "collapse by visual vs
text similarity" toggle is a small future extension (pick which `dup_group_*` the collapse
reads), not v1.

### Open questions
- Which space's columns does search dedup/sort read **by default**? v1 simplest: text-space
  (`uniqueness`/`dup_group`). Let the UI pick the collapse space, or text-only for v1?
- Surface a "N copies collapsed" count per representative hit? (Cheap to emit in
  `_collapse_dup_groups` + a HitSchema field.) v1 or defer?
- Over-fetch (request `n*3`, collapse to `n`) for stable counts, vs honest "N of M" note?
- `eps` default — pick empirically from this corpus's adjacent-chunk similarities.
- Is torch importable in the *core* raudio CLI env (it lives in the isolated vLLM servers)?
  If not, NumPy-only for v1.
- How does the frontend **know** the columns are built (to enable/disable the toggle)? Extend
  `/api/health` or `/api/columns` to report presence, or reuse the runtime feature-flag
  pattern (`framesUnavailable`).

---

## 3. More-like-this / sort-by-similarity from a hit

### Goal
A read-only `GET /api/similar/{doc_id}/{speech_id}/{chunk_id}?space=text|visual|caption&n=…`
that (1) fetches the source chunk's already-built embedding by key, (2) feeds it straight into
the existing vector-search helpers, (3) returns the standard `Hit[]` shape. Frontend adds a
"Find similar ▾" control that replaces the results list. **Zero GPU, zero re-embedding, zero
new columns.**

**Honest scope correction:** the three implemented spaces are `text|visual|caption` (matching
the atlas selector), **not** `text|visual|voice`. No `speaker_embedding` exists; "voice" is a
disabled "coming soon" item.

### Backend
New focused router `backend/similar/router.py` (precedent: `atlas/router.py`,
`search/router.py`), registered in `backend/app.py` via `include_router`. Endpoint with
Annotated params `space: Literal["text","visual","caption"] = Query("text")` and
`n: int = Query(20)` (clamp to 200). Sync handler → `run_in_threadpool`. **No embedding
client** — never calls vLLM.

Thin service fn `find_similar` in `backend/similar/service.py`, importing existing helpers:
1. Map `space → (table, column)`: `text → (chunks, "text_embedding")` via `_vector_search`;
   `visual → frame_embedding`, `caption → caption_embedding` via `_frame_search`. The
   **source** vector for all three is read from `chunks` (the `frame_idx=0` `frame_embedding`
   and `caption_embedding` were already joined onto `chunks` by
   `columns.chunk_frame_embedding_column` — verified live), so one code path fetches it.
2. Fetch source by key with a single filtered scan (the `_attach_captions`/`atlas_chunk`
   pattern): `chunks_ds.to_table(columns=[col], filter=f"doc_id='{safe}' AND speech_id={sid}
   AND chunk_id={cid}").to_pylist()`. 404 if no row; **400** if the column is NULL/absent
   (*"similarity space not built — run `raudio feature {hint}`"*).
3. `np.float32` array → `_vector_search(chunks, vec, "text_embedding", n+1, where=None)` **or**
   `_frame_search(chunk_frames, chunks, vec, n+1, where=None, column=…)`. **Request n+1** —
   the source chunk is its own nearest neighbour (distance ≈ 0).
4. **Self-exclude** the source key via `_chunk_key`, truncate to `n`.
5. Return `_postprocess_hits(hits, chunk_frames)` — alignments + captions attach identically
   to `/api/search`, so the frontend renders with the same components (no schema drift).

All helpers already exist in `backend/search/service.py`; no new projection. SQL-quote
`doc_id` with the `.replace("'", "''")` idiom the atlas/search handlers use.

### Frontend
- `api.ts`: `export type SimilarSpace = AtlasSpace` (it's the same `'text'|'visual'|'caption'`
  union already exported). `getSimilar(doc_id, speech_id, chunk_id, space, n?, fetcher?):
  Promise<Hit[]>` GETs the endpoint and parses with the existing `HitsArraySchema`.
- **a11y trap:** `hit-card.svelte` is a single `<button>` — a nested interactive dropdown is
  invalid HTML. **Put the primary "Find similar ▾" control in `player-pane.svelte`** (the
  focused-hit detail pane, no wrapping button, room under `metaRows`). Items: Text / Visual /
  Caption; **Voice disabled "soon."** Wire via a new optional prop
  `onSimilar?: (space: SimilarSpace) => void` so the page owns data flow. Optionally a
  compact, absolute-positioned, `stopPropagation`-ing icon on hit-card tiles later.
- `+page.svelte`: `async function findSimilar(h, space)` modeled on
  `seedSearchFromSelection`: set `loadingHits`, call `getSimilar(...)`, assign `hits`, keep
  `active` (or clear), `allLoaded = true`, `view = 'list'`, clear the map cross-filter. Pass
  `onSimilar` into `<PlayerPane>`. Optional "Similar to: <snippet>" banner so the user knows
  it's a similarity sort, not a query.

### Data-model impact
**None** — the strong case for this feature. Reads only existing populated columns
(`chunks.text_embedding`, `chunks.frame_embedding`, `chunks.caption_embedding`). No
`add_columns`, no FEATURES entry, no rebuild, no index build. Uses the same IVF indexes
`/api/search` already uses.

### Consequences & impacts
- **Perf:** one sub-ms scalar-filtered scan + one kNN identical in cost to an existing
  semantic/visual search. **Cheaper** than a normal semantic search — it skips the
  embed-text/embed-image HTTP round-trip to the GPU server.
- **Payload:** same `Hit[]` shape as `/api/search`; no new wire format.
- **UX:** a natural "explore from here" affordance — click through the corpus by similarity.

### Risks
- **Voice space does not exist** — show it disabled "coming soon" or omit. Proposing it as
  live would be inventing capability.
- **Self-exclusion is mandatory** — the source is its own #1 neighbour (distance 0). Fetch
  `n+1`, drop the source key.
- NULL source vector (a chunk whose representative frame wasn't embedded) → clean **400** with
  a build hint, never a 500 (caption/visual coverage is not 100%).
- Dropdown inside hit-card's `<button>` is invalid HTML — keep the trigger in player-pane.
- Scope creep: keep it a pure kNN by stored vector in v1 — no `where`/rerank (the helpers
  accept `where` later if needed).

### Effort
**M** (mostly the frontend dropdown + one page handler).

### Modality notes
`space` maps 1:1 to the atlas selector and one stored column: `text → text_embedding`
(`_vector_search` on chunks); `visual → frame_embedding`, `caption → caption_embedding`
(`_frame_search`, source read from `chunks.*`). Reuse `AtlasSpace` verbatim. Voice slots in as
`space='voice' → chunks.speaker_embedding` and documents as `space='document' →
documents.doc_embedding` — each a one-line addition to the `space → (table, column)` map.

### Open questions
- Keep the source chunk `active` in the player after "Find similar" (better UX), or clear it?
- Visible "Similar to: …" chip (more honest, clearable back to the last query), or silent
  replace?
- Expose "Find similar" from the atlas selection table and hit-table rows too, or only
  card/player in v1? (Endpoint is view-agnostic.)
- Confirm a `dropdown-menu` primitive exists under `frontend/src/lib/components/ui` (the
  deleted `demo/` tree had one); if not, three inline buttons.
- `n` default: 20 (SearchSpec) vs ~30–50 for browsing; clamp to 200.

---

## 4. Stats / histograms — aggregation panel + live facet counts

### Goal
One read-only `GET /api/aggregate?field=…&bins=N` doing a native single-column Lance scan +
PyArrow `group_by` (categorical) or numpy histogram (numeric) over `chunks_ds`, optionally
pushing the current search `WHERE` into the scan. Frontend renders distributions with the
already-installed LayerChart and shows **live per-value counts** next to each pick in
`filter-popover.svelte` (FiftyOne sidebar-stats behaviour).

### Backend
New router `backend/aggregate/router.py` (mirrors `system/router.py` + `atlas/router.py`),
registered in `backend/app.py` after `topics_router`. Endpoint with Annotated params
(`StateDep`, `field: str`, `bins: int = Query(20, ge=2, le=100)`, `where: str|None = None`).
Sync handler → `run_in_threadpool`.

- **Validate `field`** against the same classifier `system/router.py:columns()` already uses
  (number vs text/categorical) — factor it into a shared `backend/_columns.py` so `/api/columns`
  and `/api/aggregate` agree (DRY, single source of truth). Reject unknown/vector/blob fields
  with **400** (fail-fast). Accept `topic_l*`/`doc_topic`/`atlas_cluster` when present
  (presence-gate on `field in state.chunks.schema.names`).
- **Categorical:** `tbl = chunks_ds.to_table(columns=[field], filter=where);
  g = tbl.group_by(field).aggregate([([], "count_all")]).sort_by([("count_all","descending")])`
  → `{field, kind:"categorical", total, buckets:[{value, count}], distinct}`. (Measured:
  `topic_l2` → 20 groups in 5 ms; `namn` → 1139 groups in 5 ms.)
- **Numeric:** `arr = chunks_ds.to_table(columns=[field], filter=where).column(field);
  vals = arr.to_numpy(zero_copy_only=False)`; drop NaN; `np.histogram(vals, bins=bins)` →
  `{field, kind:"numeric", total, min, max, bins:[{lo, hi, count}]}`. (Measured: `duration`
  over 145k → 5 ms, range 0.017–29.99 s.)
- **`where`** reuses the exact SQL the filter UI already builds (passed verbatim into
  `ds.to_table(filter=…)`, like `SearchSpec.where` — ~12 ms filtered).

No new columns, no FEATURES entry, no index, no cache (5–20 ms/call — the `_POINTS_CACHE`
memoization from atlas is available but not worth it; `Cache-Control` on the response is fine).
Add a small offline router test (categorical + numeric + bad-field-400 + with-where).

### Frontend
- `api.ts`: `AggregateCategoricalSchema`/`AggregateNumericSchema` (discriminated on `kind`)
  and `aggregate(field, opts?)` via the existing `asJson(r, schema)` helper — same pattern as
  `listColumns()`.
- New `frontend/src/lib/components/stats-panel.svelte` (runes): props `{ field, where? }`.
  Fetch via `aggregate()` in a `$effect` keyed on `field+where` (auto-apply-on-change). Render
  with **LayerChart** (`^2.0.0-next.64`, already used by `topic-treemap.svelte`) —
  hand-drawn `<rect>`s inside `<Chart><Svg>` (the **exact** idiom `topic-treemap.svelte`
  proves; **don't** rely on a high-level `<Bars>`/`<Histogram>` whose API may differ across
  `next.*` builds). Each bar clickable → `onselect(value | [lo,hi])` → the parent builds a
  WHERE clause via the **same** `filter-popover.svelte:buildClause()`
  (`field = 'value'` categorical; `field >= lo AND field < hi` numeric bin). **Top-N + "other"
  rollup** for high-cardinality categoricals. Use the `layerchart-svelte5` skill for the
  snippet/tooltip patterns. Avoid pulling `d3-scale` (compute rect x/width arithmetically).
- Field picker: populate from `listColumns()`, filtered to useful facets (categorical with
  `2 ≤ distinct ≤ ~250`, plus numeric) — `distinct` comes free in the categorical response.
- **Live counts in `filter-popover.svelte`:** when a categorical column is picked, call
  `aggregate(colName, { where: spec.where })` once and show the count next to each value (so
  counts reflect the active filter). Guided/auto-apply, no raw SQL.
- Placement: a collapsible **Stats** section/toggle next to List/Table/Map in `+page.svelte`
  (or embedded in the search-settings popover) — reflect the **whole chunks table** (or active
  filter), **not** Atlas-only.

### Data-model impact
**None** — the key cost/benefit point. Do **not** add a derived column or precomputed
artifact. Every aggregate is a live single-column scan finishing in 5–20 ms over all 145,175
rows. No `add_columns`, no index, no FEATURES entry, no staleness. Deliberate contrast with
atlas/topics (which *are* precomputed) precisely because scalar aggregation is cheap and
would only go stale if precomputed.

### Consequences & impacts
- **Perf:** excellent and honestly cheap — 5–20 ms/field, single column-file read, no GPU;
  filtered ~12 ms.
- **Payload:** tiny — categorical ≤ ~250 `{value,count}` (top-N capped), numeric `bins` (def.
  20) `{lo,hi,count}`; a few KB.
- **UX:** turns the opaque filter builder into a faceted, count-annotated picker + a real
  duration/topic distribution view — high discovery value.
- **Scale honesty:** right **only** because single-user + 145k fits a sub-20 ms full scan; at
  multi-tenant / 10–100M rows you'd precompute or use Lance pushdown stats — explicitly out of
  scope (would be over-engineering).

### Risks
- High-cardinality categoricals (`namn` ~1139, `referenskod`/`extraid` ~1153, `topic_l0` ~229)
  overflow a bar chart — **require** top-N + "other" rollup and a curated facet allowlist that
  excludes one-per-video id columns.
- `language` has exactly **1** distinct value (`'sv'`) — pointless; hide it. The plan's example
  field list is partly aspirational and must be pruned to fields that actually vary
  (`topic_l*`, `doc_topic`, `atlas_cluster`, `duration`).
- Passing `where` verbatim into `to_table(filter=…)` inherits the same
  SQL-injection-by-design surface the existing `where` already accepts — acceptable (same
  single-user trust boundary), but not new hardening.
- LayerChart is pre-release — follow the hand-drawn-`<rect>` idiom, not high-level components.
- `doc_topic`/`topic_l*` presence depends on `raudio feature topics` — gate on schema presence
  and fall back to always-present `duration`/`atlas_cluster`.

### Effort
**M.**

### Modality notes
Aggregation operates on **scalar metadata** columns (modality-independent), so the core needs
**no** `space` selector. One legitimate tie-in: the **cluster** facet should be space-aware —
`atlas_cluster` (text), `atlas_img_cluster` (visual), `atlas_cap_cluster` (caption) are three
columns, all present. When the panel is co-located with the Atlas view, let `field=cluster`
resolve to the active space's cluster column via the same `_SPACES` map in `atlas/router.py`,
so "colour the map by cluster" and "histogram the cluster sizes" agree. Documents/audio
contribute no facet fields today; the endpoint extends to `documents.lance` later via the same
code path on `state.docs_ds`.

### Open questions
- Stats panel placement: a new top-level tab next to Map/List/Table, inside the
  search-settings popover, or inline counts only in `filter-popover`? (Prefs lean compact bar
  + Settings panel.)
- Respect the **current search filter** (`spec.where`, sidebar-stats semantics) or always show
  full-corpus? Likely **both** — full-corpus in the standalone panel, filter-scoped in the
  picker.
- Confirm the default facet allowlist `{topic_l2, doc_topic, atlas_cluster, duration}` and
  whether to expose `topic_l0`/`topic_l1` behind a top-N view.
- Should the `cluster` facet follow the Atlas active space when co-located, or default to
  text-space `atlas_cluster`?
- Numeric bar click → half-open range (`>= lo AND < hi`) appended to `whereSql`, or a
  brush/range-select for multi-bin selection?

---

## 5. Tags + saved views (curation backbone)

### Goal
Add the curation layer raudio lacks: per-chunk **tags** (multi-value labels keyed by the
existing `(doc_id, speech_id, chunk_id)` identity) and **saved views** (a named `SearchSpec`
snapshot). Tags become a first-class filter (`tag:X` pill) and a write-back target from search
hits, the hit-table, and atlas lasso selections; saved views re-run a curated query from a
dropdown.

### The load-bearing decision: storage for mutable state
This is the **first mutable, user-authored state** in an otherwise read-only/derived Lance
world. **Recommendation: SQLite, not a Lance table.** One file, transactional, row-level
update, single-writer-friendly at 145k/single-user. This keeps the derived Lance datasets
immutable and reproducible (matches REPRODUCE), avoids `merge_insert`-on-mutable-state churn,
and sidesteps Lance version explosion. (See **cross-cutting** for the full trade-off.)

### Backend
New focused router `backend/tags/router.py` (mirrors `topics/router.py` / `atlas/router.py`:
`APIRouter(prefix="/api")`, `StateDep` reads, sync handlers threadpooled):
- `GET /api/tags?doc_id&speech_id&chunk_id` — tags for one chunk.
- `POST /api/tags` `{doc_id, speech_id, chunk_id, tags: list[str], op: "add"|"remove"|"set"}`.
- `POST /api/tags/bulk` `{keys: [...], tags, op}` — lasso→tag and table multi-select.
- `GET /api/tags/vocab` → distinct tag names + counts (autocomplete + filter dropdown).
New `backend/views/router.py`: `GET /api/views`, `POST /api/views {name, spec}`,
`DELETE /api/views/{name}`. All registered in `backend/app.py`.

**Storage** — new `backend/tags/store.py` opening one `<db_path>/curation.sqlite` (sibling of
`chunks.lance`) at startup in `backend/state.py:open_resources` → new `AppState.curation`
field. Schema:
```sql
tags(doc_id TEXT, speech_id INT, chunk_id INT, tag TEXT, created_at,
     PRIMARY KEY(doc_id, speech_id, chunk_id, tag))   -- one row per (chunk, tag)
views(name TEXT PRIMARY KEY, spec_json TEXT, created_at)
```
add/remove = single `INSERT OR IGNORE` / `DELETE`; set = `DELETE`+`INSERT` in a tx. Connect
`check_same_thread=False` + a module-level `threading.Lock` (or WAL).

**Join into search (read path):** in `backend/search/service.py`, after `_postprocess_hits`,
add `_attach_tags(curation, hits)` — a structural twin of `_attach_captions`: one
`SELECT … FROM tags WHERE (doc_id,speech_id,chunk_id) IN (…)`, group by `_chunk_key`, set
`hit["tags"] = [...]`. Thread `curation` through `run_search` like `chunk_frames`. `tags` is
**not** a Lance column, so `_PAYLOAD_COLUMNS` is untouched.

**Tag filter:** add `tags: list[str] | None` to `SearchSpec`. With no query, resolve matching
chunk keys from SQLite into the existing key-filter browse branch; combined with a query,
post-filter the hit list by membership (cheapest correct option — SQLite can't prefilter a
Lance vector scan), **over-fetching** when `spec.tags` is set to stay honest about `n`.

**Atlas write-back:** `atlas/router.py` already POSTs `_rowid`s; the chunks scan already
returns `doc/speech/chunk`, so resolve rowids → **stable keys** at write time and call the
bulk-tag store. **Never store `_rowid` in SQLite** (version-scoped).

**No FEATURES entry, no `raudio feature` step** — tags are mutable user state, not a derived
column. (Optional separate `raudio tag export`/`import` CLI to snapshot `curation.sqlite` →
JSONL for backup/repro.)

### Frontend
- `api.ts`: `TagsSchema`, `getTags`, `setTags`, `bulkTags`, `getTagVocab`,
  `listViews`/`saveView`/`deleteView`. `HitSchema` gains `tags: z.array(z.string()).optional()`;
  `SearchSpec` gains `tags?: string[]`. Wire `tags` into the GET/POST builders like `topic`.
- `active-filters.svelte`: a removable `tag:` pill per active tag (mirroring the Topic/Name
  pill block).
- `hit-card.svelte` + `hit-table.svelte`: render tag chips from `hit.tags` (beside the 🎬
  caption line; a new `tags` table column). A small "+tag" affordance calls `setTags`.
- `search-bar.svelte`: a "Views" split-button next to `FilterPopover` — a dropdown of saved
  views (click → apply stored spec + submit) plus "Save current view…" (name prompt →
  `saveView`).
- Atlas: `cross-filter.svelte.ts` already holds `selectedIds`; the page resolves them to
  `_rowid`s. Add "Tag selection" (→ `bulkTags`, keys resolved server-side) and "Save as view"
  to `AtlasMap.svelte`'s selection toolbar — closes the lasso→tag→view loop.
- `+page.svelte`: a tag-filter handler (mutate `spec.tags`, re-run) + refresh chips after a
  write (re-run or optimistic local update). New small `tag-input.svelte` (autocomplete over
  `getTagVocab`) reused by hit chips, bulk dialog, and the tag filter.

### Data-model impact
**No change to any Lance schema, no rebuild/migration** — the central honesty. One new mutable
`<db_path>/curation.sqlite` (two tables), created on first write — the **only** mutable file in
the system. A covering index on `(doc_id, speech_id, chunk_id)` makes the per-search join
O(hits). Hits gain a `tags` field (empty array when none) — additive, zod-optional. Because
tags are keyed by **stable** ids (not `_rowid`), a `chunks.lance` rebuild does **not** orphan
them. **Backup-critical:** REPRODUCE must list `curation.sqlite` explicitly (Path A restore).

**Rejected alternative — `tags.lance` + `merge_insert`:** (a) the repo already documents
`merge_insert` fragility; (b) every tag toggle bumps the dataset version → version sprawl;
(c) Lance has no transactional row update → concurrent toggles race; (d) SQLite is simpler,
transactional, one file — matches the DRY-but-simple ethos.

### Consequences & impacts
- **Perf:** one indexed SQLite query per search (O(#hits), sub-ms) — negligible. The
  tag-filter-with-query case post-filters in Python (over-fetch to stay honest).
- **UX:** a real curation loop — tag from search/table/lasso, filter by tag, save the view.
- **Schema migration:** none for Lance.
- **What could break:** a now-writable backend needs a defined path (db-sibling); concurrency
  is a non-issue at single-user but a Lock/WAL keeps two open tabs correct.

### Risks
- Storing `_rowid` for atlas bulk-tag would orphan tags on the next rebuild — **must** resolve
  `rowid → (doc_id,speech_id,chunk_id)` at write time and store only the stable key.
- Tag-filter + vector/FTS query can't be SQLite-prefiltered — over-fetch or a key-set
  where-clause to avoid under-filling the page.
- A `tags.lance` + `merge_insert` would version-sprawl and inherit documented fragility — the
  plan deliberately avoids this.
- `curation.sqlite` is backup-critical, non-derived — REPRODUCE/runbook must list it (Path A).
- A saved view = a frozen `SearchSpec`; if `SearchSpec` evolves, old views could fail zod —
  store loose/forward-compatible JSON and validate leniently on apply.
- Tag write-back needs an optimistic-update or re-fetch path or chips go stale — pick one.

### Effort
**M.**

### Modality notes
Tags attach to the **chunk** identity `(doc_id, speech_id, chunk_id)`, which is
modality-agnostic — the same row is the unit for text, visual/scene, and the planned
audio/document spaces. So tags work uniformly across every search mode with **zero**
per-modality code (the join keys on the chunk, not the embedding column). The atlas
`space=text|visual|caption` selector is orthogonal — a lasso in any space resolves to the same
chunk keys. Saved views capture the `mode` (and thus modality) inside the stored `SearchSpec`,
so a restored view reproduces its original modality. Tags are the one cross-modality primitive.

### Open questions
- Confirm SQLite over a Lance tags table? (Strong recommendation: SQLite.)
- `curation.sqlite` location — db-sibling (travels with the corpus) vs user-config dir
  (survives db swaps)? Db-sibling matches "tags keyed to this corpus."
- Should saved views persist the result-list view mode (list/grid/table/map) + table columns,
  or only the `SearchSpec`? (FiftyOne `save_view` is query-only.)
- Tag-filter-with-query: post-filter (simple, may under-fill) vs resolve-to-keys-and-prefilter
  (correct counts, more code) — which first?
- Multi-select in the hit-table for bulk tagging, or lasso + single-hit tagging enough for v1?
- A tag colour channel on the atlas (`colorBy: 'tag'`)? Nice but a follow-up.

---

## Cross-cutting concerns

### (a) Per-modality embedding columns are the shared lever
Every retrieval feature here keys off **one embedding column per modality**, selected by a
`space` parameter that reuses the atlas `--space` machinery verbatim
(`FeatureRunOptions.space` Literal → per-space output namespacing → `_run_*` router switch).
This is why uniqueness, more-like-this, and the cluster facet all "just work" across
`text|visual|caption` with no per-modality branches, and why `voice`/`document` are usually a
**one-line** addition to a `space → column` map the day those columns exist. Do not invent
spaces outside the 4-modality doctrine.

### (b) The big one — mutable user state in a read-only/derived Lance world
Tags + saved views (#5) are the **only** features that write user-authored state. Everything
else in raudio is immutable/derived/reproducible Lance. The recommended split:

| store | best for | why / why not |
| ----- | -------- | ------------- |
| **Lance table** (`tags.lance` + `merge_insert`) | ❌ | version sprawl per toggle, documented `merge_insert` fragility, no transactional row update, races |
| **SQLite** (`curation.sqlite`) | ✅ tags + views | one file, transactional, row-level add/remove/set, single-writer-friendly, keyed by *stable* chunk ids so it survives a Lance rebuild |
| **flat JSON** | ⚠️ views only | fine for a handful of view blobs, but a second mechanism — fold into the same SQLite to avoid sprawl |

Consequences of going mutable: a defined, backup-critical file path (db-sibling); REPRODUCE
Path A must list `curation.sqlite` (it is **not** rebuildable like Lance columns); writes must
resolve `_rowid → stable key` (never persist version-scoped rowids); a Lock/WAL for two-tab
correctness.

### (c) Index / rebuild costs & compaction caveats
- **Derived columns** (`uniqueness`/`dup_group`) are **global fits** — all-or-nothing
  `--all`/overwrite rebuilds, **not** incrementally fill-NULL-able like `text_embedding`. New
  ingested chunks stay NULL until a full rebuild; the read path treats them as
  optional/nullable so it degrades gracefully. Ids like `dup_group`/`atlas_cluster` are
  **non-deterministic across rebuilds** — only safe used within a single result list, never
  persisted as references.
- **Attach is metadata-only.** All new columns use the `projection._attach_column_by_row_id`
  `add_columns`-by-`_rowid` path — new column files, **no fragment rewrite, no `merge_insert`**
  — the same safe path the atlas/projection columns use on this wide schema.
- **No new vector index** for uniqueness (scalar columns). More-like-this reuses the existing
  IVF indexes; stats builds none.
- **Projection footgun:** selecting a missing Lance column **errors** (not silent). Optional
  columns (`uniqueness`/`dup_group`, future facets) **must** be presence-gated on
  `schema.names`, exactly like `caption`, or searches 400 on un-built DBs.
- **Compaction:** metadata-only adds leave the data fragments untouched; a later `compact`
  rolls the new column files into fragments normally — no special handling, but a rebuild of a
  global-fit column should be followed by the usual maintenance, not assumed live.

### (d) Single-user-scale honesty (YAGNI)
At single-user / 145k these designs are right **because** the corpus fits cheap full scans and
in-memory passes:
- Stats stays a **live** endpoint (5–20 ms) — no precompute, no cache, no derived column.
- Uniqueness is a one-time **offline** build (~3.4 min CPU / seconds GPU); search-time dedup is
  O(n ≤ 200).
- Tags are a SQLite-and-a-router feature — no service, queue, or Lance write path.
- Group-by-video is pure presentation — no server-side group pagination or group-count
  endpoint.

Explicitly **not** building for multi-tenant / 10–100M rows (precomputed aggregates, pushdown
stats, distributed kNN). That would be over-engineering here and is out of scope.

---

## Explicitly NOT doing (FiftyOne features that don't fit)

- **Model evaluation / metrics** (confusion matrices, mAP, P/R curves) — no ground-truth
  labels or detection task here; this is a retrieval/exploration tool, not a benchmark harness.
- **Annotation-tool integrations** (CVAT/Labelbox round-trips) — out of scope; tags (#5) are
  the deliberately minimal in-app curation primitive.
- **Label-quality / "Brain" mistakenness** scoring — depends on labels we don't have; the
  uniqueness brain-equivalent (#2) is the one piece that applies and is scoped in.
- **Geo / map (lat-long) visualization** — no geospatial data; the only "map" is the embedding
  atlas.
- **3D / point-cloud / mesh** modalities — outside the 4-modality doctrine.
- **Full plugin framework / operator SDK** — over-engineering at single-user; new features are
  a focused router + (optionally) one FEATURES entry, not a plugin runtime.
