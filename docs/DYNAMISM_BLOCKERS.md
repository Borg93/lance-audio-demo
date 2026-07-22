# Dynamism blockers — the hardcoding inventory

> A code-grounded audit of **every place that must be edited to add a column, an
> embedding/vector space, a modality, a filter, a search mode, or a per-hit
> action** — i.e. what blocks the dynamic, modality-agnostic, schema-driven model
> in [WHATS_LEFT §0/§5](WHATS_LEFT.md) and the low-coupling design in
> [MODULAR_PLAN.md](MODULAR_PLAN.md). Findings are from a full read of `backend/`,
> `frontend/`, and `src/ratch/{model,features,ingest,cli}`. Three kinds of
> coupling, deepest first:
>
> - **Modality** — the data model assumes *audio/transcript* (deepest; blocks
>   document/image/pure-audio as peers).
> - **Capability** — *what you can do* (search modes, filters, actions) is bolted
>   to named columns (blocks new operations).
> - **Display** — *what's shown* is a fixed column list (blocks new columns
>   surfacing).

> Status: this is an **inventory**, not a changelog — nothing here is fixed yet.
> Line numbers are as of the audit; treat them as anchors, not guarantees.

---

## The five root-cause patterns

Almost every blocker is one of these. Fix the pattern, not the instance:

1. **Literal column names** in `.select([...])` / `columns=[...]` / field access
   → derive the column set from `dataset.schema`, excluding by **role**
   (blob/embedding/internal), never by name.
2. **Closed enums of operations** (`SearchMode`, atlas `ColorBy`, filter ops)
   → derive operations from **column roles** (`embedding`→`vector:<col>`,
   `fts`→keyword, `facet`→filter).
3. **Modality assumptions baked into the schema** (`speech_id`, `audio_path`,
   `alignments_json`, single `EMBED_DIM`) → a generic media-row model + an
   embedding-space registry; audio fields become *optional, role-tagged*.
4. **Hardcoded table names/paths** (`db / "chunks.lance"`, `table="chunks"`)
   → a `DatasetRegistry` + `TableDescriptor` (see MODULAR_PLAN §4).
5. **Duplicated literals across layers** (search modes live in the backend enum
   *and* 3 frontend files) → one source of truth, served from an endpoint.

---

## Layer 1 — Schema / data model (deepest: modality coupling)

`src/ratch/model/schema.py`, `features/columns.py`, `ingest/ingest.py`.

| # | Blocker | file:line | Blocks | Fix |
|---|---|---|---|---|
| 1 | **Single fixed `EMBED_DIM = 2048`** — every vector column + `_vectors_to_arrow` validates this one dim | `model/schema.py:36`; `features/columns.py:46,49-54` | a 2nd embedding **space** at another dim (e.g. 768-d doc vectors, Jina-omni); switching models mid-corpus | an **EmbeddingSpace registry** `{name, dim, model_id}` as table metadata; features declare their space |
| 2 | **Audio-centric `CHUNK_SCHEMA`** — `speech_id`, `chunk_id`, `audio_path`, `audio_duration`, `audio_frames`, `sample_rate`, `language_prob`, `alignments_json` | `model/schema.py:64-101` | ingesting documents/images without dummy audio fields; any non-speech modality | a **generic media-row model**: identity/source/timing/content/metadata *roles*; audio fields optional |
| 3 | **Chunk identity `(doc_id, speech_id, chunk_id)`** hardcoded as THE key | `features/columns.py:43-45` (`CHUNK_KEYS`/`FRAME_KEYS`), used in every join/null-fill | a document key `(doc_id, para_id)` or image regions | keys declared per-table in a `TableDescriptor`, not a module constant |
| 4 | **`DOC_SCHEMA` = one media URI + one thumbnail per row** | `model/schema.py:130-155` | multi-page PDFs, image stacks, multi-media docs | a one-to-many `media` table `(doc_id, idx, type, blob)` |
| 5 | **Speaker tables assume audio identity** (`speaker_label`, abs `start/end`, pyannote) | `model/schema.py:211-286` | author/entity identity for docs; person-in-image | generic `EntityEmbedding{entity_type, space, vec}` |
| 6 | **`flatten_chunks` iterates speeches→chunks with alignments** | `ingest/ingest.py:165-202` | document/image ingest (needs dummy speech structure) | `flatten_media_rows(doc, descriptor)` by role; alignments optional |

## Layer 2 — Enrichment / features (no DAG; table+client coupling)

| # | Blocker | file:line | Blocks | Fix |
|---|---|---|---|---|
| 7 | **`Feature` has no declared `inputs`/`outputs`/`depends_on`** — dependencies invisible, cascades fail at runtime | `features/columns.py:325-333` (`Feature` model) | a planner/Ray driver that walks the enrichment DAG incrementally | add `inputs/outputs/depends_on` to `Feature` (MODULAR_PLAN §3b) |
| 8 | **Each `Feature` hardcodes `table=` + path** `db / "chunks.lance"` / `"chunk_frames.lance"` | `features/columns.py:587-646, 349, 424` | adding a 3rd table (e.g. `doc_chunks`) without new `_run_*` fns | `TableDescriptor` carries name+path; `run` receives it |
| 9 | **Fixed column-name constants** `TEXT_EMBED_COLUMN`… input/output names inline | `features/columns.py:37-41` | two embeddings on one table; renames; multi-space | `ColumnSpec` inputs/outputs per stage |
| 10 | **Each `_run_*` imports + builds its own vLLM client by URL** | `features/columns.py:342-459` | swapping models; sharing a client/actor pool; Ray Serve | a **ModelRegistry**; stages declare `required_models`, clients injected |
| 11 | **Modality-specific CLI** `extract-chunk-frames` / `extract-speaker-turns` assume ffmpeg+MP4 + chunk keys | `cli/media.py:88-236, 238-415` | frame/representative-image extraction for PDFs/images | modality handlers registered by type (audio/doc/image) |

## Layer 3 — Backend / query (capability coupling)

`backend/search/*`, `atlas/`, `voice/`, `graph/`, `system/`, `state.py`.

| # | Blocker | file:line | Blocks | Fix |
|---|---|---|---|---|
| 12 | **`_HIT_COLUMNS` / `_PAYLOAD_COLUMNS`** fixed projection on every hit | `search/constants.py:10-34`; `.select(_HIT_COLUMNS)` `search/service.py:122,239` | new metadata column surfacing in results | derive `SELECT` from schema minus blob/embedding roles |
| 13 | **`SearchMode` closed enum** (fts/semantic/visual/scene/scene_fts/hybrid/all) | `search/spec.py:16-24` | a new vector space being searchable at all | role-derived `vector:<col>` / `fts:<col>` dispatch |
| 14 | **`_MODE_HANDLERS` dispatch dict + per-mode fns naming a column** — `_search_semantic`→`text_embedding`, `_search_hybrid`→`vector_column_name="text_embedding"`, visual→`frame_embedding`, scene→`caption_embedding` | `search/service.py:155,232,~323-331`; `frames.py:82` | new embedding column = backend edit in 5+ spots | one generic vector-search fn parametrized by column (from role) |
| 15 | **Fixed filter fields** `language/namn/referenskod/extraid/topic` in spec + builder | `search/spec.py:43-50`; `search/filters.py:15-46` | new facet filterable without 3 edits | build filters from `facet`-role columns |
| 16 | **Atlas space triplets** `atlas_*`/`atlas_img_*`/`atlas_cap_*` + `--space` literals | `atlas/points.py:27-31` | a new projection space appearing on the map | register `{space → x/y/cluster cols}`, advertise via API |
| 17 | **Voice column lists** `_TURN_HIT_COLUMNS`, `_IDENTITY_COLUMNS`; `vector_column_name="embedding"` | `voice/service.py:63,67,143` | new voice metric/space | schema-derived, like §12 |
| 18 | **Graph node→file map** `{"Entity":"kg_entities.lance", …}` | `graph/router.py:46-52` | new node type without a manual entry | derive from a KG table registry |
| 19 | **`state.py` opens a fixed set of tables by literal name/path, handle pinned at startup** | `backend/state.py:open_resources` | new tables; seeing Ray-added columns without restart | `DatasetRegistry` + `checkout_latest()` refresh |

## Layer 4 — Frontend (capability + display coupling)

`frontend/src/lib/*`, `routes/*`. **All hand-written zod — no OpenAPI codegen.**

| # | Blocker | file:line | Blocks | Fix |
|---|---|---|---|---|
| 20 | **`HitSchema` / `DocumentSchema` / `VoiceHitSchema`** fixed zod structs (~17 named fields) | `lib/api.ts` (HitSchema; Document; Voice) | any new column reaching the UI | typed core + open `fields` map; generate from OpenAPI |
| 21 | **`TABLE_COLUMNS` + `DEFAULT_TABLE_COLS` + `MAP_TABLE_COLS` + `WRAP_KEYS`** hand-listed | `lib/components/hit-table.svelte:25-92,150`; `routes/+page.svelte:123,173-184` | new column in the table/map | render from `/api/schema` roles |
| 22 | **Search modes duplicated in 3+ places** — `SearchModeSchema` enum + `SEARCH_MODES` array + per-component copies | `lib/api.ts:16-24`; `lib/workflow/graph.svelte.ts:68-76`; search-bar/SearchNode/search-settings | new mode usable in the UI | one `/api/search-modes` endpoint, render selector from it |
| 23 | **Audio playback tied to `audio_path`/`doc_id`** | `hit-table.svelte:363-401`; `hit-card.svelte:105-141` | non-audio modality (invisible/broken play) | role `media,kind=audio` → conditional affordance |
| 24 | **Frame thumbnail tied to `speech_id/chunk_id/frame_blob`** + fixed `/api/chunk-frame/{…}` URL | `hit-card.svelte:161-179` | image/doc corpora without per-chunk frames | schema advertises frame endpoint + key fields |
| 25 | **Karaoke word-render tied to `alignments_json`** (`alignments[].words[].start/end`) | `transcript-highlighter.svelte` (word render); alignment plumbing in `player-pane.svelte:46-72` | non-speech corpus (blank transcript) | role `alignments,modality=audio` → conditional |
| 26 | **Detail rows + card title hand-written** (`player-pane` 10 fields; `hit.namn ?? audio_path ?? doc_id`) | `player-pane.svelte:221-238`; `hit-card.svelte:35` | new field in detail/title | render from schema; `role=title` |
| 27 | **Active-filter pills + quick filters fixed** (`language/namn/referenskod/extraid/topic`; `sv/en` dropdown) | `active-filters.svelte:14-26`; `filter-popover.svelte:233-244` | new filter in the pill bar | derive from facet roles |
| 28 | **Atlas `ColorBy` enum + legend titles hardcoded** (`cluster/language/topic/doc_topic/doc`) | `AtlasMap.svelte:~228-239,423`; `atlas-legend.ts:139-149` | new categorical color dimension | projection advertises `color_columns` |
| 29 | **Voice speaker chips tied to `speaker_label/turn_*`; confidence bands hardcoded** | `hit-table.svelte:48-51`; `api.ts:1024-1028` | non-voice corpus (blank chip); retuning | `modality=voice` role; `/api/voice/config` |
| 30 | **Workflow node registry hardcoded** (`query/image/filter/atlas/search/combine/tagger/results/export`) | `lib/workflow/node-types.ts:13-23` | a new node type without code | runtime/plugin registration (lower priority) |

---

## Worst offenders (fix these and most of the rest follow)

1. **`state.py` pinned handle + no `DatasetRegistry`** (#19) — gates *everything*
   dynamic; nothing sees a new column/table without it.
2. **`SearchMode` enum + per-mode column dispatch** (#13/#14) — the capability
   coupling; new vector space is dead weight until fixed.
3. **`_HIT_COLUMNS`** (#12) + **`HitSchema`/`TABLE_COLUMNS`** (#20/#21) — the
   display path, backend→frontend.
4. **`Feature` has no DAG** (#7) + **fixed `EMBED_DIM`/`CHUNK_SCHEMA`** (#1/#2) —
   the modality/enrichment foundation for "multimodal in general."
5. **Hand-written zod, no OpenAPI codegen** (#20) — every backend change is a
   manual frontend re-type.

---

## Phased fix roadmap

Each phase removes a *pattern*, not a list of instances.

- **Phase 0 — the schema seam (unblocks all):** `DatasetRegistry` + `/api/schema`
  (roles) + `checkout_latest()` refresh (#19); typed-core+`fields` payload (#20);
  derive `SELECT` from schema (#12). No GPU, low risk.
- **Phase 1 — display dynamism:** render `TABLE_COLUMNS`, detail rows, cards,
  pills from `/api/schema` (#21,#26,#27); generate TS client from OpenAPI (#20).
- **Phase 2 — capability dynamism:** retire `SearchMode` enum → role-derived
  `vector:<col>`/`fts:<col>` dispatch (#13,#14,#15); `/api/search-modes` to kill
  the 3× frontend duplication (#22); atlas color/space from advertised roles
  (#16,#28).
- **Phase 3 — modality dynamism:** `EmbeddingSpace` registry (#1); generic
  media-row model + role-tagged optional audio fields (#2,#3,#4,#5,#6); modality
  affordances on the frontend (#23,#24,#25,#29).
- **Phase 4 — enrichment dynamism:** `Feature` → DAG with `inputs/outputs`
  (#7,#9); `TableDescriptor` (#8); `ModelRegistry` (#10); modality handlers (#11)
  — this is where the Ray data-evolution layer (§1) lands.

The throughline matches MODULAR_PLAN: **roles + descriptors + a schema endpoint**
replace **literals + enums + fixed paths**, so adding a column/space/modality is
data, not a code edit.
