# Merging ra-anno (document/HTR/OCR viewer + annotator) into lance-media

Assessment + architecture for folding **ra-anno**
(github.com/AI-Riksarkivet/ra-anno) into this repo, and splitting the result into
**three services + one frontend** on lance-ray / Lance / WebGPU / Arrow.

## 0. First: what is actually agnostic (stop overselling it)

| Plane | Agnostic? | Reality |
|---|---|---|
| **Compute / stages** | **Yes — the declaration** | The 3 stage shapes describe a transform (read cols/blobs → produce cols/rows) without saying *how* it runs. lance-ns proves it: the SAME transform runs in-process (`transform_stage`) OR as a distributed Ray Data job (`ray_stage_job`). Ray Data is a **pluggable executor**, not baked in. **Caveat:** our *current* ratch driver IS coupled to Ray Data (ray.data actors) — the shape is executor-agnostic, the runner we shipped is Ray Data. |
| **Data / search** | **Yes** | descriptor-driven backend + type-driven search (FTS / vector / hybrid) work over any Lance table; a new embedding column is searchable with zero edits. HTR text + page-image embeddings fit this. |
| **Viewer** | **NO (this was the oversell)** | lance-media's frontend is a `<video>` player with a **temporal** playhead + temporal-only alignments (`{start,end}` seconds). It cannot render a page image, and has no **spatial** (bbox/polygon) overlay. For HTR/OCR/documents it is the wrong viewer. |

**ra-anno is exactly the missing viewer plane** — and it's built on the stack we
want, so it's a fit, not a graft.

## 1. What ra-anno is (verified from source)

A document annotation platform for Riksarkivet — **replaces FiftyOne (dataset
browser) + Label Studio (annotation)** with one Arrow-native app.

- **Stack:** SvelteKit 2 · Svelte 5 (runes) · **PixiJS v8 (WebGPU)** · **Apache
  Arrow v21** · Tailwind 4 · bits-ui/shadcn — i.e. our shell, plus a real GPU
  document engine. No IIIF/OpenSeadragon (good — you didn't want that).
- **Engine** (`src/lib/engine/pixi`, plain TS classes):
  - `ImagePlugin` — loads the page image as a `Texture`, owns viewport zoom/pan,
    `isRenderGroup=true` for GPU-accelerated transforms, feeds viewport bounds for culling.
  - `ArrowDataPlugin` — **columnar, Rerun-inspired**: reads geometry columns as
    zero-copy `Float32Array` views, batches by status color into ~5 `Graphics`
    (5 draw calls for 5 or 5000 boxes), viewport-culls, polygon access via flat
    `valueOffsets`+`values` `subarray()`.
  - `AnnotationPlugin` — draw state machine (rect/polygon), hit-testing, cached
    `getBoundingClientRect` via ResizeObserver.
- **Data model — 3 tables, zero joins:** `documents` · `pages` (image column) ·
  `annotations` (`x,y,width,height` Float32, `polygon` `list<float32>`,
  `text/label/status` Utf8, `confidence`, **`embedding` added later via
  `add_columns()`**).
- **Arrow IPC everywhere** — no JSON for bulk data. Server streams Arrow IPC →
  browser incremental `tableFromIPC` → `$state.raw` → `$effect` → PixiJS. 50K
  annotations render progressively.
- **Local-first editing** — an **overlay/WAL** over the immutable server Arrow
  table (`fieldOverrides` / `appendedRows` / `deletedIndices`); field edits are
  O(1) with no rebuild; undo/redo = Arrow snapshots (~5KB each); **save = delta
  Arrow IPC → `Lance merge_insert("id")` → one atomic version** (If-Match ETag).
- **Data access, open question they left:** a **Flight SQL server** (pyarrow.flight,
  warm NVMe cache, shared with batch pipelines) vs the **`@lancedb/lancedb` TS SDK**
  direct from SvelteKit. They keep Flight SQL for scale + cache coherency with the
  batch writers; TS SDK for light metadata/dev.

**Verdict:** ra-anno is more aligned with the lakehouse target than our current
frontend — it's the spatial/document half we lack, already Arrow+WebGPU+Lance.

## 2. Recommendation: fold ra-anno's ENGINE in; keep lance-media's search/atlas/compute

Don't drop lance-media — its search, atlas, descriptor backend, and ratch
pipeline are hard-won. Don't keep two repos either. **Adopt ra-anno's PixiJS+Arrow
engine as the viewer/annotator core, and bring lance-media's search + atlas +
compute alongside it.** They are complementary, not competing:

| lance-media brings | ra-anno brings |
|---|---|
| type-driven **search** (FTS/vector/hybrid) | **PixiJS document viewer** (image, GPU zoom/pan) |
| **atlas** embedding scatter (WebGPU) | **spatial annotation** (bbox/polygon draw+edit+hit-test) |
| descriptor-driven rendering, topics, KG | **Arrow-IPC-everywhere** zero-copy data path |
| the **ratch pipeline** (bronze→silver stages) | **local-first** editing + undo/redo + `merge_insert` save |
| AV player (keep for audio/video corpora) | replaces FiftyOne + Label Studio |

The descriptor picks the viewer per corpus: **AV corpus → the `<video>` player;
document/HTR/OCR corpus → the PixiJS page viewer + annotation overlay.** One shell,
two viewers, chosen by `document.mime` + capabilities.

## 3. The three services + frontend

```
                         ┌──────────────── FRONTEND (SvelteKit 2 · Svelte 5) ───────────────┐
                         │  ONE PixiJS v8 (WebGPU) + Arrow engine, one shell (bits-ui/shadcn) │
                         │  document viewer · annotation canvas · search UI · embedding atlas │
                         └───────┬───────────────────┬────────────────────┬─────────────────┘
                                 │ Arrow IPC          │ Arrow IPC          │ Arrow IPC
                    ┌────────────▼───────┐ ┌──────────▼─────────┐ ┌────────▼───────────┐
                    │ 1. VIEWER service  │ │ 2. SEARCH service  │ │ 3. ANNOTATOR svc   │
                    │ pages + image bytes│ │ FTS/vector/hybrid  │ │ annotation CRUD    │
                    │ + annotation READ  │ │ (services/search)  │ │ local-first + save │
                    │ (Arrow IPC stream) │ │ over Lance         │ │ → merge_insert(id) │
                    └────────────┬───────┘ └──────────┬─────────┘ └────────┬───────────┘
                                 └──────────── read/write ─────────────────┘
                                        Lance tables (bronze/silver/gold)
                                        via the lance-ns CATALOG (governed)
                    compute: ratch STAGES as lance-ray jobs produce pages/lines/annotations/embeddings
```

- **1 · Viewer** — serves `pages` (image bytes, immutable-cached, Range) + the
  `annotations` **read** as streamed Arrow IPC. The document display + progressive
  overlay. (ra-anno's `/api/images` + `/api/annotations` GET.)
- **2 · Search** — our existing `search_api` (FTS / vector / hybrid over Lance,
  descriptor-driven, the version-keyed result cache). Finds pages/lines/docs.
- **3 · Annotator** — annotation **write**: the local-first overlay → delta Arrow
  IPC → `Lance merge_insert("id")` → atomic version, with the QC path feeding gold.
  This is where model **predictions** land too (status="prediction") for human review.

Each service is a **catalog read/write client** (governed, per-table creds — no
static S3 keys). Backends: our FastAPI is fine for **search** (Python, pylance,
vLLM-integrated); the **viewer/annotator** want the Arrow-IPC + Flight-SQL warm
cache path ra-anno designed — decide FastAPI-Arrow vs a Flight SQL sidecar in §5.

## 4. The embeddings viewer — decision

Options: (a) our **custom WebGPU instanced-quad scatter** (`gpu-scatter.svelte`,
WGSL, renders 145k pts), (b) **PixiJS-rendered points** via the same
`ArrowDataPlugin` columnar-batch pattern, (c) Apple's **embedding-atlas
EmbeddingView** — already **rejected** (dense-palette render crash, heavier dep).

**Recommendation: unify on the PixiJS + Arrow engine (b).** The scatter is the
*same problem* as the annotation canvas — read `Float32Array` x/y columns, render
instanced primitives, viewport-cull, hit-test via a spatial grid — so rendering it
through `ArrowDataPlugin` (points instead of boxes) collapses **two GPU renderers
into one engine, one Arrow data path**. Keep the custom WGSL scatter only if
profiling shows PixiJS instancing can't hold the point count you need (millions) —
**measure before committing**; at 145k either handles it.

## 4a. The audio viewer — peaks.js (the AV analogue of the page viewer)

Audio needs its own region-aware viewer exactly as documents need the page/bbox
viewer. **Decision taken (shipped): wavesurfer.js** — its regions plugin drives the
shipped audio annotator (`frontend/apps/annotator`), and the temporal facet landed
exactly as sketched below. peaks.js was the original recommendation and was
rejected in implementation (wavesurfer's plugin set + decode path fit the shipped
corpus). The original rationale, kept for the record:

- It is **annotation-first** — labeled time **segments** + **points** are the core
  primitives (BBC built it for spoken-word *archive* annotation), which is our exact
  use: speaker turns, ASR chunks, human-labeled regions.
- It supports **precomputed waveform data** — so we add a **`waveform` silver stage**
  (bronze audio → min/max peaks) and stream compact peaks (Arrow), instead of
  decoding gigabytes of archival audio in the browser. This is the audio parallel of
  the `htr` stage producing bbox rows.
- Temporal annotations reuse the SAME annotator service + model as spatial ones —
  a "region with a label", geometry = `start/end` (time) instead of `x/y/w/h`
  (space). One annotation table, one save → `merge_insert` path.

wavesurfer.js is the alternative if you want its spectrogram/minimap plugins and the
files are short enough to decode client-side. (A third option — a PixiJS-native
waveform on the same Arrow engine — unifies everything but is more to build; defer.)

## 5. Compute mapping (this closes the loop with the earlier corrections)

- **bronze** = raw media Lance tables you send in (page images, audio, video, maps).
- **silver** = ratch **stages** as lance-ray jobs:
  - `htr` / `ocr` (`APPEND_ROWS`: page-image blob → `annotations`/`lines` rows with
    text + **bbox/polygon**) — the same shape as `transcribe` (audio→chunks).
  - `embed` (`SCAN_COLUMN`/`BLOB_COLUMN`: text/line/region → `+embedding`).
  - atlas / topics / KG — also silver (aggregation ≠ a tier).
- **gold** = QC-gated, curated promotion of silver (validator `can_promote` +
  quality gate) — the trustworthy surface the 3 services read.
- **model predictions** written straight to `annotations` (status="prediction")
  are just another silver writer; the annotator service serves them for human
  correction, and the human's save is a new version — the human-in-the-loop is
  lineage, not a side channel.

So HTR/OCR is not a special pipeline — it's `bronze image → htr stage → annotations
(silver) → viewer/annotator services`, the identical machinery as audio→chunks.

## 5c. Annotation across modalities — can we, and what's needed

The engine's model (`schema.ts`) is genuinely document-grade: `shape_type` ∈
{rectangle, rotation(oriented), polygon, baseline(HTR line), line, point, mask},
plus `text/label/status(prediction→reviewed→accepted)/group_id(link lines→region)/
difficult/mask/metadata`. But it is **spatial-image ONLY** — zero time/video.

| Capability | Now? | Notes |
|---|---|---|
| **Bulk selection** | ✅ shipped | LassoTool → `selectedSet` → `highlightSet`, `batchUpdateLocal` + the sidebar bulk edit |
| **Annotate on audio** | ✅ shipped | wavesurfer.js regions + the `t_start/t_end` temporal facet; same store + annotator service (region+label) |
| **Draw on video (spatial, on a frame)** | ✅ shipped | `<video>` + the PixiJS overlay synced to `currentTime`; shapes pin to the playhead (`t_start`) — E2E-proven |
| **Frames that draw on video** | ✅ shipped | the frame path IS the image path; shapes overlay back at their time |
| **Track/interpolate across frames (CVAT-style)** | ❌ future | keyframe interpolation between annotated frames using per-object tracks (`group_id` exists as the track key) — the one genuinely open item |

**The unifying idea:** keep ONE `annotations` table + ONE annotator service. Extend
the model with an optional **temporal facet** (`t_start/t_end` for segments, `t`/
`frame_idx` for a shape pinned to a video moment). `shape_type` + the media kind
disambiguate. Spatial-only rows are unchanged (documents/HTR); audio/video rows add
the time columns. No fork of the data model or the service.

## 5d. The layout + service segmentation

**Frontend = one shell + a viewer REGISTRY keyed by media kind** (from the
descriptor's `document.mime`), all sharing `AnnotationStore` + `AnnotationSidebar`
+ `Toolbar` + the annotator service:

```
/annotate/[dataset]/[docId]/[unit]
   ├─ image/PDF  → ImageCanvas   (ra-anno PixiJS engine, as-is)
   ├─ audio      → WaveformCanvas (peaks.js — temporal segments/points)
   └─ video      → VideoCanvas    (<video> + PixiJS overlay @currentTime + a timeline)
        shared: AnnotationStore · Sidebar · Toolbar · undo/redo
```

**Services (the 3, unchanged by modality):**
1. **Viewer** — media bytes (image/audio/video, Range) + annotation READ (Arrow IPC).
2. **Search** — FTS / vector / hybrid.
3. **Annotator** — annotation CRUD (Arrow IPC → `merge_insert`); model predictions
   (`status="prediction"`) land here for human review — SAME table, SAME service,
   for all three modalities.

**Compute (silver stages) feed the annotator:** `htr`/`ocr` (image→regions),
`asr` (audio→segments), `extract_frames` (video→frames), `embed`. They WRITE
predictions into the one `annotations` table; humans correct → a new Lance version.

So the segmentation is: **one uniform annotation model + one annotator service; the
frontend segments by a per-modality VIEWER component.** That's the only axis that
forks — and it forks in the view layer, exactly where it should.

## 6. Decisions taken

1. **FastAPI, not Flight SQL** — the three FastAPI services serve the Arrow path
   (annotations ride Arrow IPC over HTTP); a Flight/cache sidecar remains a
   merge-time option for the hot image path, not built here.
2. **Bun** — standardized (turborepo + adapter-bun).
3. **One repo, one frontend** — ra-anno's engine model folded in as
   `frontend/packages/engine`; the AV player stays; the descriptor selects.
4. **Spatial-alignment schema** — ra-anno's `annotations` columns adopted as the
   contract (see `ANNOTATIONS_SCHEMA_CONTRACT.md`), temporal facet included.
