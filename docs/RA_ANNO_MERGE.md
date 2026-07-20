# Merging ra-anno (document/HTR/OCR viewer + annotator) into lance-media

Assessment + architecture for folding **ra-anno**
(github.com/AI-Riksarkivet/ra-anno) into this repo, and splitting the result into
**three services + one frontend** on lance-ray / Lance / WebGPU / Arrow.

## 0. First: what is actually agnostic (stop overselling it)

| Plane | Agnostic? | Reality |
|---|---|---|
| **Compute / stages** | **Yes — the declaration** | The 3 stage shapes describe a transform (read cols/blobs → produce cols/rows) without saying *how* it runs. lance-ns proves it: the SAME transform runs in-process (`transform_stage`) OR as a distributed Ray Data job (`ray_stage_job`). Ray Data is a **pluggable executor**, not baked in. **Caveat:** our *current* rmedia driver IS coupled to Ray Data (ray.data actors) — the shape is executor-agnostic, the runner we shipped is Ray Data. |
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

Don't drop lance-media — its search, atlas, descriptor backend, and rmedia
pipeline are hard-won. Don't keep two repos either. **Adopt ra-anno's PixiJS+Arrow
engine as the viewer/annotator core, and bring lance-media's search + atlas +
compute alongside it.** They are complementary, not competing:

| lance-media brings | ra-anno brings |
|---|---|
| type-driven **search** (FTS/vector/hybrid) | **PixiJS document viewer** (image, GPU zoom/pan) |
| **atlas** embedding scatter (WebGPU) | **spatial annotation** (bbox/polygon draw+edit+hit-test) |
| descriptor-driven rendering, topics, KG | **Arrow-IPC-everywhere** zero-copy data path |
| the **rmedia pipeline** (bronze→silver stages) | **local-first** editing + undo/redo + `merge_insert` save |
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
                    │ + annotation READ  │ │ (our search_api)   │ │ local-first + save │
                    │ (Arrow IPC stream) │ │ over Lance         │ │ → merge_insert(id) │
                    └────────────┬───────┘ └──────────┬─────────┘ └────────┬───────────┘
                                 └──────────── read/write ─────────────────┘
                                        Lance tables (bronze/silver/gold)
                                        via the lance-ns CATALOG (governed)
                    compute: rmedia STAGES as lance-ray jobs produce pages/lines/annotations/embeddings
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

## 5. Compute mapping (this closes the loop with the earlier corrections)

- **bronze** = raw media Lance tables you send in (page images, audio, video, maps).
- **silver** = rmedia **stages** as lance-ray jobs:
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

## 6. Open decisions (call these before building)

1. **Flight SQL server vs `@lancedb/lancedb` TS SDK vs our FastAPI** for the
   viewer/annotator Arrow path. Flight SQL gives a warm NVMe cache shared with the
   batch writers (coherency); FastAPI keeps one backend language. Likely: FastAPI
   for search, a Flight-SQL-or-Arrow sidecar for the hot annotation/image path.
2. **Deno vs Bun** — ra-anno's docs say Deno, its lockfile says Bun; lance-media is
   Bun (adapter-bun). Standardize on **Bun**.
3. **One repo, one frontend** — merge ra-anno's `src/lib/engine` in as the viewer
   core; the AV player stays for audio/video corpora; descriptor selects.
4. **Annotation spatial-alignment schema** — adopt ra-anno's `annotations` columns
   as the descriptor's spatial-alignment capability (the analogue of our temporal
   `alignments`), so the backend/frontend stay descriptor-driven.
