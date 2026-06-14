# raudio vs. LightlyStudio — what we do better, what we can do better

> A code-grounded comparison against [LightlyStudio](https://github.com/lightly-ai/lightly-studio)
> (the data-curation product) and its [`lightly`](https://github.com/lightly-ai/lightly)
> SSL library. Written after reading both codebases in full — storage layer,
> backend services, frontend, and the SSL engine — not the READMEs. Goal:
> understand where our **Lance + Arrow + Ray, multimodal-first** bet is stronger,
> and where their **curation/annotation product** is ahead of us. Cross-refs into
> [WHATS_LEFT.md](WHATS_LEFT.md) where a finding sharpens a planned bet.

> **The one-sentence framing.** They built a **curation + annotation + evaluation
> product for vision datasets** on a **relational core** (DuckDB/Postgres). We
> built a **multimodal search + exploration engine** on a **Lance-native columnar
> core**. The two overlap on "explore a big dataset of media + embeddings," and
> that overlap is where the lessons are.

> **Where we sit in the landscape.** Three adjacent tools, none of which is us:
> **Rerun** = synchronized *timeline* multimodal viewer, but robotics/CV-leaning
> (point clouds, 3D, `mcap`/`.rrd`); **FiftyOne** = general vision-*dataset*
> explorer on **MongoDB**; **LightlyStudio** = vision *curation + selection* on
> DuckDB + a Rust selection engine. **We are "Rerun for media archives"** —
> aimed at **video · image · audio · text** (not robotics formats), where
> **Lance stores, Ray evolves the columns, DuckDB (lance extension) explores**, and the
> schema is dynamic because the data is alive. The combination of timeline +
> ANN/FTS search + voice + KG + OLAP over an S3-native columnar store is a seat
> none of the three occupy. See [WHATS_LEFT.md §0](WHATS_LEFT.md) for the model.

---

## 1. Architecture at a glance

| Axis | **raudio (us)** | **LightlyStudio (them)** |
|---|---|---|
| System of record | **One Lance dataset**, ~11 tables (`src/raudio/model/schema.py`) | **Relational**: DuckDB (default) / Postgres+pgvector, via SQLModel+SQLAlchemy+Alembic (`db_manager.py`) |
| Vectors | **Lance IVF_PQ** ANN + tuned `nprobes`/refine (`backend/search/constants.py`) | SQL column — pgvector `Vector()` / DuckDB `ARRAY(Float)`, cosine `<=>`, **no ANN index** (`db_vector.py`) |
| Full-text | **Tantivy BM25** native FTS on `text`/`caption` | none of note (filtering is SQL `WHERE`) |
| Lance / Arrow | the **whole store** | **interchange only** — Arrow serializes 2-D embeddings for HTTP; Lance touched in ~12 files, not the SoR |
| Media | Blob V2 (URIs + inline bytes) in-dataset, range-readable | **file-path references** (`file_path_abs`), fsspec; real video-frame model (PTS ms, rotation) |
| Mutable state | none yet (planned Postgres+Redis — [§7](WHATS_LEFT.md)) | first-class: tags, metadata, annotations in SQL |
| Modalities | **text + image + caption + voice/audio + ASR** (Qwen3-VL 2048-d, WeSpeaker 256-d) | **image + video** (MobileCLIP 512-d, Perception Encoder) |
| Search modes | **7** (keyword/semantic/visual/scene/hybrid/fused + rerank) + cross-encoder reranker | text→image cosine similarity + metadata filters |
| Extras we have | knowledge graph, 2-D Atlas, diarization/voiceprints, synchronized A/V playback, MCP server | — |
| Extras they have | **Mundig** sampling engine, few-shot classifier + active learning, annotation, model evaluation, multi-format export, plugin system | — |
| Backend shape | FastAPI + DI factory; `serve-all.sh` choreography | FastAPI, layered **routes → services → resolvers → SQLModel**; `*Base/*Create/*Table/*View` model roles |
| Frontend | SvelteKit, **hand-written zod**, **hardcoded columns** | SvelteKit + shadcn + TanStack Query, **OpenAPI-generated client**, **fully schema-driven** |
| Distribution | separate FE + BE, CLI + HTTP | **`pip install`**, Python SDK, GUI served from the package, single artifact |

---

## 2. What we do better

1. **Vector search that actually scales.** Their similarity is a raw SQL cosine
   (`db_vector.cosine_distance(...)`, `<=>`) with **no HNSW/IVF index** — O(n) per
   query. That's fine for COCO-128 / an ImageNet subset on a laptop (their stated
   target), but it would not give us low-latency ANN over 145k+ chunks. Our
   **Lance IVF_PQ** with tuned `nprobes=20`/`refine_factor=3`
   (`backend/search/constants.py`, the recall fix in [INVESTIGATION.md](INVESTIGATION.md))
   is the right call at our scale. **Don't trade Lance-native ANN for pgvector
   brute force.**

2. **One columnar store for media + vectors + FTS + JSONB.** We keep text,
   metadata, three 2048-d embedding families, word alignments (JSONB), thumbnails,
   frames, and media URIs in **one Lance dataset** with no sidecar files and no
   disk walks ([STORAGE.md](STORAGE.md)). They spread it across a relational schema
   plus file-path media plus Arrow-on-the-wire. Our model is simpler to reason
   about for read-heavy search.

3. **Genuinely multimodal, and audio-first.** They are image+video only — **no
   audio, no ASR, no speaker/voice, no keyword/FTS search, no reranker.** We have
   Qwen3-VL text+image+caption spaces, WeSpeaker voiceprints, diarization, a
   cross-encoder reranker, and **synchronized audio/video playback with word-level
   alignment**. The `lightly` SSL library is likewise image/video-only (15+ SSL
   losses, but no text/audio/CLIP). Our multimodal surface is a real moat.

4. **Knowledge graph + Atlas + MCP.** None of these exist on their side. The KG
   (entities/relationships over chunks, [GRAPH.md](GRAPH.md)), the 2-D Atlas map,
   and the MCP server ([MCP.md](MCP.md)) that exposes search to LLM agents are all
   ours alone.

5. **No single-writer bottleneck baked in.** Their own backend guide flags
   DuckDB's **single-writer model** as a live limitation of their
   `persistent_session()` design — because *their DuckDB is the system of record*.
   Ours isn't: the SoR is **Lance (MVCC/ACID)**, and where we add DuckDB as the
   OLAP layer ([§3](WHATS_LEFT.md)) it's the official **`lance` DuckDB extension**
   querying Lance — DuckDB is a stateless query engine over an MVCC store, so the
   single-writer limit simply doesn't apply. Same engine, opposite outcome,
   because of *where the data lives*.

6. **The architecture is a different *category* — storage/compute separation +
   Arrow end-to-end (the real moat).** This is the deepest difference, and it's
   structural, not a feature any of them can bolt on. **None of the three separate
   storage from compute**: FiftyOne couples both in MongoDB, LightlyStudio in an
   embedded DuckDB/Postgres, Rerun in a closed `.rrd` viewer — all "app + embedded
   DB," single-node-ish, with vectors either external (FiftyOne) or brute-forced
   (Lightly). Ours is a **lakehouse**: **Lance on S3** (open columnar table format,
   MVCC) as the system of record, **KubeRay/vLLM** as elastic compute that *evolves
   the columns* (one model layer, online query-time + offline backfill — see
   [MODULAR_PLAN §2](MODULAR_PLAN.md)), and **DuckDB** as an optional SQL surface
   over the same files. Storage and compute scale independently. And the whole
   stack speaks **one columnar language — Arrow — with no serialization boundary**:
   Lance (Arrow on S3) → Ray (Arrow batches) → **Arrow IPC** on the wire
   (`frontend/src/lib/api.ts` decodes via `tableFromIPC`, not JSON) → Arrow JS in
   the browser → **WebGPU** buffers near-zero-copy (`gpu-scatter`/`gpu-graph` WGSL
   renderers). FiftyOne/Lightly marshal rows → JSON → DOM at every hop; Rerun *is*
   wgpu but is a closed playback silo with no search. **The renderer is replaceable;
   the unbroken Arrow pipeline from object store to GPU is the part that's ours** —
   and it's validated by where data-infra is heading (Lance + Ray, Daft), not a
   lone bet. See [WHATS_LEFT §0](WHATS_LEFT.md) for the full model.

---

## 3. What they do better (and what we should borrow)

1. **Schema-driven frontend — they solved our [§5](WHATS_LEFT.md) cleanly.**
   - The TS API client is **generated from the backend's OpenAPI** via
     `@hey-api/openapi-ts` (`export_schema.py` → `openapi.json` → frontend
     codegen). We hand-maintain zod in `frontend/src/lib/api.ts`.
   - A runtime field-schema endpoint `GET /metadata/info → list[MetadataInfoView]`
     (`api/routes/api/metadata.py:26`, built per-call with name/type/min/max in
     `resolvers/.../get_metadata_info.py:49`) returns **every field**, and
     components render **generically**: `CombinedMetadataDimensionsFilters.svelte`
     loops discovered fields → sliders; `MetadataSegment.svelte` iterates
     `sample.metadata_dict.data` → detail rows. Add a field on the backend and it
     appears with **zero frontend edits**.
   - **Our gap is narrower than it looks — and the fix is precise.** Our search
     endpoints *already* return `list[dict[str, Any]]` (raw `qb.to_list()`), so the
     wire is schema-agnostic; the hardcodes are (a) the `_HIT_COLUMNS` constant the
     query `.select()`s (`backend/search/constants.py`, `search/service.py:122`),
     (b) the startup-pinned Lance handle (`backend/state.py:open_resources`), and
     (c) the frontend zod `HitSchema`. **Borrow:** generate our client from
     FastAPI's OpenAPI (delete the zod); turn `/api/columns` into a live
     `/api/schema` with roles; derive the `SELECT` from the schema; and
     `checkout_latest()` the handle on a §8 event. Then a Ray-added column reaches
     the UI with no codegen and no restart (full mechanism in
     [WHATS_LEFT §5](WHATS_LEFT.md)). The single highest-value steal.

2. **Dynamic fields without losing types — the JSON dual-column pattern.** Rather
   than per-field tables (FiftyOne) or EAV, a `SampleMetadataTable` holds two JSON
   columns: `data` (values) **and** `metadata_schema` (a per-key **type
   registry**, enforced). Arbitrary user fields, still type-safe. A clean model
   for our [§5](WHATS_LEFT.md) "typed core + dynamic extras."

3. **A real selection/curation engine.** Their Rust **Mundig** (`sampling/mundig.py`)
   composes **diversity + similarity + metadata-weighting + class-balancing** in a
   single optimization pass, and computes **typicality** (kNN, k=20) and
   **near-duplicate** scores stored as queryable metadata
   (`metadata/compute_typicality.py`, `compute_similarity.py`). This *is* our
   TODO "uniqueness / near-dup collapse / more-like-this" — shipped. We don't need
   their closed-source Rust wheel: the `lightly` lib's pattern (**scores as typed
   columns + a separate selection pass**) maps directly onto **Lance columns + a
   Ray selection job** ([§1](WHATS_LEFT.md)).

4. **Curation as first-class mutable state.** **Tags** are a primary entity
   (`tag_resolver.py`), sampling results are persisted as tags, and a composable
   `SampleFilter` / `QueryExpr` DSL combines tags + metadata ranges + annotations.
   This is the concrete shape our [§7](WHATS_LEFT.md) Postgres state should take —
   we can lift their tag/filter model almost wholesale.

5. **Product packaging & DX.**
   - **Python SDK with a fluent query builder** — `ls.ImageDataset.load_or_create()`,
     `dataset.query().match(ImageSampleField.tags.contains(...))`, `ls.start_gui()`.
     We have only CLI + HTTP; an SDK would make raudio scriptable/embeddable.
   - **Single-artifact distribution** — the built Svelte app is copied into the
     Python package and served by FastAPI (`api/routes/webapp.py`); `pip install`
     and go. Relevant to the [STUDIO_MERGE.md](STUDIO_MERGE.md) shell and an
     end to our `serve-all.sh` two-process dance.
   - **Layered backend** (routes → services → resolvers) with explicit
     `*Base/*Create/*Table/*View` model roles — tidier than our current split and
     a good target while doing [§5](WHATS_LEFT.md).

6. **Few-shot classifier + active-learning loop.** A Random-Forest-on-embeddings
   classifier (`few_shot_classifier/`) with confidence-based sample suggestion
   (uncertainty sampling). A lightweight, high-value curation feature we could add
   on top of stored embeddings without a GPU.

7. **Off-the-shelf where it's commodity.** Their embedding map reuses Apple's
   **`embedding-atlas`** (WebGL) component rather than a bespoke renderer. Worth
   weighing against our custom Atlas — reusing it could cut maintenance.

8. **Migrations discipline.** Alembic on Postgres (with a custom `VectorType`
   render hook). When we add Postgres state ([§7](WHATS_LEFT.md)), adopt Alembic
   from day one rather than `create_all()`.

---

## 4. What's genuinely out of scope (theirs, not ours)

Their **annotation** stack (COCO/YOLO/segmentation import, SAM autolabeling
plugin), **model-evaluation** framework (OD/classification/segmentation metrics),
and **multi-format export** (labelformat) are *labeling-tool* features. raudio is
a **search/exploration** engine over an existing archive, not a labeling tool —
these are deliberately not our game (though the **plugin/operator system**, scoped
ROOT/COLLECTION/SAMPLE, is an interesting extensibility model if we ever open up
custom enrichment steps).

---

## 5. Net read & priorities

**Validated by the comparison:**
- Our **Lance-native ANN + FTS** core is the right bet at our scale — their SQL
  brute-force vectors are the thing *we* do better, so keep it.
- Our **multimodal/audio/voice/KG** surface is a real differentiator; lean in.
- The **Postgres-for-state** decision ([§7](WHATS_LEFT.md)) is exactly how they
  model tags/metadata — confirms the direction.

**Highest-value things to steal, in order:**
1. **OpenAPI-generated client + runtime field-schema → schema-driven UI**
   ([§5](WHATS_LEFT.md)). Biggest leverage, lowest risk.
2. **Tags + composable filter DSL** as the first Postgres-backed curation state
   ([§7](WHATS_LEFT.md)).
3. **Selection/uniqueness/typicality** as Lance-column scores + a Ray selection
   pass ([§1](WHATS_LEFT.md) + the TODO curation items).
4. **Python SDK + single-artifact serving** for DX
   ([STUDIO_MERGE.md](STUDIO_MERGE.md)).
5. **JSON dual-column (`data` + `metadata_schema`) dynamic-field pattern** and
   **Alembic** when Postgres lands.
