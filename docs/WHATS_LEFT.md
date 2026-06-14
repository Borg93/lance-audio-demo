# What's left — the next architecture pass

> A forward-looking design brief for the **bigger structural bets** that aren't
> yet captured in [TODO.md](TODO.md). Where [TODO.md](TODO.md) is the
> value-per-effort backlog of shippable increments, this doc is the **"how the
> system should be shaped next"** discussion — rewrites, infra choices, and
> open questions that change the architecture rather than extend it. Read it
> alongside [GUIDE.md](GUIDE.md) (the current architecture map) and
> [STORAGE.md](STORAGE.md) (the Lance contract).

> Status legend: ✅ done · ⏳ in progress · 📋 planned · 🟡 optional/parked ·
> ❓ open question (decision needed before work starts).

The product is **shipped and working** — none of this is a blocker. These are
deliberate, larger investments, ordered roughly by how much they unlock for the
rest of the system.

---

## 1. 📋 Rewrite inference — offline on Ray Data actors, online on Ray Serve + vLLM

Today inference is **split across two worlds** that don't share a runtime:

- **Offline** (the write side): batch feature passes driven by the CLI
  (`raudio feature text_embedding` / `frame_embedding`, captioning, summaries)
  talk to a **long-running vLLM HTTP server** (`make embed-server` /
  `rerank-server`, pinned to ports 8001/8002 — see
  [`scripts/serve-all.sh`](../scripts/serve-all.sh)). The embedding client
  ([`src/raudio/vllm/embedding.py`](../src/raudio/vllm/embedding.py)) fans out
  in-flight HTTP requests over a `ThreadPoolExecutor` (`TEXT_CONCURRENCY=32`,
  `IMAGE_CONCURRENCY=8`) and relies on vLLM's continuous batching. The
  **resume/checkpoint logic is hand-rolled** in
  [`src/raudio/features/engine.py`](../src/raudio/features/engine.py): a Lance
  `@batch_udf` with `merge_insert` null-fill for scan-derived columns, and a
  two-pass compute→attach with a **JSONL sidecar checkpoint** for blob-derived
  columns (frame embeds/captions). Orchestration is ad-hoc shell + Makefile
  targets, with separate one-off scripts (`scripts/build_topics.py`,
  `scripts/kg/*`, `scripts/caption_eval.py`). **This is exactly the hand-rolled
  concurrency + resume machinery Ray Data's `map_batches` + actor pool replaces.**
- **Online** (the read side): the FastAPI backend calls the *same* vLLM servers
  per query for query-vector embedding and reranking. This is already cleanly
  dependency-injected — `run_search()` (`backend/search/service.py:385-401`)
  lazily gets the embedder via a factory (`backend/clients.py`), and upstream
  failures map to a 503 `ServiceUnavailableError` — so the *online* rewrite is
  mostly swapping what's behind the factory, not surgery on the search code.
- There isn't one model server but **four+**, each its own vLLM process on its
  own port: embed (`:8001`), rerank (`:8002`), caption/Gemma (`:8003`, external —
  we don't even start it), summarize (`:8004`). `serve-all.sh` brings them up
  **sequentially and health-gated** because starting embed+rerank at once trips
  vLLM's GPU memory-profiling race.

**What we want:**

- **Offline → Ray Data with actor-based stateful inference.** Model the feature
  passes as a Ray Data pipeline: `read_lance() → map_batches(EmbedActor, …) →
  write_lance()`. Each `EmbedActor` (or `CaptionActor`, `RerankActor`) holds a
  warm model on a GPU (`num_gpus=1`, `concurrency=N` for N replicas), so Ray
  schedules the GPU pool and handles backpressure / retries / checkpointing
  instead of our hand-rolled concurrency + resume logic. This replaces the
  "start a vLLM server, then run a CLI that HTTP-fans-out at it" two-step with a
  single managed job.
- **Online → Ray Serve in front of vLLM.** Co-locate the serving stack under
  Ray Serve deployments (autoscaling replicas, request batching at the Serve
  layer, a single deployment graph for embed + rerank) rather than two manually
  health-gated server processes.
- **KubeRay is the deployment target.** The actor pool (offline jobs) and the
  Ray Serve deployments (online) run on a **KubeRay** `RayCluster` /
  `RayService` — Kubernetes-native autoscaling, GPU scheduling, and lifecycle —
  rather than a hand-managed local Ray process. This is the key piece that makes
  the actor model worth adopting: KubeRay owns the cluster, we own the pipeline
  and deployment definitions.
- **Stop having "separate scripts."** Fold `scripts/build_topics.py`,
  `scripts/kg/build_kg.py`, the eval scripts, etc. into the same Ray-driven
  pipeline surface so they're **integrated with the rest of the codebase**
  (shared config, shared dataset handles, shared actors) rather than detached
  entrypoints with their own argument parsing and lifecycle.

**Why it's worth it:** one runtime for batch + serving, real GPU scheduling
across the actor pool, native fault tolerance, and an end to the
serve-then-CLI-then-shell choreography in `serve-all.sh` / the Makefile.

> **The Ray/KubeRay/GPU stack must stay optional.** It's the heavy path for the
> full semantic/visual/hybrid experience. A user who only wants **FTS over a
> Lance dataset** must be able to run that with **none of it** — no Ray, no
> KubeRay, no GPU, no vLLM. The groundwork is already there: the backend defers
> all vLLM imports to request time (`backend/clients.py`), and the `fts` /
> `scene_fts` modes never touch the embedder. What's **missing** is graceful
> *cross-mode* fallback — today `hybrid`/`semantic` with `text_embedding` absent
> returns a 400, rather than degrading to FTS (see §4). Keep the no-GPU path
> first-class rather than making the GPU stack a hard dependency of "search the
> archive."

**❓ Open questions**

- Ray cluster shape on a single local GPU node — is the orchestration overhead
  worth it at ~145k rows, or do we adopt Ray purely for the programming model
  (local mode) and grow into a cluster later?
- Does Ray Data read Lance efficiently enough (fragment-parallel scans) to keep
  the GPU saturated, or do we need a custom datasource?
- Migration order: rewrite offline first (lower risk, no live traffic), then
  online once the actor abstractions are proven.

---

## 2. 📋 Lance table maintenance — GC, reindex, manifest compaction

The whole archive is **one Lance dataset** (`transcripts_v2.lance/`, see
[STORAGE.md](STORAGE.md)). Every feature pass appends/merges, every reindex adds
index versions, and old dataset versions accumulate on disk. Today's only nod to
this is a parked "prune old dataset versions (disk)" line in
[TODO.md](TODO.md#parked).

We need a **first-class maintenance story**:

- **Compaction** — `dataset.optimize.compact_files()` to coalesce the many small
  fragments that incremental `merge_insert` / append passes leave behind (the
  diarization and feature backfills are the worst offenders).
- **Garbage collection** — `dataset.cleanup_old_versions(older_than=…)` to
  reclaim the superseded manifests/fragments that stale-version retention pins.
- **Reindex** — rebuild IVF_PQ ANN + Tantivy FTS indices after large appends so
  new rows are actually covered (an unindexed tail silently degrades recall —
  cf. the `nprobes`/recall gotchas in [INVESTIGATION.md](INVESTIGATION.md)).
- **Manifest hygiene** — schedule the above (a `raudio maintain` CLI target +/or
  a periodic job) so it isn't a manual ritual.

**❓ Open questions:** retention window for old versions (we sometimes want to
roll back a bad feature pass); compaction during vs. between feature passes;
whether maintenance becomes one of the Ray jobs from §1.

---

## 3. 📋 Lance namespacing (+ maybe DuckDB-over-Lance)

Right now the "database" is a single dataset directory referenced by path
(`DB=transcripts_v2.lance`). As we add tables (documents, chunks, chunk_frames,
speaker_turns, speakers, topics, …) and variant builds, a **flat directory of
`.lance` folders** stops scaling.

- **Adopt Lance Namespace** — a catalog/namespace layer so tables are addressed
  logically (namespace + table name) instead of by filesystem path, with
  consistent listing/versioning across the table set. This also cleans up the
  shard tables (`speaker_turns_shard{i}.lance`) and merge dance.
- **DuckDB + Lance** — wire DuckDB to query Lance directly (scanner → Arrow) for
  the analytical/faceted side: stats, histograms, group-by-video, and the
  curation panels in [TODO.md](TODO.md#curation--exploration-roadmap). SQL over
  Lance is a better fit for those than bespoke Python scans, and it dovetails
  with §4.

**❓ Open questions:** which namespace backend (directory-based vs. a real
catalog like REST/Glue) for a single-node deploy; how namespacing interacts with
the maintenance jobs in §2.

---

## 4. 📋 Lightweight FTS path — DuckDB-WASM only, no GPU

Search currently assumes the **vLLM GPU stack is up** for the semantic/visual/
hybrid/fused modes. There's no graceful "no-GPU" mode for a lightweight or fully
in-browser deployment.

- **CPU/keyword fallback** — when no GPU is present, serve **FTS-only** search
  (Tantivy BM25 is already in the dataset; semantic legs disabled) so the app is
  still useful without the embedding/rerank servers.
- **DuckDB-WASM in the browser** — for a static/edge deployment, ship the
  columnar text + metadata so the frontend can run keyword search and faceted
  filters **entirely client-side** via DuckDB-WASM, no backend round-trip. Pairs
  with the DuckDB-over-Lance work in §3 and the schema-flexibility work in §5.

**❓ Open questions:** how much of the corpus is shippable to the browser (text +
metadata only, not media/embeddings); whether DuckDB-WASM can read Lance
directly or we export a Parquet/Arrow slice for the client.

---

## 5. 📋 Schema flexibility — stop hardcoding columns (FiftyOne-style)

Both backend and frontend are **hardcoded against the current columns**
(`doc_id`, `namn`, `referenskod`, `bildid`, `extraid`, `text`, `caption`,
`text_embedding`, `frame_embedding`, … — see
[`src/raudio/model/schema.py`](../src/raudio/model/schema.py)). Adding a field
means touching the Pydantic models, API serializers, the zod schema, and the
SvelteKit components. This is the main thing blocking schema evolution.

**Where it's actually hardcoded** (frontend audit — the coupling is concrete,
not abstract):

- **The hit contract is a fixed zod struct** — `HitSchema` in
  [`frontend/src/lib/api.ts`](../frontend/src/lib/api.ts) pins ~17 named fields
  (`doc_id`, `speech_id`, `chunk_id`, `start`, `end`, `text`, `namn`,
  `referenskod`, `bildid`, `extraid`, `caption`, …). Any new column is invisible
  until this struct (and its TS type) is edited.
- **The results table is a hardcoded column list** — `TABLE_COLUMNS` in
  [`frontend/src/lib/components/hit-table.svelte`](../frontend/src/lib/components/hit-table.svelte)
  enumerates every column with a per-field `render`/`sortValue`; `DEFAULT_TABLE_COLS`
  / `MAP_TABLE_COLS` / `WRAP_KEYS` are hand-listed field names. The
  resizable-columns + visibility-toggle work is real but only toggles columns
  from this fixed set.
- **Detail + cards are hand-written field rows** — `player-pane.svelte` adds
  `Caption/File/Time/Name/Reference/Image ID/Extra ID/Language/Segment` one line
  at a time; `hit-card.svelte` / `doc-tile.svelte` hardcode `namn ?? audio_path
  ?? doc_id` as the title.
- **Quick filters are a fixed enum** — `SearchSpec` (`api.ts`) and the rendered
  pills (`active-filters.svelte`) only know `language`, `namn`, `referenskod`,
  `extraid`, `topic`; the language dropdown (`filter-popover.svelte`) hardcodes
  `sv`/`en`. Document browse (`DOC_COLUMNS`) is likewise a fixed 7-field list.

**What's already dynamic** (the seam to build on): `/api/columns` **exists** on
both ends — served by `backend/system/router.py` (introspects the `chunks`
schema, maps types to a `ColumnKind` of `number`/`boolean`/`time`/`text`, and
**skips `alignments_json` + every `*_embedding` column**) and consumed by the
*advanced* filter builder (`filter-popover.svelte` → `listColumns()` in
`api.ts`). The Atlas also reads its colour dimensions
(`language`/`namn`/`topic`/`doc_topic`) dynamically from the Arrow payload. So
the introspection endpoint is there — but it's **scalar-only and role-less**
(name+type, nothing about display/facet/embedding roles), and it doesn't yet
drive the results table, detail views, cards, or quick filters.

**Goal: FiftyOne-like flexibility** — the UI and API adapt to *whatever* columns
the dataset has, instead of a fixed contract:

- **Extend `/api/columns` into a real field-schema** — name, type, nullability,
  plus a **role/tag** (`filterable` / `displayable` / `embedding` / `blob` /
  `fts` / `facet`) and display hints (label, default-visible, detail-visible),
  the FiftyOne sample/field-schema idea adapted to Lance.
- **Schema-driven rendering** — generate `TABLE_COLUMNS`, the detail field rows,
  card title/subtitle, and the quick-filter set from that schema instead of
  hardcoding. Keep `HitSchema` as a typed *core* with a dynamic "extras" bag so
  TS types stay honest for the known fields while new columns flow through.

**Why now:** every other item here (new embedding spaces, voice/speaker columns,
KG-derived fields, configurable chunking in §6) adds columns. Without this, each
one is a frontend+backend edit across the files listed above.

**❓ Open questions:** how far to genericize (full dynamic schema vs. typed core
+ dynamic extras); how to keep TS types honest against a dynamic schema; how to
preserve the curated, hand-tuned default views (title selection, default
columns) on top of a generic renderer.

---

## 6. 📋 Better preprocessing & configurable chunk units

Chunking is currently **fixed upstream** by the ASR pipeline
([PIPELINE.md](PIPELINE.md)) — speech segments → ~30 s `AudioChunk`s, with the
known "one press conference floods the page" redundancy of near-identical
adjacent chunks ([TODO.md](TODO.md#curation--exploration-roadmap)).

- **Configurable chunk units** — make the unit of a "chunk" (and thus a search
  hit) a configurable preprocessing step: fixed-duration windows, sentence/
  semantic boundaries, speaker-turn boundaries (we already have diarization),
  or overlapping windows — chosen per build via config, not hardcoded.
- **Better preprocessing** — text normalization, dedup/uniqueness collapse at
  ingest (cf. the planned `feature uniqueness`), and chunk-merge so retrieval
  units are meaningful rather than mechanical 30 s slices.
- **Config surface** — these knobs live in one place (a build config) and flow
  through the Ray pipeline (§1), not scattered constants.

**❓ Open questions:** re-chunking means re-embedding (expensive) — do we keep
multiple chunk granularities side-by-side, or pick one per build? Interaction
with stored `alignments_json` word timings.

---

## 7. 📋 State management — Postgres for app state, Redis for cache

The first curation features (tags, saved views — [TODO.md](TODO.md#curation--exploration-roadmap)
item 5) introduce **mutable application state**, which Lance (append/columnar,
not a transactional KV) isn't the right home for. On-demand diarization caching,
speaker naming, and any user/session state need the same. Lance stays the
**immutable corpus** store; mutable state lives elsewhere.

The decision (not a menu — these are the two stores):

- **Postgres — the system of record for application/frontend state.** Tags,
  saved views, speaker names, user/session state, and any "work to do" queue
  for §8's eventing. Relational, transactional, multi-user — the durable home
  for everything that isn't the immutable Lance corpus.
- **Redis — the cache layer.** Hot, ephemeral data: query-vector cache (the
  parked "query-vector LRU cache" in [TODO.md](TODO.md#parked) belongs here),
  search-result caching, session/rate-limit data, and a fast scratch space in
  front of Postgres. Can double as the broker/coordination layer for §1's jobs
  and §8's events.

So: **Lance** = immutable corpus + vectors + FTS; **Postgres** = durable mutable
state; **Redis** = cache + ephemeral/coordination. Three stores, three clear
roles — no SQLite, no state sprawl across ad-hoc files.

**❓ Open questions:** does Redis also serve as the §8 event bus, or do we keep
events in Postgres (`LISTEN/NOTIFY` / an outbox table) and use Redis purely as a
cache? How much of the existing on-disk state (e.g. diarization shard tables)
migrates to Postgres vs. stays Lance.

---

## 8. ❓ Eventing / orchestration — Dapr? Data-arrival events vs. polling

The pipeline today is **manually triggered** (Makefile targets, batch backfills,
`serve-all.sh`). There's no event when new data lands or a feature pass
finishes — downstream steps are run by hand or chained in shell.

- **Event-driven ingestion** — emit/consume events ("new alignment JSON
  arrived", "embedding pass done", "video N diarized") so the pipeline reacts
  instead of being poked. Pairs with on-demand diarization
  ([TODO.md](TODO.md#in-flight)) and the Ray jobs in §1.
- **Dapr** — as a possible building-block layer (pub/sub, state store
  abstraction over §7, bindings, service invocation) so we don't hardcode a
  specific broker. Attractive if we go multi-service (Ray Serve + FastAPI + KG +
  workers); heavier than warranted for a single node.

**❓ Open questions:** is this premature at single-node scale (YAGNI), or does
on-demand diarization + the Ray rewrite already create enough moving parts to
justify a pub/sub spine? A "work to do" table in **Postgres** (§7) polled by
workers — or Postgres `LISTEN/NOTIFY` / a **Redis** stream — may beat a full
Dapr deployment until services actually multiply.

---

## 9. 📋 Knowledge graph — needs significant attention

The KG is the **least mature** subsystem. It currently lives in detached scripts
(`scripts/kg/build_kg.py`, `export_chunks.py`, `refine_person_types.py`,
`generic_sv.py`, `adapter.py`) with a backend `graph/` surface and the MCP graph
tools. Recent work was deterministic generic-noun cleanup and entity-type
refinement (git history), but it's a long way from first-class.

Areas that need real investment:

- **Integration, not scripts** — fold KG construction into the unified pipeline
  (§1) so it's incremental and reproducible, not a one-off batch.
- **Entity resolution / disambiguation** — the "which Anders" problem
  ([TODO.md](TODO.md)) deserves a proper coref/linking pass, ideally tied to the
  **voice/speaker identity clusters** ([VOICE.md](VOICE.md)) and named speakers.
- **Schema & storage** — decide how the graph is stored/queried (Lance tables +
  DuckDB joins per §3? a dedicated graph store?) and how it stays in sync as
  chunks/entities change.
- **Quality** — extraction precision/recall, typing, relation quality, and an
  eval harness (the topic/caption evals are a template).
- **Surfacing** — richer graph queries and UI beyond the current examples rail.

**❓ Open questions:** graph storage backend (stay Lance-native vs. adopt a graph
DB); how tightly to couple KG entities to speaker identities; whether the graph
is rebuilt or incrementally maintained under the new eventing model (§8).

---

## 10. 📋 Unify multimodal search — evaluate Jina v5 Omni

Retrieval today runs on **three separate embedding families**, all from
Qwen3-VL-Embedding-2B over vLLM ([EMBEDDINGS.md](EMBEDDINGS.md),
[`src/raudio/vllm/embedding.py`](../src/raudio/vllm/embedding.py)): `text_embedding`
(transcript text), `frame_embedding` (chunk-level image vector), and
`caption_embedding`. The search modes (semantic / visual / scene / hybrid /
fused) exist partly *because* these are distinct vectors in distinct columns
that have to be combined at query time.

**Bet:** evaluate **Jina v5 Omni** (an omni-modal embedding model — text, image,
and audio into **one shared space**) as a path to *unify* search:

- **One embedding space** instead of three columns → semantic, visual, and (new)
  **audio** retrieval become one ANN query, not three legs to fuse. Simplifies
  the fused/hybrid logic in `backend/search/` and the schema in §5.
- **Native audio retrieval** — query the press-conference *audio* directly
  (prosody, non-speech cues), not only its transcript text, which the current
  text+image-only stack can't do.
- **Run it as a §1 actor** — slots straight into the Ray Data embed-actor model;
  it's just a different encoder behind the same `EmbeddingClient` protocol.

This is an **evaluation**, not a commitment: benchmark Jina v5 Omni retrieval
quality (the [`evals/`](../evals) + topic/caption eval harness) against the
current Qwen3-VL three-vector setup before migrating any column.

**❓ Open questions:** does a single unified space match per-modality specialist
vectors on recall, or do we keep both? Embedding dim / storage cost vs. the
current 2048-d columns; re-embedding the whole corpus is a §1 batch job;
licensing/serving (does it run under vLLM or need its own actor runtime?).

---

## 11. 📋 Replace easytranscriber — a Ray-native ASR pipeline

The entire write side starts with **easytranscriber** (+ easyaligner) — the
4-stage VAD → Whisper → wav2vec2 CTC → forced-alignment pipeline wrapped by
`raudio transcribe` ([PIPELINE.md](PIPELINE.md)). It works, but it's a poor fit
going forward: it's an **external dependency we don't control**, its stage-by-
stage, directory-dumping design (`output/vad/`, `output/transcriptions/`,
`output/emissions/`, …) is built for single-process batch runs, and it does **not
map cleanly onto Ray** (the actor/`map_batches` model in §1) — `pyannote` is only
a transitive dep through it ([TODO.md](TODO.md#in-flight)), and the resume story
is filesystem-staging, not a managed pipeline.

**Bet:** rewrite ASR as a **Ray-native pipeline** we own:

- Each stage (VAD, transcription, alignment) becomes a **Ray Data stage /
  actor** holding a warm model on the GPU, streaming Arrow batches between
  stages instead of staging intermediates to per-stage directories.
- The KB models stay the same (KB-Whisper, wav2vec2-voxrex, pyannote VAD — the
  *quality* isn't the problem); what changes is the **orchestration**: managed
  GPU scheduling, backpressure, retries, and checkpointing from Ray, on the same
  KubeRay cluster as everything else in §1.
- `pyannote` becomes a **first-class dependency** rather than transitive.
- Output lands **directly in Lance** (no `output/*/` JSON hop), so transcription
  and the downstream feature passes share one runtime and one dataset.

**❓ Open questions:** how much of easyaligner's forced-alignment we reimplement
vs. vendor; whether to keep an easytranscriber-compatible path during migration;
validating the rewrite produces identical word alignments (regression-test
against the current `alignments_json`).

---

## 12. 📋 Analytics & charting — study the corpus, not just search it

There is **no analytical/visualization surface** for understanding the data in
aggregate. The Atlas map is a 2-D embedding projection and the topic treemap
exists, but there are **no charts** — no distributions, time series, or
faceted breakdowns to actually *study* the corpus. The
[TODO.md](TODO.md#curation--exploration-roadmap) "Stats / histograms" item is the
seed; this is the broader bet.

**Bet:** a real analytics layer to make graphs over the data:

- **Aggregate charts** — distributions and counts over any facet (per `namn`,
  `referenskod`, `language`, `topic`, speaker, time/duration): histograms, time
  series, top-N bars, co-occurrence. The frontend already ships LayerChart
  (used by the topic treemap) — extend it into a charts panel.
- **Powered by DuckDB-over-Lance (§3)** — these are `GROUP BY`/aggregate queries,
  exactly what DuckDB is for; a `/api/stats` (or DuckDB-WASM client-side, §4)
  surface rather than bespoke Python scans.
- **Schema-driven (§5)** — chartable dimensions come from the field-schema
  (anything tagged `facet`), so new columns become chartable automatically.
- **Curation loop** — feeds the faceted filter panel and group-by-video / near-
  dup work in [TODO.md](TODO.md#curation--exploration-roadmap); seeing the
  distribution is how you find the redundancy to collapse.

**❓ Open questions:** server-side `/api/stats` vs. fully client-side DuckDB-WASM
(§4); which chart library surface (lean on the existing LayerChart); precompute
vs. on-the-fly aggregation at 145k+ rows.

---

## How these fit together

These bets reinforce each other, which is why they're one doc:

- **§1 (Ray)** is the new runtime that **§2, §6, §9** all run on, and that
  **§8** would coordinate.
- **§3 (namespacing/DuckDB)** + **§4 (DuckDB-WASM)** + **§5 (schema flexibility)**
  are the "query & present any dataset" stack.
- **§7 (state)** and **§8 (events)** are the connective tissue once there's more
  than one moving service.

Suggested sequencing: **§5 (schema flexibility)** and **§2 (maintenance)** are
the highest leverage / lowest risk and unblock the most other work; **§1 (Ray
rewrite)** is the big foundational change to land before **§6/§9** build on it;
**§3/§4** and the **§7/§8** infra choices follow once the data and runtime shapes
settle.
