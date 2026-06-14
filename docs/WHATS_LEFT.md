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
  in-flight HTTP requests (`TEXT_CONCURRENCY=32`) and relies on vLLM's
  continuous batching. Orchestration is ad-hoc shell + Makefile targets, with
  separate one-off scripts (`scripts/build_topics.py`, `scripts/kg/*`,
  `scripts/caption_eval.py`).
- **Online** (the read side): the FastAPI backend calls the *same* vLLM servers
  per query for query-vector embedding and reranking.

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
> KubeRay, no GPU, no vLLM. Tantivy BM25 already lives in the dataset, so
> keyword search is a pure-CPU read against Lance. Keep that path first-class
> and degrade gracefully (see §4) rather than making the GPU stack a hard
> dependency of "search the archive."

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
(`doc_id`, `namn`, `referenskod`, `text`, `text_embedding`, `frame_embedding`,
… — see [`src/raudio/model/schema.py`](../src/raudio/model/schema.py)). Adding a
field means touching the Pydantic models, API serializers, and the SvelteKit
components. This is the main thing blocking schema evolution.

**Goal: FiftyOne-like flexibility** — the UI and API adapt to *whatever* columns
the dataset has, instead of a fixed contract:

- A **schema/columns introspection endpoint** (`/api/columns` already exists in
  embryo per [TODO.md](TODO.md)) that reports each field's name, type, and role
  (filterable / displayable / embedding / blob / FTS).
- **Generic, schema-driven rendering** on the frontend — results table columns,
  filter panels, and detail views generated from that introspection rather than
  hardcoded field names (the resizable-columns work is a start).
- **Field "roles"/tags** so the system knows a column is an embedding vs. a
  facet vs. a media blob without naming it explicitly — the FiftyOne sample/
  field-schema idea adapted to Lance.

**Why now:** every other item here (new embedding spaces, voice/speaker columns,
KG-derived fields, configurable chunking in §6) adds columns. Without this, each
one is a frontend+backend edit.

**❓ Open questions:** how far to genericize (full dynamic schema vs. a typed
core + dynamic "extras"); how to keep TypeScript types honest against a dynamic
schema; preserve the curated, hand-tuned views for the default columns.

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

## 7. ❓ Mutable state — SQLite vs. Redis vs. Postgres

The first curation features (tags, saved views — [TODO.md](TODO.md#curation--exploration-roadmap)
item 5) introduce **mutable application state**, which Lance (append/columnar,
not a transactional KV) isn't the right home for. On-demand diarization caching,
speaker naming, and any user/session state need the same.

Options to decide between:

- **SQLite** — zero-ops, fits the single-local-node reality; already floated in
  [TODO.md](TODO.md) as the tags/saved-views store. Likely the default.
- **Redis** — if we need fast ephemeral state, pub/sub, or a cache/queue (ties
  into §8's eventing and §1's job coordination).
- **Postgres** — if state grows relational/multi-user and we want real
  transactions + concurrency.

**Recommendation:** start with **SQLite** for tags/views/names; only reach for
Redis/Postgres when concurrency or eventing (`§8`) actually demands it. Decide
explicitly rather than letting state sprawl across ad-hoc files.

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
justify a pub/sub spine? Polling a "work to do" table (cheap, SQLite-backed) may
beat a full Dapr deployment until services multiply.

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
