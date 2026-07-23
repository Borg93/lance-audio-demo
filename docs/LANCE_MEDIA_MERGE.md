# LANCE_MEDIA — merge-preparation goal, target architecture, acceptance criteria

Status: **COMPLETE** (all 4 prep phases shipped + pushed by 2026-07-21; detail in git history). Kept for §1–§2 (scope), §4 (target architecture, code-cited), §7 (invariants), and the §9 rask landing map (superseded in detail by [RASK_LANDING.md](RASK_LANDING.md) / [RASK_COMPARE.md](RASK_COMPARE.md)). Originally grounded in a 7-agent code/docs audit + a 3-critic adversarial verification pass.
Repos: `~/Desktop/lance-audio` (**the only repo this goal changes**), `~/Desktop/rask` (future destination — read-only context), `~/Desktop/lance-ns` (pattern source + Lance docs — read-only).

---

## 1. Goal (one sentence)

**Prepare lance-audio for its merge into rask — without performing the merge**: rewrite the batch pipeline as a media-agnostic **Ray Data + Ray-actor pipeline with vLLM** (via `lance-ray`), make the **backend schema-agnostic** (runtime table/column discovery + a per-dataset *descriptor*), and make the **frontend schema-agnostic** (descriptor-driven rendering) — all inside the lance-audio repo, structured so the later lift into rask (viewer-role + search-role services + one MFE) is a mechanical move, not a re-architecture.

"lance-audio" the concept becomes **lance-media**: blobs are bytes + MIME; audio, mp4, and images are modalities gated at stage level, never baked into the core.

## 2. Non-goals

- **The rask merge itself.** No rask bricks, gateway routes, chart entries, or MFE work in this goal — §9 records the landing map as context for a *follow-up* goal. rask and lance-ns are read-only throughout.
- No new query engine. Online search serving stays on **native lancedb/lance APIs** — `lance-ray` is for offline work (IO, backfill, distributed indexing, compaction) only.
- No full-corpus (1,573-video) reprocessing. Parity is proven on a fixed sample; scale runs are a follow-up.
- **Old-corpus full-video playback stays degraded**: `transcripts_v2.lance` media URIs point at the deleted local `input/` tree (bytes now live in the HCP `film-raudio` bucket). Only the restored sample docs (§5.1) play. Rebasing the old corpus's blob URIs to remote storage (`base_store_params`) or re-ingesting is a follow-up, not this goal.
- No adoption of lance-ns infra (FGA/OIDC/outbox/Dapr-pubsub). We copy its *patterns* (blob seam, service layering, Range/ETag serving), not its stack.
- No UI framework changes: Svelte 5 runes + bits-ui/shadcn-svelte + Tailwind, WebGPU renderers only. zod stays (valibot is a rask-merge-time concern). Frontend type-checking moves to **TypeScript 7 (the native `tsgo` compiler)** in Phase 3 wherever the toolchain supports it (`tsc` → `tsgo`; svelte-check may lag — fall back per-tool, not per-project; user-requested 2026-07-16).
- KG-building scripts (`runners/kg/`) and Toponymy topics keep their current out-of-band form.
- `evals/voice_labels_*.json` (human voice labels keyed to transcripts_v2 speaker/turn ids) are **kept intact** and stay valid — the ported voice endpoint SHOULD reproduce the known ranking quality (genuine pair at ranks 1–2, AP ≈ 0.74).

## 3. Current state at proposal time (2026-07-16) — historical

*(Condensed away 2026-07-23: the audit described the pre-rewrite tree; every gap it
listed was closed by the four phases. See git history at this file's introduction
for the original inventory.)*

## 4. Target architecture

### 4.1 Storage — Lance, blob v2, namespace

- One Lance database per corpus, addressed through **Lance Namespace** (`dir` implementation locally — `lancedb.connect(namespace_client_impl="dir", namespace_client_properties={"root": ...})` or `lance_namespace.connect("dir", {...})`; `rest` later without code changes since both hide behind `LanceNamespace`).
- All media bytes live in **blob v2 columns** (`lance.blob_field` / `lance.blob_array`; external `Blob.from_uri` for source media referenced in place, inline for thumbnails/frames). Every table-create sets `data_storage_version="2.2"`; **new** tables also set `enable_stable_row_ids=True` (both flags are create-time-only; existing tables can't be upgraded in place — the fresh sample DB, not `transcripts_v2.lance`, carries stable row ids).
- Blob detection is a single seam copied from lance-ns: `is_blob_field` via the `lance.blob.v2` Arrow extension name (`services/common/blobs.py` pattern).
- Bulk reads use `ds.read_blobs` (scheduler-batched); serving uses `ds.take_blobs` → lazy seekable `BlobFile` (`read_range` in chunks). Never thread-pool `take_blobs` + `readall()` for bulk (documented anti-pattern).

### 4.2 The dataset descriptor — the load-bearing contract

Schema-agnosticism ≠ everything inferred. Split knowledge in two:

**Discovered at runtime** (mechanical, from namespace/dataset introspection):
tables; column names/types; vector columns = FixedSizeList<float> (dim = pyarrow `list_size`; the namespace REST `JsonArrowSchema` spells it `length`); blob columns = `lance.blob.v2` extension; existing indexes + their columns/types (`ds.list_indices()` / `ListTableIndices`); row counts; dataset version.

**Declared per dataset** (semantic roles), stored as table **schema metadata** under the reserved key `lance_media.descriptor`, with a **config-file override** in `config/descriptors/` for datasets we can't rewrite (transcripts_v2 uses the config file; new tables get the metadata stamped at create time — see P1.5):
- `identity`: key fields + composed row-key (generalizes `hitKey`)
- `document`: which table is doc-level, its media blob column + mime column, thumbnail column, row-table→doc join key
- `time`: start/end fields on the doc clock (optional — image corpora have none)
- `display`: title fallback chain, body-text field, caption field, metadata fields with labels
- `search`: FTS column(s) + FTS config (language, with_position, stop-words), vector column ↔ query-encoder binding (model id + endpoint), default mode set
- `atlas`: spaces (projection column prefix ↔ source embedding column), categorical channels
- `capabilities`: optional sub-resources (word alignments, speaker turns, voiceprints, topics, kg tables) — probe-gated exactly like today's frontend

The backend merges discovered + declared and serves it; the frontend renders **only** from the descriptor. The current corpus becomes *one descriptor instance* (a config file, **outside `src/` and `backend/` code paths**), not code.

### 4.3 Pipeline — Ray Data + actors + vLLM (Phase 1)

- Driver pattern per stage: `lance_ray.read_lance(uri|namespace+table_id, columns=needed, filter=residual-NULL)` → `ray.data.Dataset.map_batches(StageActor, concurrency=(min,max), num_cpus/num_gpus=…, batch_size=…)` → write back via the existing engine seams (`upsert_scan_column`'s merge_insert fill-NULL path for scan columns, `_rowid`-keyed attach for blob columns) or `lance_ray.add_columns(transform=…)` where a whole-column distributed backfill fits (transform returns **only** the new columns, positionally aligned).
- **Blob flow rule (invariant §7.10)**: heavy-blob stages (frame extraction, diarize, voiceprint, media transcode) read **only keys/`_rowid` + mime** through Ray Data; each actor opens the dataset itself and streams bytes lazily via `ds.take_blobs` → `BlobFile` (with `base_store_params` for external URIs), piping into ffmpeg. Only small inline blobs (thumbnails, frame JPEGs) may flow through `map_batches` blocks. Never materialize full videos into the Ray object store.
- **Writer topology (invariant §7.11)**: actors compute, the **driver commits**. All `merge_insert`/`add_columns` commits happen driver-side, serialized per table. Only plain Appends (`lance.fragment.write_fragments` per worker + one `LanceOperation.Append` commit, or `write_lance(mode="append")`) may involve parallel workers — Appends never conflict; Merge conflicts with nearly everything (this retires both the `_shard{i}` staging tables and their failure mode).
- **Actors are clients, vLLM stays a server**: vLLM cannot be a project dependency (torch cu128 pin), so GPU inference remains in the 4 vLLM server processes (Ray Serve on KubeRay is a rask-merge-time option); Ray actors hold warm HTTP clients / warm local models (pyannote, WeSpeaker) and replace both the ThreadPoolExecutor fan-out and the OS-level shard processes.
- Stage registry (evolves `FEATURES`): each stage declares name, source table, input columns (incl. blob column), output columns (name → Arrow type), a **media gate** (MIME predicate; `None` = all rows), and a client requirement. The registry is the media-agnosticism boundary: core engine + driver import no modality code; ffmpeg/pyannote/ASR live in `modalities/` behind gates.
- Ingest generalizes to the lance-ns seam: `SourceAdapter → SourceObject{uri, bytes|reference}` → documents row (`doc_id = sha1(source_uri)[:16]`, mime sniffed not extension-assumed, media as external blob or ingested bytes via `write_lance(..., external_blob_mode="reference"|"ingest")`).
- Distributed index builds via `lance_ray.create_index` (IVF_PQ) / `create_scalar_index` (FTS/BTREE). Compaction: `lance_ray.compact_files(compaction_options=lance.optimize.CompactionOptions(defer_index_remap=True), …)` — the lance-ray function has **no** direct `defer_index_remap` kwarg — or native `ds.optimize.compact_files(defer_index_remap=True)`; always followed by `optimize_indices`.
- Global fits stay driver-side single tasks (EVoC atlas projection, speaker clustering) — they mathematically cannot be row-parallel.
- Runs on a local `ray.init()` cluster in this goal; KubeRay submission is rask-merge-time.

### 4.4 Backend — schema-agnostic, pre-split for the future services (Phase 2)

Still one FastAPI process in this repo, but internally reshaped into **two cleanly separable router groups** matching the future rask split (viewer-role + search-role; neither named `viewer`):

- `backend/media_api/` — datasets + descriptor endpoints, blob/Range streaming, thumbnails/frames, doc transcript, diarization, atlas points, topics/graph/voice sub-resources (capability-gated).
- `backend/search_api/` — FTS/vector/hybrid search driven by the descriptor; query-encoder clients (vLLM embed/rerank).

Rules that make the later lift mechanical:
- Each group has its own routers + state deps and **no imports from the other group**; shared primitives live in `backend/core/` + a small shared `backend/lancekit/` module (blob seam, descriptor model, introspection).
- **No `ratch`/`ratch` imports in `backend/`** by end of Phase 2 — current runtime dependencies (topic_tree, alignments parse, vllm clients, voiceprint/diarize) are vendored into backend modules or inverted behind small interfaces. (During Phase 1–2 transition the thin `ratch` shim of P1.1 keeps the old backend runnable.)
- All table/column knowledge flows from the **descriptor**; the only hardcoded names allowed in service code are the descriptor metadata key and reserved names (`__manifest`).
- Serving keeps the proven contracts: `take_blobs` → `BlobFile` chunked `read_range` in `StreamingResponse`, `Accept-Ranges` + single-range parsing, **no `BaseHTTPMiddleware`**, CORS `expose_headers=[Content-Range, Content-Length, Accept-Ranges]`, RFC 9457 problem+json, sync handlers on the threadpool.
- Composition stays a thin factory per group (`create_media_app(...)`, `create_search_app(...)` mounted together today) so each can later be re-hosted under rask `service_kit.make_service_app` without touching routers.

**Config contract** (env vars the reshaped backend reads; S3/chart wiring is deferred to the merge goal):

| var | meaning | dev default |
|---|---|---|
| `MEDIA_DB_ROOT` | namespace `dir` root containing the Lance DBs | repo root (so both `transcripts_v2.lance` and the sample DB are visible) |
| `MEDIA_DESCRIPTOR_DIR` | config-file descriptor overrides | `config/descriptors/` |
| `MEDIA_EMBED_URL` / `MEDIA_RERANK_URL` | query-encoder servers (renamed from `MEDIA_*`) | `http://127.0.0.1:8001` / `:8002` |

### 4.5 Frontend — descriptor-driven (Phase 3)

The SvelteKit SPA stays in this repo. The generic layer is kept as-is (gpu-scatter, gpu-graph, atlas math, cross-filter, timelines, filter builder, table prefs); every schema-bound spot (`api.ts` Hit mirror, `TABLE_COLUMNS`, hit-card fields, player metaRows, atlas spaces/channels, workflow `scope.ts` SQL, active-filter pills) is rewritten to read the **descriptor** fetched at startup. Row identity = `descriptor.identity` composition (replaces hardcoded `hitKey`). Search modes, filters, media/player URLs, atlas spaces: all descriptor-driven. Corpus-specific copy (guide page, graph Cypher presets) is either descriptor-fed or explicitly quarantined as corpus content.

## 5–6. Phases, baselines, acceptance criteria — executed

*(Condensed away 2026-07-23: all four phases ran to their acceptance criteria and
pushed 2026-07 — Ray Data+actors via lance-ray, media-agnostic ratch, schema-agnostic
descriptor backend/frontend, and the split services. The full phase plans + evidence
rules live in git history; the proofs live in `docs/proofs/` and the test suite.)*

## 7. Invariants (must survive every phase)

1. **Lance 4.0 blob landmine**: `merge_insert` crashes the blob decoder on wide/blob schemas → **on blob-bearing tables**: keep delete+append, derived columns only via `add_columns`-family paths. Non-blob scan tables may use the engine's merge_insert fill-NULL on key columns — that's the sanctioned resume path.
2. `add_columns` transforms return **only** the new columns, positionally aligned; backfills never run concurrently with ingest on the same table (Merge conflicts with ~everything); plain Appends are concurrency-safe.
3. Compaction invalidates indexes and row addresses → `CompactionOptions(defer_index_remap=True)` (lance-ray) or native `defer_index_remap=True`, then `optimize_indices`; `_rowid`-holding features rely on stable row ids on new tables; **never compact transcripts_v2** (no stable row ids).
4. `data_storage_version="2.2"` and `enable_stable_row_ids=True` are **create-time-only**; blob thresholds are baked per column schema.
5. Index invariants: IVF_PQ cosine 256/64 (voice 16), built only at zero NULLs and rows ≥ partitions; FTS Swedish + `with_position=True` + keep stop-words **as descriptor config**; never BTREE-index `extraid`-style columns (planner bug); `chunk_frames.frame_idx` matched in Python, not SQL; multi-vector tables always get explicit `vector_column_name`.
6. Serving: no `BaseHTTPMiddleware`; CORS `expose_headers` for Range; `take_blobs`-based streaming; pure-metadata scans via `ds.to_table(filter=…)`; `_score` only selectable in FTS.
7. Identity: `doc_id = sha1(source_uri)[:16]` (regex-whitelisted in serving); re-ingest = delete doc's rows + append; per-item failures warn+skip, loud failures reserved for correctness.
8. Merge-readiness discipline: nothing named `viewer`; media_api/search_api stay import-independent; backend ends ratch/ratch-free; composition thin enough for rask `service_kit.make_service_app` later.
9. House rules: plain Pydantic models, Lance schemas stay pyarrow, no god files, WebGPU only, conventional commits on `main` **without AI attribution**; activate `svelte-runes`/`writing-python` before edits (execution note, not a goal-condition clause).
10. **Blob flow**: heavy blobs never transit Ray Data blocks — actors stream via `take_blobs`/`BlobFile` from the dataset directly; bulk byte reads use `read_blobs`.
11. **Writer topology**: actors compute, driver commits; merge_insert/add_columns commits serialized per table driver-side; only plain Appends may be committed from parallel workers.

## 9. Future rask landing map (context only — a separate follow-up goal)

| piece (after this goal) | rask target | notes |
|---|---|---|
| `backend/media_api/` | brick `components/services/media_api`, :8805, gateway `{prefix}/media` | compose via `service_kit.make_service_app`; full add-a-brick checklist (workspace members, ruff isort, pytest testpaths, `projects/<name>/`, `.docker/<name>.dockerfile`, Makefile `COMPOSE_IMAGES`, chart `services:` entry, gateway `_routes()` + `RASK_<NAME>_URL`, `scripts/dev-micro.sh`); **never named `viewer`** |
| `backend/search_api/` | brick `components/services/media_search`, :8806, prefix registered **before** `media` and the core catch-all | lancedb deps in the brick's own pyproject — service-kit stays dependency-light; decide where the WeSpeaker upload-encoder lives (optional heavy dep vs sidecar) |
| frontend | the dummy `studio` MFE (`/default/studio`, port 5177) or a new `media` MFE | `@rask/ui` shell + `@rask/api` conventions; zod → valibot swap; Bun; path-routed via `microfrontends.json` + ingress |
| `src/ratch` pipeline | a rask brick (`components/cli/` or `packages/`) | submitted via ray-kit `JobSubmissionClient`; KubeRay in prod |
| Lance DB | rustfs/S3 bucket + `dir`→`rest` namespace flip; rebase old-corpus blob URIs (`base_store_params`) or re-ingest | storage_options via the settings helper pattern |
| gotchas | `RASK_API_PREFIX` is `/api` in chart+dev-micro but `/api/v1` in code defaults; Dapr eats trailing slashes (keep SlashToleranceMiddleware) | verified 2026-07-16 |
