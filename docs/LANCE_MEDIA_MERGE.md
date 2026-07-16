# LANCE_MEDIA — merge-preparation goal, target architecture, acceptance criteria

Status: **PROPOSED** (2026-07-16). Grounded in a 7-agent code/docs audit + a 3-critic adversarial verification pass.
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
- KG-building scripts (`scripts/kg/`) and Toponymy topics keep their current out-of-band form.
- `evals/voice_labels_*.json` (human voice labels keyed to transcripts_v2 speaker/turn ids) are **kept intact** and stay valid — the ported voice endpoint SHOULD reproduce the known ranking quality (genuine pair at ranks 1–2, AP ≈ 0.74).

## 3. Current state (condensed audit)

**Pipeline** (`src/raudio/`): Makefile-orchestrated DAG of resumable single-process Typer commands. Concurrency = `ThreadPoolExecutor` HTTP fan-out to 4 vLLM servers (:8001 embed, :8002 rerank, :8003 caption, :8004 summarize), ffmpeg thread pools, and OS-level process sharding with `{table}_shard{i}.lance` staging + `fold_shards`. **Zero `ray` imports.** The good news: `features/engine.py` (upsert_scan_column / upsert_blob_column / ensure_*_index) is deliberately client-free, and `embed_columns.py` binds columns to structural client Protocols — the Ray seam already exists. Blob v2 is already in use (documents.media_blob = external-URI blob; thumbnails/frames = inline blob; all blob tables at `data_storage_version="2.2"`).

**Backend** (`backend/`): read-only FastAPI over one Lance DB. ~32 hardcoded schema couplings (table names in `state.py`, column lists in `search/constants.py`, `filters.py`, `atlas/points.py`, `voice/service.py`, …) but the load-bearing primitives are already generic: `media/blobs.py` (take_blobs + Range streaming), RFC 9457 handlers, `/api/columns` introspection. It **imports `raudio` in 12+ modules** (clients.py, deps.py, search/*, atlas/points.py, voice/encoder.py, mcp/tools.py, …) — not standalone.

**Frontend** (`frontend/src/`): Svelte 5 SPA. Schema leaks are concentrated and enumerable: `api.ts` zod `Hit` mirror, `hitKey = doc|speech|chunk`, `TABLE_COLUMNS`, atlas space/channel names, workflow `scope.ts` raw SQL, graph Cypher presets. Already generic: both WebGPU renderers, atlas math, cross-filter mechanics, `filter-popover` (driven by `/api/columns`), table-prefs, timelines (modulo key names).

**Operational reality (verified 2026-07-16)**: `input/` is empty — all 1,154 `documents.media_blob` external URIs (`file://…/input/sv/*.mp4`) dangle locally; the corpus lives in the HCP `film-raudio` bucket. The vLLM servers are down (:8001 answers 404, :8002–:8004 closed) and the Makefile embed/rerank targets need the known `--with kernels` removal. All 21 test files import `raudio`. Any acceptance check that needs media bytes or embeddings must go through §5 first.

**rask** (context for §9): Polylith-ish monorepo — `packages/` (service-kit, ray-kit, storage, `@rask/ui`, `@rask/api`) / `components/` (services :8801–:8810 behind gateway :8888, SvelteKit MFEs path-routed) / `projects/`. `search_api` is the template for a Lance-backed service. **`components/services/viewer` is a ghost and the name is forbidden.** No Tiltfile: dev = `scripts/dev-micro.sh`, deploy = Helm `chart/`. Ray = KubeRay (prod) / `make ray-up` (local), jobs via ray-kit `JobSubmissionClient`.

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
- **No `raudio`/`rmedia` imports in `backend/`** by end of Phase 2 — current runtime dependencies (topic_tree, alignments parse, vllm clients, voiceprint/diarize) are vendored into backend modules or inverted behind small interfaces. (During Phase 1–2 transition the thin `raudio` shim of P1.1 keeps the old backend runnable.)
- All table/column knowledge flows from the **descriptor**; the only hardcoded names allowed in service code are the descriptor metadata key and reserved names (`__manifest`).
- Serving keeps the proven contracts: `take_blobs` → `BlobFile` chunked `read_range` in `StreamingResponse`, `Accept-Ranges` + single-range parsing, **no `BaseHTTPMiddleware`**, CORS `expose_headers=[Content-Range, Content-Length, Accept-Ranges]`, RFC 9457 problem+json, sync handlers on the threadpool.
- Composition stays a thin factory per group (`create_media_app(...)`, `create_search_app(...)` mounted together today) so each can later be re-hosted under rask `service_kit.make_service_app` without touching routers.

**Config contract** (env vars the reshaped backend reads; S3/chart wiring is deferred to the merge goal):

| var | meaning | dev default |
|---|---|---|
| `MEDIA_DB_ROOT` | namespace `dir` root containing the Lance DBs | repo root (so both `transcripts_v2.lance` and the sample DB are visible) |
| `MEDIA_DESCRIPTOR_DIR` | config-file descriptor overrides | `config/descriptors/` |
| `MEDIA_EMBED_URL` / `MEDIA_RERANK_URL` | query-encoder servers (renamed from `RAUDIO_*`) | `http://127.0.0.1:8001` / `:8002` |

### 4.5 Frontend — descriptor-driven (Phase 3)

The SvelteKit SPA stays in this repo. The generic layer is kept as-is (gpu-scatter, gpu-graph, atlas math, cross-filter, timelines, filter builder, table prefs); every schema-bound spot (`api.ts` Hit mirror, `TABLE_COLUMNS`, hit-card fields, player metaRows, atlas spaces/channels, workflow `scope.ts` SQL, active-filter pills) is rewritten to read the **descriptor** fetched at startup. Row identity = `descriptor.identity` composition (replaces hardcoded `hitKey`). Search modes, filters, media/player URLs, atlas spaces: all descriptor-driven. Corpus-specific copy (guide page, graph Cypher presets) is either descriptor-fed or explicitly quarantined as corpus content.

## 5. Prerequisites, baselines, evidence rules

**Do §5.1–§5.3 before any rename or rewrite.** They exist because the naive checks are otherwise unrunnable (no local media, servers down) or gameable.

### 5.1 Media availability

- Select **5 fixed sample docs** from `transcripts_v2.lance` documents (record their `doc_id`s + source filenames in `scripts/sample_docs.txt`, committed). Pull their `.mp4`s from the HCP **`film-raudio`** bucket back into `input/sv/` using the same hf/S3 tooling used for the upload (always pass `--config`; see the HCP memory note). This un-dangles those 5 docs in transcripts_v2 **and** provides ingest input for parity.
- For the mixed-media smoke (P1.4), generate **synthetic fixtures** instead of downloading: `ffmpeg` `testsrc` mp4 (video), sine-wave wav (audio), any png (image), committed under `tests/fixtures/media/`. Deterministic, no network.

### 5.2 Server + runtime bring-up (per-check requirements)

- vLLM embed :8001 (+ rerank :8002 for hybrid/rerank checks; caption :8003 + summarize :8004 only for the full-stage parity run): start via `scripts/serve-all.sh` / `make stack-up` **with the known fix — drop `--with kernels`** (kernels API change broke the Makefile targets). Servers start sequentially (vLLM memory-profiling race). Both 2B models fit one GPU at 0.45 mem-fraction.
- Ray: local `ray.init()` (no cluster infra). GPU note: vLLM servers own the GPU; Ray actors are CPU-side clients (pyannote/WeSpeaker CPU or small-GPU) — sequence GPU-heavy stages rather than co-residing.
- Old backend for P2.6 goldens: `make backend` (or uvicorn equivalent) against `transcripts_v2.lance`.

### 5.3 Baselines captured BEFORE the P1.1 rename

- **P1.7 baseline**: run the *current* `raudio` CLI over the 5 sample docs into a fresh baseline DB (`baselines/old_pipeline.lance/`), and record the 3 fixed FTS queries' top-10 key sets. If the rename lands first anyway, run the old side from a pinned git worktree of the pre-rename commit.
- **P2.6 goldens**: with the old backend running against transcripts_v2, record responses for a fixed query set (≥3 FTS, ≥3 vector, ≥2 hybrid, ≥2 filtered) into `baselines/search_golden.json`, committed. The Phase-2 parity script replays these against the new search service **on the same DB** and compares top-10 key sets; its output must name the DB path compared.
- **Backend survival**: P1.1 ships a thin `raudio` shim package re-exporting from `rmedia`, so the existing backend (12+ raudio imports) keeps starting until Phase 2 severs the dependency (P2.8).

### 5.4 Which dataset backs which check

| check | dataset |
|---|---|
| P1.4 smoke, P1.8 resume | fresh smoke DB from synthetic fixtures |
| P1.7 parity | `baselines/old_pipeline.lance` vs fresh new-pipeline DB (same 5 docs) |
| P2.4 blob/Range + ffprobe, P3.1 playback | the 5 restored sample docs (transcripts_v2 or the fresh sample DB) |
| P2.5/P2.6 search + parity, P2.7 sub-resources, P3.4 atlas | `transcripts_v2.lance` (descriptor via config file; accepts non-stable `_rowid`; **never compact it**) |
| P2.7 empty-state probe, P3.3 acid test | the P1.4 smoke DB (different keys, no time axis) |

### 5.5 Evidence rules (bind the evaluator)

- **Markers need evidence**: `SMOKE OK` / `PARITY OK` / `RESUME OK` / `SEARCH PARITY OK` count **only** when accompanied in the same output by the evidence lines their criterion names. A bare marker = failure.
- **Grep gates print state**: every "no hits" gate runs as `grep -rn <pattern> <path> && echo 'GATE FAIL' || echo 'GATE OK (0 hits)'` — the evaluator must see `GATE OK`, never empty output. Gates use `--exclude-dir` so the expected hit count is exactly zero (no "only in fixtures" judgment calls).
- New scripts written for checks (parity, smoke, resume) must have their source surfaced in the conversation at least once before their output is trusted.

## 6. Phases and acceptance criteria

MUST = phase-gating; SHOULD = do when reached, may slip. Every criterion's Check output must be shown in the conversation per §5.5.

### Phase 1 — Ray Data pipeline, media-agnostic core

- **P1.0 (MUST)** §5.1 media + §5.3 baselines in place; `scripts/sample_docs.txt`, `tests/fixtures/media/`, `baselines/` committed.
  **Check**: `ls` of the three paths + the sample doc_ids printed.
- **P1.1 (MUST)** Package reshaped: `src/raudio` → `src/rmedia` with `core/` (engine, registry, driver), `modalities/` (av, image), `clients/` (vLLM HTTP); CLI `rmedia`; thin `raudio` shim re-exporting from `rmedia` keeps `backend/` importable; tests ported to `rmedia` with **no reduction in test count**.
  **Check**: `uv run rmedia --help` exits 0; core-purity gate `grep -rn "ffmpeg\|pyannote\|audio_path\|modalities\|subprocess" src/rmedia/core/ && echo 'GATE FAIL' || echo 'GATE OK (0 hits)'` prints GATE OK; a contract test imports `rmedia.core` and asserts `sys.modules` contains no `rmedia.modalities`/`rmedia.clients`; pytest `N passed` line surfaced with N ≥ the pre-rename count.
- **P1.2 (MUST)** Ray Data drivers for every per-row fan-out stage (embed text/frame, caption, summarize, frame extraction, diarize, voiceprint): `lance_ray.read_lance → map_batches(actor pool) → engine upsert / lance_ray.add_columns`, honoring §4.3 blob-flow + writer-topology rules. Old mechanisms gone.
  **Check**: negative gates `grep -rn "ThreadPoolExecutor" src/rmedia/ --exclude-dir=clients && echo FAIL || echo 'GATE OK'` and `grep -rn "_shard" src/rmedia/ && echo FAIL || echo 'GATE OK'`; positive gate `grep -rln "map_batches\|lance_ray" src/rmedia/core/` lists the driver module(s); one real stage executed on the smoke DB with Ray actor-pool progress lines surfaced; `uv run rmedia pipeline plan` prints the stage DAG with actor/GPU config.
- **P1.3 (MUST)** Stage registry declares inputs/outputs/media-gate/client per §4.3; a `tests/test_registry.py` test feeds a mixed-MIME batch through a gated stage and asserts the skipped-row count is logged and nothing raises.
  **Check**: `uv run pytest tests/test_registry.py -v` output surfaced, all passed.
- **P1.4 (MUST)** Media-agnostic ingest: `SourceAdapter` seam (local dir + S3); smoke ingest of the synthetic video+audio+image fixtures creates `documents` rows with correctly sniffed MIMEs and resolving blobs.
  **Check**: smoke script prints `SMOKE OK` **plus** the 3 sniffed MIME strings, the 3 doc_ids, and the byte counts returned by first/last-row 1-byte `read_range` probes.
- **P1.5 (MUST)** Single create path: `create_dataset()` helper is the only table-create site, sets `data_storage_version="2.2"` + `enable_stable_row_ids=True`, declares blob columns via `lance.blob_field`, and stamps the `lance_media.descriptor` schema-metadata key from registry/ingest config. A unit test creates a table via the helper and asserts all three.
  **Check**: `grep -rnE "lance\.write_dataset|lance_ray\.write_lance|\.create_table\(" src/rmedia --include='*.py' | grep -v "core/dataset.py" && echo 'GATE FAIL' || echo 'GATE OK'` (pattern amended 2026-07-16: the naive form matched its own replacements — `overwrite_dataset` contains the substring); the unit test's pytest line; smoke DB tables show the metadata key via `ds.schema.metadata`.
- **P1.6 (MUST)** Distributed indexing/compaction per §4.3 (incl. `CompactionOptions(defer_index_remap=True)`); zero-NULL + rows≥partitions gates preserved as tested logic.
  **Check**: unit test for the gate logic (blocked when NULLs>0 or rows<partitions) passes; one real index built on the parity DB with `ds.list_indices()` output surfaced showing type/params (IVF_PQ cosine 256/64; 16 for voice).
- **P1.7 (MUST)** Parity on the 5 sample docs: `baselines/old_pipeline.lance` vs the new pipeline's fresh DB — identical table set, row counts, doc_ids, key uniqueness; per-embedding-column min cosine ≥ 0.999; the 3 fixed FTS queries return the same top-10 key sets. The script computes both sides from the two DBs (no hardcoded expectations).
  *Amendments (2026-07-16, evidence-based)*: (a) the image path (`frame_embedding`) uses floor 0.995 **plus a stronger check** — every frame JPEG byte-identical by sha1 across DBs (measured vLLM self-jitter on identical bytes 0.9997 back-to-back; cross-session bf16 batching variance exceeds 1e-3 through the vision tower); (b) generative `caption` strings compare as process, not tokens — the caption server is not self-deterministic even at temperature 0 (measured 4/6 exact-match on identical bytes back-to-back), so the check is full population + reported match rate, with `caption_embedding` cosine ≥ 0.999 evaluated over the caption-matching rows.
  **Check**: `uv run python scripts/parity_check.py` prints per-table row-count pairs, per-column min cosine, the 3 key-set comparisons, then `PARITY OK`.
- **P1.8 (MUST)** Resume/idempotency: interrupt the driver mid-stage (or simulate), re-run → completes.
  **Check**: output shows NULL-residual counts before/after resume, duplicate-key query result = 0, then `RESUME OK`.
- **P1.9 (MUST)** Repo health: `uv run pytest` exit 0 (surfaced `N passed`), `uv run ruff check` clean, `ty` clean on `src/rmedia/` and `tests/`.
- **P1.10 (SHOULD)** ASR (easytranscriber JSON ingest) wrapped as a registry stage.

### Phase 2 — Schema-agnostic backend (in place)

- **P2.1 (MUST)** Backend reshaped into `backend/media_api/` + `backend/search_api/` + shared `backend/core/`+`backend/lancekit/`; groups do not import each other; both mounted in one app.
  **Check**: cross-import gates (both directions) print GATE OK; `curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/livez` prints 200.
- **P2.2 (MUST)** Discovery + descriptor endpoints: `GET /api/datasets` (tables, schema JSON, vector cols + dims via `list_size`, blob cols, indexes); `GET /api/datasets/{id}/descriptor` returns merged discovered+declared. The descriptor test **cross-checks against the live dataset**: identity fields exist in the schema, the media blob column is a real `lance.blob.v2` column, every vector binding names an actual FixedSizeList column with matching dim.
  **Check**: `curl` outputs surfaced; the cross-check test's pytest line.
- **P2.3 (MUST)** Corpus-literal gate: `grep -rn "text_embedding\|referenskod\|namn\|chunk_frames\|speaker_turns" backend/media_api backend/search_api --exclude-dir=tests && echo 'GATE FAIL' || echo 'GATE OK (0 hits)'` prints GATE OK. The transcripts descriptor lives in `config/descriptors/`, outside code paths.
- **P2.4 (MUST)** Blob/Range serving for any blob column (dataset id + table + column + row key): 206 with correct `Content-Range` for a mid-file range; 200 without Range; works for external-URI media blobs (restored sample docs) and inline thumbnails/frames.
  **Check**: `curl -r 100-199 -s -D -` output showing `206` + `Content-Range` + 100 bytes; `ffprobe` over the HTTP URL succeeds on a restored sample video.
- **P2.5 (MUST)** Search driven by descriptor: FTS/vector/hybrid take a dataset id; explicit `vector_column_name` always passed; multi-vector tables work; filters compiled from descriptor-declared fields; rerank via descriptor encoder bindings. Requires embed(+rerank) servers up (§5.2).
  **Check**: three `curl` searches (fts, vector, hybrid) against transcripts_v2 return model-shaped hits (output surfaced).
- **P2.6 (MUST)** Semantic parity vs the §5.3 goldens, **same DB both sides** (transcripts_v2): FTS same top-10 key set; vector same top-10 given same encoder; filters equivalent.
  **Check**: parity script prints the query list, both top-10 key sets side by side per query, the DB path compared, then `SEARCH PARITY OK`.
- **P2.7 (MUST)** Sub-resources as capabilities: doc transcript, diarization, atlas points (generic Arrow IPC: descriptor-declared x/y/cluster/rowid + channels), topics, voice, graph — served when declared; `built:false`/404 empty-state otherwise. The capability-less probe target is the P1.4 smoke DB (expected, not scope creep).
  **Check**: `curl` per endpoint on transcripts_v2; one probe on the smoke DB returns the empty-state contract.
- **P2.8 (MUST)** Standalone backend: zero `raudio`/`rmedia` imports in `backend/` (shim no longer needed by backend); backend tests pass without the pipeline package importable.
  **Check**: `grep -rn "import raudio\|from raudio\|import rmedia\|from rmedia" backend/ && echo 'GATE FAIL' || echo 'GATE OK (0 hits)'`; `uv run pytest tests/backend` (exact path per repo layout) surfaced `N passed`.
- **P2.9 (SHOULD)** MCP tools re-pointed at the descriptor-driven services; ported voice endpoint validated against `evals/voice_labels_*.json` (genuine pair ranks 1–2, AP ≈ 0.74). State where the WeSpeaker upload-encoder lives (optional heavy dep of search group vs media group).

### Phase 3 — Schema-agnostic frontend (in place)

- **P3.1 (MUST)** The SPA boots from `GET /api/datasets/{id}/descriptor`: search modes, hit cards, hit table columns, player meta rows, filters — all descriptor-rendered; zod schemas describe the descriptor + envelopes, not corpus columns.
  **Check**: `bun run check` (or `npm run check`) + frontend tests green (output surfaced); **text-form runtime evidence** via playwright (headless WebGPU needs `--enable-unsafe-webgpu`, else headed): DOM dump showing ≥N hit-card nodes for a search on transcripts_v2 and a `<video>` element with populated `src` and `readyState ≥ 2` on a restored sample doc. Screenshots supplementary only.
- **P3.2 (MUST)** Corpus-literal gate: `grep -rn "referenskod\|namn\|text_embedding\|speech_id" frontend/src --include=*.ts --include=*.svelte --exclude-dir=guide && echo 'GATE FAIL' || echo 'GATE OK (0 hits)'` prints GATE OK (the guide route is quarantined corpus content); hardcoded `hitKey` removed — identity composed from `descriptor.identity`.
- **P3.3 (MUST)** **Acid test**: the same build renders the P1.4 smoke DB given only its descriptor. Required structural deltas vs transcripts: different identity key fields, no time axis, different/absent vector columns, different media shape (image + audio docs).
  **Check**: one dev-server run serving both dataset ids; DOM dumps for each surfaced; `git diff --stat frontend/src` printed **empty** between the two renders.
- **P3.4 (MUST)** Atlas descriptor-driven: WebGPU scatter fed by the generic points endpoint; spaces/channels from descriptor.
  **Check**: a debug hook (e.g. `window.__atlasStats = {pointsDrawn}`) printed via playwright showing pointsDrawn > 0 on transcripts_v2.
- **P3.5 (SHOULD)** Timelines, voice UI, topic tree, graph explorer behind descriptor capabilities; workflow `scope.ts` SQL generated from descriptor identity.

### Phase 4 — Merge-readiness (no code moves)

- **P4.1 (MUST)** `docs/RASK_LANDING.md` maps each piece to its rask target per §9, lists remaining merge-time work (service-kit composition, gateway route order, chart/values, dev-micro line, valibot swap, S3/rustfs data move, URI rebase for old-corpus media), and **enumerates any SHOULD features not yet ported** (voice/MCP/topics/graph — each "ported" with its check, or "explicitly deferred").
  **Check**: file content surfaced; the P2.1/P2.8/P2.3/P3.2 gates re-run and printing GATE OK.
- **P4.2 (MUST)** Runbook updated: `docs/REPRODUCE.md` + Makefile targets reflect the Ray pipeline (`rmedia` commands); the `features-all`-equivalent target completes end-to-end on the 5-doc sample.
  **Check**: the make target's completion output (exit 0) surfaced, plus `uv run rmedia pipeline plan` reflecting the documented DAG.

## 7. Invariants (must survive every phase)

1. **Lance 4.0 blob landmine**: `merge_insert` crashes the blob decoder on wide/blob schemas → **on blob-bearing tables**: keep delete+append, derived columns only via `add_columns`-family paths. Non-blob scan tables may use the engine's merge_insert fill-NULL on key columns — that's the sanctioned resume path.
2. `add_columns` transforms return **only** the new columns, positionally aligned; backfills never run concurrently with ingest on the same table (Merge conflicts with ~everything); plain Appends are concurrency-safe.
3. Compaction invalidates indexes and row addresses → `CompactionOptions(defer_index_remap=True)` (lance-ray) or native `defer_index_remap=True`, then `optimize_indices`; `_rowid`-holding features rely on stable row ids on new tables; **never compact transcripts_v2** (no stable row ids).
4. `data_storage_version="2.2"` and `enable_stable_row_ids=True` are **create-time-only**; blob thresholds are baked per column schema.
5. Index invariants: IVF_PQ cosine 256/64 (voice 16), built only at zero NULLs and rows ≥ partitions; FTS Swedish + `with_position=True` + keep stop-words **as descriptor config**; never BTREE-index `extraid`-style columns (planner bug); `chunk_frames.frame_idx` matched in Python, not SQL; multi-vector tables always get explicit `vector_column_name`.
6. Serving: no `BaseHTTPMiddleware`; CORS `expose_headers` for Range; `take_blobs`-based streaming; pure-metadata scans via `ds.to_table(filter=…)`; `_score` only selectable in FTS.
7. Identity: `doc_id = sha1(source_uri)[:16]` (regex-whitelisted in serving); re-ingest = delete doc's rows + append; per-item failures warn+skip, loud failures reserved for correctness.
8. Merge-readiness discipline: nothing named `viewer`; media_api/search_api stay import-independent; backend ends raudio/rmedia-free; composition thin enough for rask `service_kit.make_service_app` later.
9. House rules: plain Pydantic models, Lance schemas stay pyarrow, no god files, WebGPU only, conventional commits on `main` **without AI attribution**; activate `svelte-runes`/`writing-python` before edits (execution note, not a goal-condition clause).
10. **Blob flow**: heavy blobs never transit Ray Data blocks — actors stream via `take_blobs`/`BlobFile` from the dataset directly; bulk byte reads use `read_blobs`.
11. **Writer topology**: actors compute, driver commits; merge_insert/add_columns commits serialized per table driver-side; only plain Appends may be committed from parallel workers.

## 8. `/goal` conditions (paste-ready)

Recommended: run **one phase at a time**. Evidence rules: markers count only with their evidence lines; grep gates must print `GATE OK`.

**Phase 1**
```
Phase 1 of docs/LANCE_MEDIA_MERGE.md (lance-audio repo) is complete: prerequisites and baselines first (5 sample videos restored from the HCP film-raudio bucket into input/sv/ with scripts/sample_docs.txt committed; synthetic ffmpeg/sine/png fixtures in tests/fixtures/media/; old-pipeline baseline DB and FTS top-10 baselines captured BEFORE the rename), then the pipeline rewrite. Every MUST criterion P1.0–P1.9 has had its Check run with passing output shown in the conversation: rmedia CLI works with the raudio shim keeping backend/ importable and tests ported with no reduction in count (pytest N-passed line shown); all grep gates print GATE OK (core purity incl. modalities/subprocess; no ThreadPoolExecutor outside clients; no _shard staging; positive gate lists map_batches/lance_ray driver modules); one real Ray stage runs on the smoke DB with actor-pool progress shown; registry mixed-MIME skip test passes; smoke prints SMOKE OK with 3 sniffed mimes + doc_ids + read_range byte counts; the single create_dataset() helper gate passes with its flags-and-descriptor-metadata unit test; index-gate unit test passes and one real index shows in ds.list_indices() with IVF_PQ cosine 256/64 params; parity script (source shown) prints per-table row counts, min cosines, FTS key-set comparisons, then PARITY OK; resume shows NULL-residuals before/after and 0 duplicates then RESUME OK; ruff clean and ty clean on src/rmedia and tests. Bare markers without their evidence lines do not count. Constraints: only the lance-audio repo changes; rask and lance-ns untouched; commits on main, conventional style, no AI attribution. If blocked on the same error 3 consecutive turns, or after 40 turns, stop and summarize instead.
```

**Phase 2**
```
Phase 2 of docs/LANCE_MEDIA_MERGE.md (lance-audio repo) is complete: the backend is schema-agnostic and pre-split into backend/media_api + backend/search_api with shared core/lancekit, and every MUST criterion P2.1–P2.8 has had its Check shown in the conversation: cross-import gates both directions print GATE OK and curl -w prints 200 for /livez; GET /api/datasets and the descriptor endpoint outputs are surfaced with the descriptor cross-check test (identity fields exist, blob column is real blob-v2, vector bindings match FixedSizeList dims) passing; the corpus-literal gate over backend/media_api + backend/search_api prints GATE OK with the transcripts descriptor living in config/descriptors/; blob Range serving shows a curl -D - response with 206 and correct Content-Range plus ffprobe success on a restored sample video; fts/vector/hybrid curl searches with embed and rerank servers running return model-shaped hits; the search parity script prints each query with both top-10 key sets side by side against the pre-recorded baselines on the same transcripts_v2 DB, then SEARCH PARITY OK; capability sub-resources respond on transcripts_v2 and the smoke DB probe returns the built:false empty-state; the no-raudio/rmedia-imports gate over backend/ prints GATE OK and backend tests pass with the pytest N-passed line shown. Bare markers without their evidence lines do not count. Constraints: only the lance-audio repo changes; commits on main, no AI attribution. If blocked on the same error 3 consecutive turns, or after 40 turns, stop and summarize instead.
```

**Phase 3**
```
Phase 3 of docs/LANCE_MEDIA_MERGE.md (lance-audio repo) is complete: the frontend renders search, hit cards, hit table, player, filters, and atlas purely from the dataset descriptor, and every MUST criterion P3.1–P3.4 has had its Check shown in the conversation: frontend check and tests green (output surfaced); playwright text evidence shows hit-card DOM nodes and a video element with populated src and readyState >= 2 on a restored sample doc; the corpus-literal gate over frontend/src (guide route excluded as quarantined content) prints GATE OK and hardcoded hitKey is replaced by descriptor.identity composition; the acid test passes — one dev-server run serves both transcripts_v2 and the structurally different smoke DB (different identity keys, no time axis, different media shape) with DOM dumps for each and an empty git diff --stat of frontend/src between the two renders; the atlas WebGPU view reports pointsDrawn > 0 via its debug hook under playwright. Constraints: Svelte 5 runes + bits-ui/shadcn + Tailwind + WebGPU only; only the lance-audio repo changes; commits on main, no AI attribution. If blocked on the same error 3 consecutive turns, or after 40 turns, stop and summarize instead.
```

**Phase 4**
```
Phase 4 of docs/LANCE_MEDIA_MERGE.md (lance-audio repo) is complete: docs/RASK_LANDING.md exists with its content surfaced, mapping media_api/search_api/frontend/pipeline to their rask targets, listing all remaining merge-time work, and enumerating every SHOULD feature as ported-with-check or explicitly deferred; the P2.1, P2.3, P2.8, and P3.2 grep gates re-run and print GATE OK; docs/REPRODUCE.md and the Makefile reflect the rmedia Ray pipeline; and the features-all-equivalent make target completes on the 5-doc sample with its exit-0 output surfaced plus rmedia pipeline plan matching the documented DAG. Constraints: no code moves into rask; only the lance-audio repo changes; commits on main, no AI attribution. If blocked on the same error 3 consecutive turns, or after 20 turns, stop and summarize instead.
```

**Umbrella (whole preparation — prefer the four sequential goals above)**
```
The lance-media merge PREPARATION defined in docs/LANCE_MEDIA_MERGE.md is complete through Phase 4 — prerequisites/baselines first, then: pipeline on Ray Data actor pools via lance-ray with a media-agnostic rmedia package; schema-agnostic descriptor-driven backend pre-split into non-cross-importing media_api/search_api with zero raudio imports; descriptor-driven frontend passing the two-dataset acid test with an empty frontend diff between renders; and merge-readiness docs — WITHOUT performing the rask merge itself. Every MUST criterion P1.0–P1.9, P2.1–P2.8, P3.1–P3.4, P4.1–P4.2 has had its stated Check run with passing output surfaced in the conversation; all grep gates print GATE OK; the markers SMOKE OK, PARITY OK, RESUME OK, SEARCH PARITY OK appear together with their required evidence lines (bare markers do not count); pytest/ruff/ty and frontend check outputs are shown green in each phase. Constraints: ALL changes confined to the lance-audio repo — rask and lance-ns are read-only; commits on main, conventional style, no AI attribution. If blocked on the same error 3 consecutive turns, or after 160 turns total, stop and summarize instead.
```

## 9. Future rask landing map (context only — a separate follow-up goal)

| piece (after this goal) | rask target | notes |
|---|---|---|
| `backend/media_api/` | brick `components/services/media_api`, :8805, gateway `{prefix}/media` | compose via `service_kit.make_service_app`; full add-a-brick checklist (workspace members, ruff isort, pytest testpaths, `projects/<name>/`, `.docker/<name>.dockerfile`, Makefile `COMPOSE_IMAGES`, chart `services:` entry, gateway `_routes()` + `RASK_<NAME>_URL`, `scripts/dev-micro.sh`); **never named `viewer`** |
| `backend/search_api/` | brick `components/services/media_search`, :8806, prefix registered **before** `media` and the core catch-all | lancedb deps in the brick's own pyproject — service-kit stays dependency-light; decide where the WeSpeaker upload-encoder lives (optional heavy dep vs sidecar) |
| frontend | the dummy `studio` MFE (`/default/studio`, port 5177) or a new `media` MFE | `@rask/ui` shell + `@rask/api` conventions; zod → valibot swap; Bun; path-routed via `microfrontends.json` + ingress |
| `src/rmedia` pipeline | a rask brick (`components/cli/` or `packages/`) | submitted via ray-kit `JobSubmissionClient`; KubeRay in prod |
| Lance DB | rustfs/S3 bucket + `dir`→`rest` namespace flip; rebase old-corpus blob URIs (`base_store_params`) or re-ingest | storage_options via the settings helper pattern |
| gotchas | `RASK_API_PREFIX` is `/api` in chart+dev-micro but `/api/v1` in code defaults; Dapr eats trailing slashes (keep SlashToleranceMiddleware) | verified 2026-07-16 |
