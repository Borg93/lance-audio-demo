# RASK_LANDING — how lance-media lands in rask (S3 · Lance · Ray)

> **Scope.** This repo (lance-audio → *lance-media*) is **merge-ready but not
> merged.** Nothing here moves code into rask. This doc is the map a later goal
> follows to *add* the descriptor-driven media services to the
> [rask](../../rask) monorepo — which runs on **S3-backed Lance addressed
> through a namespace, with Ray for distributed compute.** Every claim is
> grounded in the real rask / lance-ns source (file:line cited); where the
> earlier §9 sketch in [LANCE_MEDIA_MERGE.md](LANCE_MEDIA_MERGE.md) was
> incomplete or wrong, this doc corrects it (§8).
>
> Companion runbooks: [REPRODUCE.md](REPRODUCE.md) (build the corpus + the Ray
> pipeline locally), [LANCE_MEDIA_MERGE.md](LANCE_MEDIA_MERGE.md) (the
> preparation goal + acceptance criteria this doc closes as Phase 4).

---

## 0. Status at a glance

| Piece | State | Landing target | Merge-time work |
|---|---|---|---|
| `backend/media_api/` (viewer role) | ✅ ready to lift | rask brick `components/services/media_api`, `:8805`, gateway `{prefix}/media` | wrap `create_media_app`'s routers in `make_service_app`; **add S3 `storage_options`** |
| `backend/search_api/` (search role) | ✅ ready to lift | rask brick `components/services/media_search`, `:8806`, prefix **before** `/media` + core catch-all | ditto; lancedb + encoder deps in the brick's own `pyproject` |
| `frontend/` (SvelteKit SPA) | ✅ ready to lift | a `media` MFE (`/default/media`) | adapter-static→**adapter-bun**, `@rask/ui` shell, **zod→valibot**, API target `:8000`→gateway `:8888` |
| `src/rmedia/` (Ray pipeline) | ✅ ready to lift | a rask brick job submitted to **KubeRay** via the Ray Jobs REST API | local `ray.init()` → `submit_stage_job`-style submission; entrypoint baked into the ray-lance image |
| Lance data (`transcripts_v2.lance` + smoke) | ⚠️ needs a move | S3/rustfs bucket, `dir`→`rest` namespace | copy managed blobs; **re-ingest/rebase the external `media_blob` URIs** |
| SHOULD features (voice · topics · graph) | ✅ ported + capability-gated | ride along with the two bricks | none (checks in §6) |
| MCP | ⛔ deferred by decision (P2.9) | — | re-scope in a later goal |

The backend is already the hard 80%: **module-level `app` + lifespan**, two
**import-independent** router groups meeting only at the composition root, the
**target env-var contract** (`MEDIA_DB_ROOT` / `MEDIA_DESCRIPTOR_DIR` /
`MEDIA_EMBED_URL` / `MEDIA_RERANK_URL`, `core/config.py:32-47`), **zero
`raudio`/`rmedia` imports**, and per-group thin factories (`create_media_app`
`media_api/__init__.py:44`; `create_search_app` `search_api/app.py:24`) that
re-host under `service_kit.make_service_app` without touching a router. The
**object-storage** gap that used to sit here is now closed on the core read path:
the backend serves datasets/descriptor/search from S3-backed Lance (MinIO +
RustFS, verified) behind env-gated `storage_options` — see §4.2.

---

## 1. Landing map (corrected)

| piece | rask target | port | gateway prefix | notes |
|---|---|---|---|---|
| `backend/media_api/` | `components/services/media_api` (+ `projects/media-api/`) | 8805 | `{prefix}/media` | **never named `viewer`**; datasets/descriptor, blob+Range, thumbnails/frames, transcript, diarization, atlas points, topics/graph/voice sub-resources |
| `backend/search_api/` | `components/services/media_search` (+ `projects/media-search/`) | 8806 | `{prefix}/media/search` (or sibling `{prefix}/media-search`) — registered **before** `/media` and the core catch-all | FTS/vector/hybrid; query-encoder clients; **lancedb + WeSpeaker deps in the brick's own `pyproject`**, never in service-kit |
| `frontend/` | `components/frontends/media` MFE, base `/default/media`, dev port 5180 | — | ingress path-routes `/default/media` | `@rask/ui` shell + `@rask/api`; WebGPU/arrow/xyflow port as-is |
| `src/rmedia/` pipeline | a rask brick job (`scripts/rmedia_stage_job.py` baked into a ray image) | — | — | submitted via Ray Jobs REST (`submit_stage_job` pattern); KubeRay in prod |
| Lance DB | S3/rustfs bucket + `dir`→`rest` namespace flip | — | — | `storage_options` via the catalog Settings helper; rebase/re-ingest external `media_blob` |

> ⚠️ **Naming collision (new — not in §9).** rask **already has** a
> `search-api` brick on `:8802` (Lance line-FTS + thumbnails,
> `components/services/search_api`). The incoming search brick **must not reuse
> that name/port** — use **`media_search` / `media-search` on `:8806`**. The name
> is the chart service key, the uvicorn module, the gateway app-id, the image
> name, and the workspace dir simultaneously; a clash breaks all of them.

---

## 2. Backend bricks — service-kit mechanics

Each rask backend service is **one line**:
`app = make_service_app(*, title, routers, proxy_router=None, lifespan=None)`
(keyword-only; `packages/service-kit/src/service_kit/__init__.py:86`). It
builds `Settings`, mounts each router under `settings.api_prefix`, registers the
shared RFC-9457 handlers + CORS, wires docs/openapi, and adds
`SlashToleranceMiddleware` (Dapr trailing-slash tolerance). Stateful bricks pass
a `lifespan` factory `(settings) -> asynccontextmanager(app)` that opens Lance +
S3 and sets `app.state.*` (the exact shape of our current `lifespan`).

**Add-a-brick checklist** (media_api at `:8805`; repeat for `media_search` at
`:8806`). Underscores for the *dir/package/uvicorn module* (`media_api`,
`media_api:app`); hyphens for everything *build/deploy-facing* (`media-api`) —
mixing them silently breaks `uv sync --package` and the buildx `-f/-t`:

1. **`components/services/media_api/`** — `src/media_api/{__init__.py` (the
   `make_service_app` line, moved from our `app.py`), `lifespan.py`, `health.py`,
   `routes.py`, `dependencies.py`, `schemas.py`, `service.py}` + `pyproject.toml`
   (dist name `media-api`, hatch `packages=['src/media_api']`, **own lancedb
   dep**, `[tool.uv.sources]` service-kit+storage `workspace=true`) + `tests/`
   (a `conftest.py` that sets env **before** importing the app — `make_service_app`
   runs at import time).
2. **root `pyproject.toml`** — append `components/services/media_api` to
   `[tool.uv.workspace] members`, add `media_api` to ruff `known-first-party`,
   add the tests dir to `[tool.pytest.ini_options] testpaths`.
3. **`projects/media-api/pyproject.toml`** — a code-less deployable: name
   `media-api-project`, `dependencies=['media-api']`, **its own**
   `[tool.uv.workspace] members` list of `../../` paths (distinct from the root
   members list — *§9 missed this*).
4. **`.docker/media-api.dockerfile`** — copy `search-api.dockerfile`; swap
   `--package media-api`, module `media_api:app`, port 8805 (EXPOSE +
   HEALTHCHECK + CMD `--port`).
5. **`Makefile`** — add `media-api` to `COMPOSE_IMAGES` (flows into
   `K3S_IMAGES`/`k3s-build`/`k3s-import` automatically).
6. **`gateway/__init__.py` `_routes()`** — add
   `('{prefix}/media', 'media-api', os.environ.get('RASK_MEDIA_API_URL','http://127.0.0.1:8805'))`
   **before** the two core catch-alls (`_pick_route` matches by list order,
   longest-prefix-first; `gateway/__init__.py:46-73`).
7. **`chart/values.yaml` `services:`** — a `media-api` block
   `{module: 'media_api:app', port: 8805, replicas: 1, waitFor: [...]}`;
   `fleet.yaml` templates the Deployment+Service (probes hardcoded to
   `/api/health` — `make_service_app`'s health router already serves it).
8. **`chart/templates/configmap.yaml`** — add
   `RASK_MEDIA_API_URL: http://<fullname>-media-api:8805` (a **distinct** edit
   from steps 6–7 — *§9 missed this*).
9. **`scripts/dev-micro.sh`** — `MEDIA_PORT=$((8805+OFFSET))`, export
   `RASK_MEDIA_API_URL`, and a `run media-api "$MEDIA_PORT" media_api:app` line.

**Gotchas to bake in:** (a) `Settings` has **no default** for
`RASK_VIEWER_INPUT`/`RASK_VIEWER_OUTPUT` — a new brick 500s at startup unless
both are set or service-kit's config is extended. (b) `RASK_API_PREFIX` is
`/api` in chart + dev-micro but `/api/v1` in code defaults — chart probes hit
`/api/health`; keep `SlashToleranceMiddleware`. (c) the port is declared in **~5
files** (values, configmap, dockerfile ×3, dev-micro, gateway fallback) — keep
them in sync.

---

## 3. Frontend — the `media` MFE

rask uses **vertical, route-based** MFEs (not Module Federation): each domain is
a standalone SvelteKit **SSR** app under `components/frontends/<name>`, pinned to
a static base `/default/<name>`, built as a Bun server via `svelte-adapter-bun`,
composed at one origin by Turborepo's Rust proxy (`:3024` dev) / K8s ingress.

**What changes at merge time:**
- **Adapter**: `@sveltejs/adapter-static` → `svelte-adapter-bun`; set
  `kit.paths.base='/default/media'`; drop our hand-rolled `frontend/server.ts`
  proxy (adapter-bun ships its own server).
- **Shell**: rewrite `+layout.svelte` to `import { AppShell } from '@rask/ui/shell'`;
  delete our local `$lib/components/ui/sidebar` chrome; `app.css` imports only
  `@rask/ui/styles/tokens.css`. Add a nav entry in
  `packages/ui/src/lib/shell/nav-config.ts` (a shared-library change).
- **Validation**: **zod → valibot is mandatory** — rask has *zero* zod, `@rask/api`
  hard-depends on valibot, and the toolchain doc mandates it. Port the descriptor
  + envelope schemas in `src/lib/{descriptor,api,table-columns}.ts` to
  `import * as v from 'valibot'`. Response-envelope schemas SHOULD move into
  `@rask/api`; the corpus-agnostic `DatasetView`/`Row` schemas can stay app-local
  (still valibot). Watch `v.variant`/`v.pipe`/`v.transform` for our discriminated
  unions.
- **Data target**: vite proxy `^/api` from FastAPI `:8000` → the rask **gateway
  `:8888`**; introduce `hooks.server.ts = makeGatewayHandleFetch(...)` and move
  reads into `*.remote.ts` `query()` functions (mirror `discover.remote.ts`).
  The binary **apache-arrow IPC** atlas path stays a direct `fetch`, outside the
  JSON `query()` pattern.
- **Register in four places**: `home/microfrontends.json`, root
  `package.json` workspaces, `Makefile FRONTEND_IMAGES`, `chart/values.yaml
  frontend.apps`.

**Ports as-is (no rework, just move under `src/lib`):** the WebGPU renderers
(`atlas/gpu-scatter.svelte`, `graph/gpu-graph.svelte`), the `@xyflow/svelte` +
dagre workflow graph, `apache-arrow` decode, `layerchart`, `d3-force`, and the
descriptor-driven `DatasetView` engine. Type-check already uses tsgo
(`@typescript/native-preview`) — matches rask.

---

## 4. S3 + Lance storage landing  ← the load-bearing part

rask does not read Lance from local disk; it opens **S3-backed datasets through
a namespace** using a `storage_options` dict built by one Settings helper. This
is the piece our backend does not do yet.

### 4.1 The one seam

`lance_namespace.connect(impl, properties)` + a pylance `storage_options` dict,
both from the catalog `Settings` helper
(`lance-ns/services/catalog/core/namespace.py:22`). `Settings.storage_options()`
(`config.py:244-250`) emits exactly Lance's S3 keys:

```python
{ "endpoint", "access_key_id", "secret_access_key",
  "region", "allow_http", "virtual_hosted_style_request" }
```

with **path-style forced** (`virtual_hosted_style_request=false`) because
RustFS/MinIO 403 on virtual-hosted signing. `read_lance`/`write_lance` and
`lance.dataset(...)` all accept `storage_options=` (`lance_docs/ray.md:48,238`);
in `rest` mode the namespace's own options are **merged** with these
(`ray.md:298`). The canonical builder to reuse verbatim is
`lance-ns/services/common/objectfs.py:lance_storage_options` (`:20-42`).

### 4.2 The backend's S3 access paths — DONE + verified (2026-07-16)

> **Status: the core read paths are now S3-capable and proven locally on both
> MinIO and RustFS.** The wiring is additive + env-gated (`MEDIA_S3_ENDPOINT` /
> `MEDIA_S3_ACCESS_KEY_ID` / `MEDIA_S3_SECRET_ACCESS_KEY` / `MEDIA_S3_DB_ROOT` →
> `Settings.storage_options`); all vars unset = the local `db_root` path,
> byte-identical to before (283 backend tests still green, ruff/ty clean). An
> object-store seam (`backend/lancekit/store.py`) replaces `Path.glob`/`is_dir`.
> **Verified live:** `GET /api/datasets`, `/datasets/{id}/descriptor`, and FTS
> `/api/search` all served from `smoke.lance` on MinIO; managed-blob `take_blobs`
> streamed over both stores (`scripts/move_to_s3.py`).

Two code paths were threaded (both DONE):

1. **Registry connection.** `registry.py` now takes `storage_options`, resolves
   the dataset root as an `s3://` URI (or local), connects
   `lancedb.connect(uri, storage_options=…)`, and every `handle.db.open_table`
   caller inherits it (`search_api/target.py` → search works over S3).
2. **Bare `lance.dataset` — discovery + descriptor + blob path.** `introspect`,
   `descriptor`, and `media_api/media.py` open with `storage_options` and
   enumerate via `store.list_lance_stems`. The traversal guard + dataset
   discovery use the object-store seam instead of `Path.glob`/`is_dir` (the
   `s3://`→`s3:/` collapse is gone).

**Capability routes — also DONE (2026-07-16):** `graph.py` / `topics.py` /
`diarization.py` / `voice_service.py` now open via `handle.table_uri()` +
`handle.storage_options` (grep gate: no bare `handle.path / f"…"` opens left).
Verified live on MinIO with `parity_new.lance`: `/api/voice/status` built:true
(323 turns / 15 speakers) and `/api/diarization/{doc}` built:true (114 turns) over
S3; topics/graph return the correct `built:false` (their S3 existence-check path is
exercised — the `built:true` branch needs the 10G transcripts_v2, not uploaded).
**Remaining:** external `media_blob` (file://) still needs re-ingest (§4.4).

### 4.3 Blob streaming over S3 — already-proven contract

Serving blobs from S3 needs no client S3 creds if it goes through the dataset:
open with `storage_options`, `dataset.take_blobs(column, indices=[row])`, stream
the lazy `BlobFile` in ~8 MiB `read_range` windows
(`lance-ns/services/catalog/.../dataplane.py:661,690-702`). `take_blobs` reaches
S3 objects directly (the catalog's health probe relies on it raising when an
external pointer is unreachable). **Our `media.py` Range handler already uses
exactly this `take_blobs`→`BlobFile.read_range` shape** — it just needs the
dataset opened with `storage_options`.

### 4.4 Moving the data — grounded in the real schema

I inspected the live datasets. `transcripts_v2.lance` is
**`data_storage_version=2.2`** (blob-v2 native — the modern read path, no legacy
`lance-encoding:blob` concern), with three blob-v2 columns that split into two
migration classes:

| column | table | kind | portable by plain copy? | action |
|---|---|---|---|---|
| `thumbnail` | documents | **managed/inline** (`kind:0`, bytes in-dataset) | ✅ yes | `aws s3 cp --recursive` rides along |
| `frame_blob` | chunk_frames (145,175 rows) | **managed/inline** | ✅ yes | copies with the dataset |
| `media_blob` | documents (1,154 rows) | **EXTERNAL** (`kind:3`, `blob_uri=file:///…/input/sv/*.mp4`) | ❌ **no** — absolute local `file://` | **rebase or re-ingest** |

So the move is: **(a)** `aws s3 cp --recursive transcripts_v2.lance
s3://bucket/prefix/transcripts_v2.lance` (managed thumbnail + frames + manifest +
indices are self-contained, relative internal refs — `file_format.md:3184`); then
**(b)** fix the external `media_blob`. Two options — **prefer re-ingest**:

- **(B) re-ingest** the source videos with `external_blob_mode="ingest"`
  (`ray.md:899`) so bytes become Lance-managed on S3. The source MP4s live in the
  HCP **`film-raudio`** bucket (5 restored locally in `input/sv/`); re-ingest
  pulls them into managed S3 storage and the external-URI problem disappears.
- **(A) rebase**: copy the external base to S3, rewrite manifest `base_paths`,
  and supply `base_store_params` keyed by base URI at read time (`ray.md:851,897`).
  The catalog READ path passes **no** `base_store_params` and requires every base
  to share the catalog endpoint/creds (`dataplane.py:229-234`), so a
  multi-endpoint external base is unreadable through the catalog — which is why
  **(B) is preferred**.

The `dir`→`rest` flip itself is only a change of `impl`+`properties` at the
`connect()` seam (`dir`: `{root, storage.*}`; `rest`: `{uri, headers.*}` —
`namespace.md:6326`); `build_namespace` is unchanged. `namespace_properties()`
currently only emits the `dir` shape, so a `rest` branch is a small addition.

---

## 5. Ray landing — local `ray.init()` → KubeRay

**Today (this repo):** `rmedia pipeline run <stage>` runs
`read_lance → map_batches(actor pool) → driver-side commit` on a local
`ray.init()` (`make pipeline-run` / `make features-all-ray`). Actors are warm
HTTP clients to the vLLM servers; the driver commits. This is the same shape rask
submits — only the submission changes.

**At merge time:** the pipeline becomes a **Ray Jobs REST** submission to a
KubeRay cluster, exactly like lance-ns's medallion mover
(`services/medallion/services/ray_submit.py`):

- `POST /api/jobs/` with `{entrypoint, submission_id, runtime_env:{env_vars}}`
  via plain `httpx` — **no `ray` package in the submitting brick's image**.
- `submission_id` is **deterministic per (stage, token)** so at-least-once
  redelivery re-attaches to the running job instead of racing a second writer; a
  terminally FAILED job is deleted+resubmitted, a running one is polled.
- `runtime_env.env_vars` carries `FROM_URI`/`TO_URI`/`STAGE` +
  `S3_ENDPOINT`/`S3_KEY`/`S3_SECRET`/`S3_REGION` — the S3 creds ride *in the job*.
- The entrypoint (`scripts/ray_stage_job.py`, baked into the ray-lance image at
  `/home/ray/jobs/`) reads the upstream dataset and writes the downstream at
  **file format 2.2 + stable row ids**, and — crucially — splits **TABULAR**
  (distributed `lance_ray` read→map→write) vs **MEDIA** (blob-v2 present →
  pylance-native `read_blobs`+`blob_array` driver round-trip, since `lance_ray`
  write strips blob typing). **This is the identical blob-flow contract rmedia
  already enforces** (invariants §7.10/§7.11): heavy blobs never transit Ray Data
  blocks; actors compute, the driver commits.

So the merge-time job is: bake `rmedia`'s stage registry into a
`rmedia_stage_job.py` entrypoint (map `STAGE`→registry stage), add a
`submit_stage_job`-style caller in the media brick (or reuse medallion's), point
`ray_address` at the KubeRay head (`http://ray-lance-head:8265` default), and let
KubeRay handle in-job retry. The global fits (EVōC atlas, Toponymy topics,
speaker clustering) stay single-driver — they cannot be row-parallel — and run as
their own bounded jobs.

---

## 6. SHOULD features — port status

Three of four are **fully ported** into the schema-agnostic backend+frontend,
each capability-gated (`descriptor.capabilities`; declared in
`config/descriptors/transcripts_v2.json`, absent in `smoke.json`), returning
`built:false` (never 500) when undeclared — the smoke DB is the live acid test.

| feature | status | endpoint(s) | descriptor capability | frontend | CHECK |
|---|---|---|---|---|---|
| **Voice** | ✅ ported | `/api/voice/{similar,status,identity}` (`media_api/voice.py`) | `voice: speaker_embeddings.embedding` (+ `speakers`) | in-place "Find this voice" mode on Search (mic button, upload, toggle) | `GET /api/voice/status` → `{built:true,turns,speakers}` on transcripts_v2, `{built:false,…}` on smoke (`tests/test_media_api_voice.py:227`); encoder AP≈0.74 on `evals/voice_labels_*` |
| **Topics** | ✅ ported | `/api/topics` (`media_api/topics.py`) | `topics: topics.hierarchy` | `/tree` route (TopicTreemap) | `GET /api/topics` → `{built:true,layers,n_chunks,hierarchy}` vs `built:false` (`tests/test_media_api_topics.py:80`) |
| **Graph** | ✅ ported | `/api/graph/{status,search,entity,subgraph}` + `POST /cypher` (`media_api/graph.py`, live `lance_graph` CypherEngine) | `graph: kg_entities` + `graph_presets` | `/graph` route (WebGPU GpuGraph + Cypher REPL) | `GET /api/graph/status` → `{built:true,entities,relations,mentions,videos}` vs `built:false` never-500 (`tests/test_media_api_graph.py:172`) |
| **MCP** | ⛔ **deferred** | — | — | — | dropped from Phase 2 by decision (P2.9, `LANCE_MEDIA_MERGE.md:184`); `backend/mcp/` holds only stale `__pycache__`, no `/mcp` mount. Re-mount against the descriptor-driven services is a later goal. The surviving `/api/media-clip/{doc_id}` is a general webview-excerpt route, decoupled from MCP. |

> **Doc staleness (merge-time cleanup, not a blocker):** `docs/VOICE.md`,
> `docs/GRAPH.md`, `docs/MCP.md` still cite the pre-split `backend/{voice,graph,mcp}/`
> paths; current code is under `backend/media_api/`. Refresh or supersede them
> during the lift.

---

## 7. Remaining merge-time work (consolidated)

Nothing below happens in this repo; it is the later goal's backlog.

1. **S3 wiring** (§4.2): add a `storage_options` settings helper; thread it
   through the registry connection **and** every bare `lance.dataset` open;
   replace `Path`-based existence/enumeration + the traversal guard with
   object-store equivalents.
2. **Data move** (§4.4): `aws s3 cp --recursive` the datasets; **re-ingest** the
   external `media_blob` source videos with `external_blob_mode="ingest"` (from
   the HCP `film-raudio` bucket).
3. **Backend bricks** (§2): carve `media_api` (:8805) + `media_search` (:8806)
   under `make_service_app`; 9-point registration each; **avoid the `search-api`
   name collision**; set `RASK_VIEWER_INPUT/OUTPUT`.
4. **Frontend MFE** (§3): adapter-bun + `@rask/ui` shell + **zod→valibot** (~284
   call-sites) + gateway data target; four-place registration.
5. **Ray job** (§5): bake an `rmedia_stage_job.py` entrypoint; add a Ray Jobs
   REST submitter; point at the KubeRay head.
6. **Cleanup**: confirm the stale pre-split sibling dirs
   (`backend/{media,search,atlas,system,topics,voice,graph,diarization,mcp}/`)
   are dead before the lift so the merge doesn't carry both; refresh the stale
   feature docs (§6).

**Open decisions:** (a) re-ingest vs base-rebase for external blobs — **prefer
re-ingest**; (b) whether `media_search` is a new `:8806` brick or extends the
existing rask `search-api`; (c) the MFE slot — new `media` vs the dummy `studio`;
(d) where the WeSpeaker upload-encoder lives (heavy dep of the search brick vs a
sidecar).

---

## 8. Merge-readiness verification (re-runnable)

The Phase-2/3 gates that guarantee the pieces are *liftable* — import
independence, no corpus literals, no pipeline coupling, descriptor-only frontend
— still pass, and the Ray DAG the pipeline job will submit is inspectable:

```text
P2.1a GATE OK (search_api does not import media_api)
P2.1b GATE OK (media_api does not import search_api)
P2.3  GATE OK (0 hits)   # no corpus literals in backend/media_api|search_api
P2.8  GATE OK (0 hits)   # no raudio/rmedia imports in backend/
P3.2  GATE OK (0 hits)   # no corpus literals in frontend/src

$ uv run rmedia pipeline plan
stage              shape        table         output             gate         client         actors×cpu/gpu @batch
text_embedding     scan_column  chunks        text_embedding     all          embed          1–4×1/0 @256
summary            scan_column  chunks        summary            all          summarize      1–2×1/0 @128
frame_embedding    blob_column  chunk_frames  frame_embedding    all          embed          1–2×1/0 @64
caption            blob_column  chunk_frames  caption            all          caption        1–2×1/0 @32
caption_embedding  scan_column  chunk_frames  caption_embedding  all          embed          1–4×1/0 @256
extract_frames     append_rows  chunks        chunk_frames       video        frames         1–4×2/0 @32
diarize            append_rows  documents     speaker_turns      audio+video  diarizer       1–2×4/0 @1
voiceprint         append_rows  documents     speaker_embeddings audio+video  voice_encoder  1–2×4/0 @1
```

Gate commands (verbatim from `LANCE_MEDIA_MERGE.md` §6):

```bash
grep -rn "text_embedding\|referenskod\|namn\|chunk_frames\|speaker_turns" backend/media_api backend/search_api --exclude-dir=tests && echo 'GATE FAIL' || echo 'GATE OK (0 hits)'   # P2.3
grep -rn "import raudio\|from raudio\|import rmedia\|from rmedia" backend/ && echo 'GATE FAIL' || echo 'GATE OK (0 hits)'                                                          # P2.8
grep -rn "referenskod\|namn\|text_embedding\|speech_id" frontend/src --include=*.ts --include=*.svelte --exclude-dir=guide && echo 'GATE FAIL' || echo 'GATE OK (0 hits)'          # P3.2
uv run rmedia --db $DB pipeline plan                                                                                                                                               # P4.2 DAG
```

Frontend behaviour is proven live by `frontend/e2e/evidence.mjs` (`EVIDENCE
OK`: 100 hit-cards, `<video readyState=4>`, atlas 145,175 WebGPU points, the
smoke-dataset acid test) and `frontend/e2e/smoke-all.mjs` (transcripts 6/6 +
smoke 6/6 routes clean) — see `LANCE_MEDIA_MERGE.md §10`.

---

## 9. What §9 of LANCE_MEDIA_MERGE.md got wrong

Recorded so the later goal doesn't trust the sketch over this doc:

1. **Name collision** — §9 says `search_api → components/services/media_search`
   but did not flag that rask **already has** `search_api`/`search-api` on
   `:8802`. Use `media_search`/`media-search`.
2. **Missing configmap line** — the gateway upstream
   `RASK_MEDIA_API_URL`/`RASK_MEDIA_SEARCH_API_URL` in
   `chart/templates/configmap.yaml` is a **distinct** edit from both the
   `values.yaml services:` entry and the gateway `_routes()` default.
3. **Missing per-project workspace members** — `projects/<name>/pyproject.toml`
   carries its **own** `[tool.uv.workspace] members` list, separate from the root.
4. **Hyphen/underscore split** — dir/package/module use underscores; dist
   name/dockerfile/COMPOSE/chart-key/image use hyphens. Copying literally breaks
   `uv sync --package`.
5. **Required env** — `Settings` demands `RASK_VIEWER_INPUT`/`RASK_VIEWER_OUTPUT`
   with no defaults; a new brick 500s without them.
6. **Port is declared in ~5 files** — keep values/configmap/dockerfile/dev-micro/
   gateway in sync.

Everything else in §9 held up: `:8805`/`:8806` are free, service-kit is
genuinely dependency-light (no lancedb), `SlashToleranceMiddleware` is real, the
`RASK_API_PREFIX` `/api`-vs-`/api/v1` gotcha is real, and the studio MFE is
`:5177`.

> **Verified against rask HEAD (2026-07-16).** The six load-bearing claims were
> re-checked directly against source (not just recon): the `search-api`/`:8802`
> collision — and that a `viewer` brick *also* already exists, so the "never
> named viewer" rule is a hard collision too — (`chart/values.yaml:59`,
> `scripts/dev-micro.sh:34`, `components/services/{search_api,viewer}/`); the
> keyword-only `make_service_app(*, title, routers, proxy_router=None,
> lifespan=None)` (`packages/service-kit/src/service_kit/__init__.py:89`);
> `:8805`/`:8806` unused (0 hits); gateway `_routes()` with the two core
> catch-alls `(prefix,*core)` + `("/api",*core)` pinned last and matched by list
> order in `_pick_route` (`components/services/gateway/src/gateway/__init__.py:46-71`);
> `viewer_input`/`viewer_output` as defaultless required `str` fields
> (`service-kit/config.py:49-50`); and `api_prefix` default `/api/v1`
> (`config.py:52`) vs `RASK_API_PREFIX=/api` in dev-micro (`dev-micro.sh:24`,
> whose comment documents the exact `/api/*`→catch-all 404). New-brick upstreams
> follow the `RASK_<NAME>_API_URL` convention (`RASK_MEDIA_API_URL`).
