# Structure conformance map — lance-media ↔ lance-ns

*2026-07-21 · the merge-shape proof for the "lance-ns-shaped" goal. Deviations are
justified inline; anything unmarked is a 1:1 correspondence.*

## Backend: `services/` ↔ lance-ns `services/`

| lance-media | lance-ns analog | Notes |
|---|---|---|
| `services/common/` | `services/common/` | Shared kernel. Ours: `lancekit` (dataset access, predicate, keys, registry, reader/writer, lineage_emit), `core/` (config, exceptions, handlers, middleware, probes), `schemas/`, `state.py`, `deps.py`. Theirs: blobs/objectfs/exceptions/dapr_*/oidc/fga/outbox. Same rule both sides: services import shared primitives ONLY from common, never a sibling. |
| `services/viewer/` | `services/catalog/` (read-plane role) | Media/blobs (Range/206 via `take_blobs` — the same BlobStream idiom as their `dataplane.py`), transcripts, datasets, atlas, graph, topics, voice, diarization. |
| `services/search/` | *(no direct analog — rask's search brick)* | FTS/vector/hybrid over declared bindings; `create_search_app` remains the composition/test seam. |
| `services/annotator/` | `services/medallion/` (write-plane role) | Annotations (merge_insert + 409 handshake), assist, jobs — the catalog-write client at merge. |
| `<svc>/main.py` | `<svc>/main.py` | Thin entry: module-level `app`, ALL construction in lifespan onto `app.state`, `install/register` problem handlers, `/livez` + `/readyz` gated on `startup_complete`/`shutting_down`. Byte-for-byte the same rungs. |
| `<svc>/core/config.py` | `<svc>/core/config.py` | Per-service Settings + `@lru_cache` getter. **Deviation:** shared data-plane vars stay `MEDIA_*` (one Lance root serves all three, like their shared `LANCE_*`); only service-local knobs get the per-service prefix (`VIEWER_PORT`/`SEARCH_PORT`/`ANNOTATOR_PORT`). |
| `<svc>/api/v1/router.py` (+ `endpoints/`) | same | Aggregation router per service. **Deviation:** search's single endpoint IS `api/v1/router.py` (no `endpoints/` package — one resource); viewer + annotator have `endpoints/`. **Deviation:** wire paths stay `/api/*`, not `/v1/*` — the shipped frontend contract; the `v1` directory is the structural convention. **Deviation:** the annotations resource keeps its package layout (`annotator/annotations/` with its own router composition) — it was skill-conformance-audited as a unit and the E2E pins its byte-stable OpenAPI order. |
| `<svc>/services/` | same | Infra-free logic: viewer `clips/points/voice_service/wespeaker`; search `service/target/filters/frames/postprocess/rerank/result_cache/vector/spec/clients` + `encoders/`. |
| RFC 9457 problem+json | `common/exceptions.py` | Ours: `common/core/{exceptions,handlers}.py` — domain errors → problem details, never `HTTPException`. Same contract. |
| One uv project, `pythonpath` per package | `pyproject.toml` (`pythonpath=["services","."]`) | Ours: hatch packages `services/{common,viewer,search,annotator}` → imports `from common…`, `from viewer…` — the same import convention. |
| `make services-up/down`; `rmedia serve` fans out 3 uvicorns | Tiltfile (one image, per-pod uvicorn target) | Same one-artifact/many-entrypoints model, minus k8s. |

### Deliberately NOT adopted (merge-time infra — lance-ns owns it)

- **Dapr sidecars** (pubsub, cron bindings, secret store), **dapr_auth token guard**,
  **outbox**: our services adopt the *shape* so sidecars drop in; running them here
  would duplicate lance-ns's deployment. (Standing rule: don't build the infra in
  this repo.)
- **OpenFGA/OIDC**: the author seam (`X-User` → server-stamped `reviewer`) is where
  their verified token subject plugs in; the FGA model is theirs.
- **Helm chart/Tilt**: deployment artifacts belong to the merged repo.

### Known coupling (dissolves at merge)

- **`common/lancekit/lineage_emit.py` imports `rmedia.lineage`** (`build_run_event`,
  facet builders). This is the one common→pipeline import — the lineage facet contract
  is genuinely shared between the batch derivers (rmedia) and the annotator write path.
  At merge lance-ns's catalog mover emits lineage, so this seam becomes a no-op; until
  then the shared facet logic has one home in `rmedia.lineage` rather than being copied.
  (The old "backend has zero rmedia imports" invariant retired with the monolith.)

## Frontend: `frontend/` ↔ lance-ns `frontend/`

| lance-media | lance-ns analog | Notes |
|---|---|---|
| turborepo: `apps/*` + `packages/*`, bun workspaces | same (`apps/web`, `packages/{config,ui}`) | Same toolchain (turbo, bun, oxlint/oxfmt). |
| `packages/{engine,labeling,ui,api}` (`@lance/*`) | `packages/ui` | Shared code extracted per the turborepo skill's "no shared code in apps" rule; JIT internal packages (exports at src), own deps declared, package tasks (check/test) in the turbo pipeline; apps consume `workspace:*`. `cn` + shadcn type helpers live in `@lance/ui/utils`. |
| `apps/media` (viewer zone) + `apps/annotator` | `apps/web` | Two SvelteKit zone apps composed by path routing: the viewer origin proxies `/annotate` → the annotator app (dev :5176), which owns the whole write-plane closure (`lib/viewer`, `lib/labeling` runes state). The read→write handoff is a URL (`/annotate?keys=…`) — the Selection seam — which is why the zone cut is clean. Module Federation deliberately NOT used (wrong scale). **Deviation:** the viewer app keeps the name `apps/media` (its package id is the historical `@lance-media/media`); renaming the directory is cosmetic churn deferred to the merge. |
| Dev proxy + prod `server.ts` path-route `/api/*` per domain + `/annotate` → the annotator zone | — | `/api/annotations|assist|jobs` → annotator :8103, `/api/search` → :8102, rest → viewer :8101; `/annotate` → the annotator zone app. The dev vite proxy and the prod Bun `server.ts` (viewer + annotator each) share the SAME map. A real gateway owns this routing at merge; the thin servers keep dev/prod parity for local repro. |
