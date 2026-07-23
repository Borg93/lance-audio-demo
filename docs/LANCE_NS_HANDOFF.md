# lance-ns handoff — what to confirm, what you own, what is already proven

*2026-07-21 · lance-media (repo lance-audio) → lance-ns merge handoff. Paste this whole
file into the lance-ns session. Companion detail: `ANNOTATIONS_SCHEMA_CONTRACT.md`
(schema + write ownership), `LANCE_NS_INTEGRATION.md` (architecture),
`TRAINING_LINEAGE.md` (facet contracts), `QUALITY_AUDIT.md` + `LANCEDB_SDK_AUDIT.md`
(swept backlogs).*

> **ANSWERED 2026-07-23** — a 28-agent read-only investigation of the live lance-ns
> checkout (`~/Desktop/lance-ns`, adversarially verified, file:line evidence in the
> session record) answered all 8 questions; each carries an **A:** below. Headline:
> nothing collides — lance-ns deliberately has NO schema registry ("the Lance manifest
> IS the schema"), so WE create `annotations` through the catalog and remain
> schema-of-record; the catalog wire contract exists with all four verbs; OpenLineage
> is emitted by their movers (our `lineage_emit` no-ops correctly); jobs/assist have
> no public HTTP shape yet (our seams become them); training exists, auto-retrain
> does not; GreptimeDB + one OTel Collector confirmed as the observability estate
> (our services adopted their obs contract 2026-07-23, commit `3f2a393`).

## The 8 questions to answer ("is this covered already?")

1. **Annotations as a governed catalog table** — does lance-ns host `annotations` as a
   catalog table, and does its canonical schema match our 25 contract columns + identity
   (`ANNOTATIONS_SCHEMA_CONTRACT.md` §Columns TODAY)? Who owns the schema of record at
   merge? **Settle while the table is still empty:** the canonical identity key must go
   arity-generic off `descriptor.identity.key_fields` (everything merges/keys on it —
   derivers, saves, exports), and any agreed additions from question 2 plus
   `created_at`/`updated_at` and the temporal fields (`t_start`/`t_end`/`frame_idx`)
   land BEFORE the first predictions do — retrofitting keys under data is the one
   ordering mistake this handoff exists to prevent.
   **A: NOT PRESENT — we create it.** lance-ns hosts no annotations table and defines
   no annotation schema anywhere; there is deliberately no schema registry
   (`docs/DATA-CONTRACT.md`: "the Lance manifest IS the schema; the version IS the
   handshake"). The creator of a table IS schema-of-record: `POST /v1/table/{id}/create`
   takes our 25-column Arrow-IPC schema verbatim, seeds FGA ownership
   (`grant_on_create`: creator=owner, parent edge cascades namespace grants) and
   lineage coordinates (schema METADATA only — no columns injected). Caveats:
   (a) tables routed through the medallion CASCADE gain platform columns
   `stage`/`source_rowid` (+ `thumbnail`/`embedding` for image blobs) — annotations
   write via the direct catalog path, so they stay untouched, but those four names
   are reserved-in-practice; (b) optional per-mover `requiredColumns` + the quality
   gate enforce declared columns pre-promotion, additive evolution never blocked;
   (c) maintenance policy (retention/compaction, tag-pinned versions survive) is a
   per-table/namespace/project JSON record — set one on annotations at create.
2. **The 4 training/model columns** — `trained_in_version` (int64 = Lance version),
   `margin` (f32), `logits` (list/blob), `encoder_embedding` (list/blob): defined there,
   with what type/semantics? We only READ them and deliberately did NOT guess them into
   our schema.
   **A: NOT PRESENT — ours to define.** Zero hits for all four anywhere in lance-ns
   (`margin` appears only as CSS). Confirms our decision to not guess them; define
   them (with Q1's while-empty additions) when the training loop needs them.
3. **Batch derivers write predictions** — do htr/ocr/asr/detect/embed derivers write
   `source="model:<name>@<ver>"`, `status="prediction"`, `confidence`, `uncertainty`,
   `model_version`? Is **replace-protects-humans** (`WHERE source LIKE 'model:%' AND
   status='prediction'`) implemented?
   **A: NOT PRESENT.** Their only deriver family appends artifact COLUMNS
   (thumbnail/embedding) — no prediction ROWS, no source/status stamping, no
   replace-protects-humans anywhere. OUR annotator write plane brings this contract
   with it (already shipped + tested on our side); their derivers adopt the stamping
   when they start writing annotation rows.
4. **Catalog read/write contract** — does the catalog expose
   `/v1/table/{id}/query|merge_insert|delete|blobs` that our reader/writer client
   (behind `MEDIA_READ/WRITE_BACKEND`, Local-transport parity-tested) targets unchanged?
   Note: predicates are **strings on this wire** — our shared renderer
   (`services/common/lancekit/predicate.py`) is the single quoting implementation.
   **A: ALL FOUR EXIST, plus more.** `POST /v1/table/{id}/query` → Arrow-IPC file;
   `/merge_insert` → Arrow-IPC stream body + `on`/`when_matched_update_all`/… query
   params, optional `source`+`source_version` lineage extras and an
   `X-Lance-Run-Facets` header; `/delete`; blob routes. Predicates are strings on
   the wire ✓. First integration milestone = point our RestCatalog transports at
   these and reconcile param shapes (our writer already speaks Arrow-IPC file vs
   their stream — verify at first contact).
5. **OpenLineage** — does the catalog mover emit spec-2-0-2 RunEvents on `merge_insert`
   (our `lineage_emit` then no-ops), carrying the input `DatasetVersionDatasetFacet` +
   the training-run params facets (`ratch_trainingConfig`/`ratch_selection`)?
   **A: YES.** Two emitters, both spec-2-0-2 sharing `services/common/openlineage.py`
   constants (same schemaURL our mirror pins): every catalog write endpoint
   (create/insert/merge_insert/update/delete/commit) inline-emits a measured
   RunEvent, and the medallion movers emit around stage transforms. At merge our
   `lineage_emit` no-ops and their mover speaks for the writes, as designed;
   custom params facets ride the `X-Lance-Run-Facets` header.
6. **Jobs enqueue** — a RayJob submit endpoint (our `MEDIA_JOBS_URL`) accepting
   `{producer, op, scope, exemplars}` (INSID3 propagate carries `exemplars`)?
   **A: NOT PRESENT as a drop-in.** Their Ray-submit seam is an INTERNAL client of
   the Ray Jobs REST API (`medallion/services/ray_submit.py` — the template our
   `ratch/core/jobs.py` mirrors), not a public HTTP endpoint accepting our shape.
   Decision at merge: our annotator `/jobs` endpoint BECOMES that HTTP wrapper
   (translating {producer,op,scope,exemplars} → a runner Ray Job through the shared
   seam), or the annotator submits directly via `ratch.core.jobs`. Either way the
   submit protocol is already aligned by construction.
7. **Interactive assist** — a Ray Serve endpoint (our `MEDIA_ASSIST_URL`) serving
   GroundingDINO + SAM (draw/prompt → shapes; encode-once/decode-per-click)?
   **A: NOT PRESENT** — lance-ns has zero Ray Serve code (the demo Ray head doesn't
   even expose :8000). rask's sibling checkout has working Serve deployments to
   pattern-match. Our assist lands as a `runners/` Serve deployment (the shape our
   runners already declare) behind `MEDIA_ASSIST_URL`; until then the in-repo mock
   keeps the UX testable.
8. **Training + GreptimeDB** — retrain loop + metric contract
   (`training_metrics`/`al_sampling` + drift-alert → retrain webhook): likely Phase 7 —
   confirm it is the observability layer's, not ours.
   **A: CONFIRMED — the observability layer's.** Training exists (POST /train →
   JetStream TRAINING stream → Dapr-triggered submit-and-ack Ray job; job emits its
   own lifecycle + OTel metrics `lance.training.*` from a short-lived
   MeterProvider); AUTO-retrain/drift-alert does NOT exist yet; `training_metrics`/
   `al_sampling` are unnamed there — the metric contract to converge on is their
   `lance.training.*` OTel namespace. GreptimeDB is THE unified metrics/logs/traces
   store behind ONE OTel Collector (apps export OTLP; Dapr sidecars are scraped via
   prometheus receiver; Perses dashboards).

## What lance-ns owns at merge (we deliberately did NOT build these)

- **Table maintenance** — our `ratch maintain`/`tag` (version GC + milestone tags,
  tagged + latest survive) is a thin scheduled call to be replaced by catalog-owned
  maintenance. Retention must cover the compare-versions audit horizon.
- **Catalog-routed writes + FGA** — our server stamps `reviewer` from a trusted header
  (`X-User`); lance-ns swaps in the verified token subject (OpenFGA keys on it). The
  downstream stamping seam is unchanged.
- **Export** — a separate schema-driven serializer microservice (Lance schema → COCO/
  YOLO/CSV/HF mapping registry). Not app code on either side.
- **Column definitions** in question 2 — theirs to name/type; we read.
- **Object-byte / cold-query cache tier** (foyer) — ours is only the version-keyed
  search RESULT cache.

## Proven seams (evidence; commits named inline, un-tag/resync land with this goal's commit)

- **S3 write plane** — annotations over S3 (MinIO at proof time; the store of record
  today is RustFS at `127.0.0.1:9100`): wire GET (version header), save-insert
  merge_insert commit, stale `base_version` → 409, `?version` time-travel, tag batch;
  `materialize-blobs` run (376 MB managed; 65 MB blob streams over S3). `d00211f`.
- **Derivers model** — lance-ns fan-out §4 proven 2026-07-16: 166/166 tests, patch at
  `docs/proofs/lance-ns-media-derivers.patch`.
- **3-service import boundaries** — the annotator + search services import zero viewer
  modules (shared kernel = `services/common` — lancekit + core/schemas/deps); the
  annotator-service lift touches no viewer code. `4283800`.
- **Write-plane UX** — save/tags/un-tag/compare-versions/saved-views browser-proven
  (E2E 51 checks across annotator/temporal/read-plane suites); the 409 optimistic-
  concurrency handshake is API-proven (S3 combo + backend tests).
- **Schema single source** — backend `EMPTY_SCHEMA` ≡ seeder ≡ engine, test-enforced.

## Merge mechanics (verified 2026-07-23 against the live checkout)

**Correction to our assumption:** lance-ns is NOT a uv workspace — it is ONE uv
project ("virtual app": the image runs `uv sync --no-install-project`; service code
is COPY'd, `PYTHONPATH=/srv/services`, `pythonpath=["services","."]`). Adding a
backend service = drop `services/<name>/` in (zero dockerfile edits — the shared
image copies the whole `services/` tree), add its light deps to the ONE
`pyproject.toml`, then register: a Deployment+Service template following the
`chart/templates/{services,medallion}.yaml` pattern, a `services.<name>:
{module, port, daprAppId, replicas}` block in `chart/values.yaml`, Dapr sidecar via
pod ANNOTATIONS gated on `dapr.sidecars`, OTel via the `lance.otelEnv` helper +
the `opentelemetry-instrument uvicorn` command (behind `lance.otelEnabled`) — the
contract our services adopted 2026-07-23 (`common/obs.py`, commit `3f2a393`).
Frontend zones register in `chart/values.yaml frontend.apps` (one Deployment per
entry, k8s name `web-<name>`); our two zone apps slot in as `media` + `annotator`
entries. Heavy model deps NEVER enter the shared image — `runners/<name>/` become
their own images on KubeRay worker groups (our TODO § Merge-time). Ray telemetry
crosses the Jobs boundary there (TRACEPARENT + OTEL_* injected into runtime_env —
their `ray_submit` does; our `jobs.py` gains the same two lines at merge).

## First integration milestone — **PROVEN 2026-07-23**

Ran live, end-to-end, from this repo: the lance-ns catalog booted from its checkout
(read-only tree, `uvicorn catalog.main:app` against our RustFS at `127.0.0.1:9100`,
auth/FGA off = their default) and the whole loop passed through OUR transports:

- **Table created THROUGH the catalog**: `POST /v1/namespace/media/create` +
  `POST /v1/table/media$annotations/create` (Arrow-IPC stream body) with the
  25-column contract schema + identity + `created_at`/`updated_at` → 31 columns,
  landed at `s3://lance-catalog-m1/…` v1; maintenance policy set at create
  (`retention_days=30, retain_versions=100`); schema round-trips via `/query`.
- **The loop**: read → human save (`merge_insert`) → **409 handshake** verified
  against THEIR version primitive (`describe?load_detailed_metadata=true` →
  `version`; `check_base_version_value` rejects the stale save) → model-prediction
  write (`source="model:x@1"`, `status="prediction"`) → **replace-protects-humans**
  (predictions deleted, human rows survive) → insert-only leg never clobbers a
  human row → re-read. Evidence: `tests/test_catalog_live.py` (auto-skips without
  `MEDIA_CATALOG_URL`) — 6/6 green against the live catalog.
- **Wire fixes were ours alone**: merge bodies are Arrow-IPC **stream** (their Rust
  reader rejects IPC *file* — fixed in `RestCatalogWriteTransport`); the version
  primitive needed `load_detailed_metadata=true` (plain describe is location-only).
- **OpenLineage**: with `LANCE_LINEAGE_EMIT_ENABLED=true` the catalog emitted
  **6 spec-2-0-2 RunEvents for our writes** (captured at a local sink:
  `job=lance-catalog/merge_insert.media$…`, output facets `version`+`schema`);
  our own emit no-ops on the catalog path (`Settings.effective_lineage_sink` —
  asserted in the test) so runs are never double-counted.

Training, GreptimeDB, and the query engine sequence after; the annotator service's
full catalog-mode save path (wiring `table_version()` into its wire GET) is the
next increment on our side.
