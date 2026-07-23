# lance-ns handoff — what to confirm, what you own, what is already proven

*2026-07-21 · lance-media (repo lance-audio) → lance-ns merge handoff. Paste this whole
file into the lance-ns session. Companion detail: `ANNOTATIONS_SCHEMA_CONTRACT.md`
(schema + write ownership), `LANCE_NS_INTEGRATION.md` (architecture),
`TRAINING_LINEAGE.md` (facet contracts), `QUALITY_AUDIT.md` + `LANCEDB_SDK_AUDIT.md`
(swept backlogs).*

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
2. **The 4 training/model columns** — `trained_in_version` (int64 = Lance version),
   `margin` (f32), `logits` (list/blob), `encoder_embedding` (list/blob): defined there,
   with what type/semantics? We only READ them and deliberately did NOT guess them into
   our schema.
3. **Batch derivers write predictions** — do htr/ocr/asr/detect/embed derivers write
   `source="model:<name>@<ver>"`, `status="prediction"`, `confidence`, `uncertainty`,
   `model_version`? Is **replace-protects-humans** (`WHERE source LIKE 'model:%' AND
   status='prediction'`) implemented?
4. **Catalog read/write contract** — does the catalog expose
   `/v1/table/{id}/query|merge_insert|delete|blobs` that our reader/writer client
   (behind `MEDIA_READ/WRITE_BACKEND`, Local-transport parity-tested) targets unchanged?
   Note: predicates are **strings on this wire** — our shared renderer
   (`services/common/lancekit/predicate.py`) is the single quoting implementation.
5. **OpenLineage** — does the catalog mover emit spec-2-0-2 RunEvents on `merge_insert`
   (our `lineage_emit` then no-ops), carrying the input `DatasetVersionDatasetFacet` +
   the training-run params facets (`ratch_trainingConfig`/`ratch_selection`)?
6. **Jobs enqueue** — a RayJob submit endpoint (our `MEDIA_JOBS_URL`) accepting
   `{producer, op, scope, exemplars}` (INSID3 propagate carries `exemplars`)?
7. **Interactive assist** — a Ray Serve endpoint (our `MEDIA_ASSIST_URL`) serving
   GroundingDINO + SAM (draw/prompt → shapes; encode-once/decode-per-click)?
8. **Training + GreptimeDB** — retrain loop + metric contract
   (`training_metrics`/`al_sampling` + drift-alert → retrain webhook): likely Phase 7 —
   confirm it is the observability layer's, not ours.

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

- **S3 write plane** — annotations over MinIO: wire GET (version header), save-insert
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

## First integration milestone (once the questions are answered)

Point our catalog reader/writer at a **live lance-ns namespace** (drop the in-process
Local transport) and run: annotation read → human write → batch deriver prediction →
re-read, end-to-end through the catalog, with OpenLineage emitted by the mover. That
proves the merge seam live; training, GreptimeDB, and the query engine sequence after.
