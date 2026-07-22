# Annotations table — schema + write-ownership contract (merge handoff)

> The `annotations` Lance table is a **shared table with two writers**. This is the exact
> per-column contract so the lance-ns side knows what *it* writes vs what *we* write, and
> so nobody redefines the other's columns. Grounded in `backend/media_api/annotate.py`
> `_EMPTY_SCHEMA` (the table today) + the training/AL additions from `TRAINING_LINEAGE.md`.

## The two writers

- **US — the human write path** (`save_annotations`, interactive → `merge_insert` = one
  atomic version): human-authored content + human provenance + audit.
- **THEM — lance-ns batch/training derivers** (silver/gold stages as lance-ray jobs;
  predictions → rows): model provenance + scores + the training-loop columns.
- **Replace-protects-humans:** a deriver's write is predicated
  `WHERE source LIKE 'model:%' AND status='prediction'`, so re-running a model **never
  clobbers a human-accepted/edited row.** (Contract to confirm on the lance-ns side.)

## Columns TODAY (`_EMPTY_SCHEMA`, live + round-tripping) + identity

| Column | Type | Written by | When | Owner |
|---|---|---|---|---|
| `doc_id`,`speech_id`,`chunk_id`,`frame_idx` | str,int,int,int | both (server-stamps from route keys; deriver stamps its rows) | on insert | **descriptor** (arity-generic `identity_values`) |
| `id` | str | both | on create | shared (human=uuid; tag=`tag:{doc}:{k}:{label}`; deriver=model id) |
| `shape_type` | str | both | on create | shared (`rectangle`/`polygon`/`segment`/`tag`/`mask`…) |
| `x`,`y`,`width`,`height`,`rotation`,`polygon` | f32…,list\<f32\> | both | draw / prediction | shared (geometry) |
| `t_start`,`t_end` | f32 | both | audio/video segment or pinned moment | shared (temporal) |
| `text` | str | both | HTR/transcription | shared |
| `label` | str | both | human label / predicted class | shared |
| `status` | str | both | human `accepted`/`rejected`; deriver `prediction` | shared **by value** |
| `source` | str | both | human `"human"`; deriver `"model:<name>@<ver>"` | shared **by value** |
| `reviewer` | str | **US** | every save (X-User author seam) | **ours** |
| `confidence` | f32 | **THEM** | deriver inference | **lance-ns** |
| `uncertainty` | f32 | **THEM** | deriver inference (norm. entropy/margin) | **lance-ns** |
| `model_version` | str | **THEM** | deriver (`<name>@<ver>`) | **lance-ns** |
| `group`,`group_id` | str,str | US (curation) | human grouping | ours (mostly) |
| `reading_order` | i32 | US | human HTR ordering | ours |
| `difficult` | bool | US | human flag | ours |
| `links` | str(JSON) | US | human relations | ours |
| `mask` | str(b64 PNG) | both | brush / predicted mask | shared |
| `metadata` | str(JSON) | both | either | shared |

## Columns to ADD — and who defines them (the open contract)

| Column | Proposed type | Written by | Read by | **Who defines it** |
|---|---|---|---|---|
| `created_at` | timestamp[us] | **US** (on insert) | viewer recency, replay order | **ours** (write-path) |
| `updated_at` | timestamp[us] | **US** (every write) | audit | **ours** (write-path) |
| `trained_in_version` | **int64** (a Lance version) | **THEM** (training svc, gated complete) | replay `< current`; viewer "which model consumed this" | **lance-ns** |
| `margin` | f32 | **THEM** (deriver) | real AL uncertainty | **lance-ns** |
| `logits` | list\<f32\> or blob | **THEM** (deriver) | distributional uncertainty | **lance-ns** |
| `encoder_embedding` | list\<f32\> or blob | **THEM** (batch encode: SAM/DINO) | interactive SAM decode-per-click; diversity AL | **lance-ns** |

**Consequence:** we do **not** add the bottom-4 to our `_EMPTY_SCHEMA` — they're lance-ns's to
define (name/type/semantics), and guessing wrong = migration-to-match-theirs, not migration
avoided. `created_at`/`updated_at` are the only additions that are ours (our write path stamps
them). Everything else is already in the table.

## Questions for the lance-ns session ("is this covered already?")

1. **Annotations as a governed catalog table** — does lance-ns host `annotations` as a
   catalog table, and does its canonical schema match the 25 columns above + identity? Who
   owns the schema of record at merge?
2. **The 4 training/model columns** — `trained_in_version` (int64 = Lance version),
   `margin` (f32), `logits` (list/blob), `encoder_embedding` (list/blob): are these defined
   there, and with what type/semantics? (We only READ them.)
3. **Batch derivers write predictions** — do `htr/ocr/asr/detect/embed` derivers write rows
   with `source="model:<name>@<ver>"`, `status="prediction"`, `confidence`, `uncertainty`,
   `model_version`? Is **replace-protects-humans** (`WHERE source LIKE 'model:%' AND
   status='prediction'`) implemented?
4. **Catalog read/write contract** — does the catalog expose `/v1/table/{id}/query`,
   `/merge_insert`, `/delete`, `/blobs` that our reader/writer client (behind
   `MEDIA_READ/WRITE_BACKEND`, Local transport parity-tested) targets unchanged?
5. **OpenLineage** — does the catalog **mover** emit spec-2-0-2 RunEvents on `merge_insert`
   (so our `lineage_emit` becomes a no-op), and does it carry the **input
   `DatasetVersionDatasetFacet`** + a **training-run params facet** contract
   (`ratch_trainingConfig`/`ratch_selection` per `TRAINING_LINEAGE.md`)?
6. **Jobs enqueue** — is there a RayJob submit endpoint (our `MEDIA_JOBS_URL`) accepting
   `{producer, op, scope, exemplars}` (INSID3 propagate carries `exemplars`)?
7. **Interactive assist** — is a Ray Serve endpoint (our `MEDIA_ASSIST_URL`) serving
   GroundingDINO + SAM (draw/prompt → shapes; encode-once/decode-per-click)?
8. **Training + GreptimeDB** — is the retrain loop + the **metric contract**
   (`training_metrics`/`al_sampling` tables + the drift-alert → retrain-webhook trigger)
   covered? (Likely Phase 7 / not yet — this is the observability layer's, not ours.)

## First integration milestone (once the above is confirmed)

Point our catalog reader/writer at a **live lance-ns namespace** (not the in-process Local
transport) and run **annotation read → human write → a batch deriver prediction → re-read**
end-to-end through the catalog, with OpenLineage emitted by the mover. That proves the merge
seam live; everything else (training, GreptimeDB, the query engine) sequences after it.
