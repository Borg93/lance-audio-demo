# Annotation platform — gap audit → merge-aware roadmap

Synthesized from a 5-finder gap audit (read plane · write plane/3-modes · lance-ns merge ·
cross-cutting infra + AL loop · tool-parity vs FiftyOne/Label-Studio/X-AnyLabeling/CVAT).
See [ACTIVE_LABELING.md](ACTIVE_LABELING.md) · [LANCE_NS_INTEGRATION.md](LANCE_NS_INTEGRATION.md) ·
[RA_ANNO_MERGE.md](RA_ANNO_MERGE.md).

## The headline

**The merge is NOT blocked by missing UI — the read/find plane is already FiftyOne-class**
(type-agnostic search: keyword/vector/hybrid/scene/image/voice + any declared embedding · a real
WebGPU atlas with lasso/box/cluster embedding-interaction cross-filtered with search · a sortable
HitTable · grid/list/table/map views). **It is blocked by a write plane that literally cannot
create, move, or delete an annotation, and by every lance-ns governance seam (canonical identity
key, catalog-routed writes, per-user FGA auth) being unwired.** Land those cheap-but-foundational
pieces first; the batch-deriver / active-learning half is the largest lift and comes after.

## Merge-blockers (most-blocking first)

1. **Canonical identity key** (S to decide / M to plumb). Reconcile engine `schema.ts`
   `page_id/dataset_id` vs backend `doc_id/speech_id/chunk_id/frame_idx`; make `annotate.py`
   arity-generic off `descriptor.identity.key_fields`. Everything merges/keys on it — derivers,
   lineage, atlas `hitKey`, human saves. Cheap now, catastrophic once prediction rows exist.
2. **The write plane must actually persist new shapes / geometry moves / deletes** (L). Today
   `im.onCommit` only sets `_geoDirty`, `ArrowDataPlugin` has no append, and backend `merge_insert`
   is update-only — so all 9 drawing tools silently drop shapes and Delete is a no-op. The engine's
   `AnnotationStore` already implements append/delete/structural-undo; it is simply **unwired**.
3. **Route writes through the governed catalog** (M) — a `TableWriter` seam mirroring the built-but-
   unwired `reader.py`, calling the catalog `POST /{id}/merge_insert` instead of in-process
   `ds.merge_insert`. Yields OpenLineage MERGE_INSERT + FGA gate + quality gate + credential vending
   for free. Today every save is ungoverned/unlogged/unauthenticated.
4. **Per-user OpenFGA authz + request-scoped bearer tokens** on read AND write (L). Backend has NO
   auth layer (CORS-only); `reader.py` uses one static token. Without per-user identity, lineage
   author binding is empty, reviewer provenance is self-asserted, write-tier STS creds unreachable.
5. **Batch job-submit seam + annotation-emitting fan-out silver deriver** (XL). `apply()` returns a
   hardcoded `{queued:'not wired'}`; no `/jobs` route. Every non-manual mode (AI-assist, bulk auto,
   judge, propagate, re-predict) is inert. Needs the proven-but-unmerged 1-blob→N-row fan-out
   (`docs/proofs/lance-ns-media-derivers.patch §4`) + replace-protects-humans policy.
6. **Real per-modality uncertainty/confidence** (L). Columns round-trip but are seed-only, so the
   review queue can't actually rank — the one load-bearing AL idea is inert. Depends on #5.
7. **Gold QC-gate** (curated silver→gold via `quality.py` + `can_promote`) — net-new + undesigned.
   Design early, build later; a first mechanical merge can read silver directly.

## Phased roadmap

| Phase | What | Depends on | Merge |
|---|---|---|---|
| **0 · Schema & identity** (settle before predictions land) | canonical identity key (arity-generic off descriptor) · add `created_at/updated_at/trained_in_version` + temporal `t_start/t_end/frame_idx` + `encoder_embedding` while the table is empty · `/api/schema` roles endpoint | — | blocker |
| **1 · Make the write plane WRITE** (app-local, no catalog dep) | wire the engine `AnnotationStore` (append/delete/structural undo) into the controller · `annotate.py` add insert+delete branches + geometry fields · If-Match/version 409 handshake · label-class config + hotkeys + multi-apply-to-selection + accept-and-advance + port the dropped `AnnotationTable` | 0 | blocker |
| **2 · Catalog-routed reads+writes** (the mechanical merge seam) | `TableWriter` → catalog `POST /{id}/merge_insert` · wire the built `reader.py` into the GET + `/blobs` + `/query` · OpenLineage falls out free | 0 | blocker |
| **3 · Governance** | backend auth (OIDC middleware) · thread per-user bearer through reader+writer · OpenFGA on read+write · STS write creds · bind lineage author to `token.sub` | 2 | blocker |
| **4 · Batch compute = silver derivers** (the AI/auto half) | job-submit seam (query-selection → RayJob) + status polling · fan-out annotation deriver + replace-protects-humans · real per-modality uncertainty + curation/brain scores · wire the producer registry + model-catalog UI | 0,2,3 | blocker |
| **5 · Read = true catalog read layer** | compare-versions (list/checkout/diff + panel) · saved/named views · **read→annotate handoff** + modality-agnostic individual viewer (retire video-only PlayerPane) · faceting + virtualization | 1,2 | needed |
| **6 · Multimodal + assist + interop** | audio (peaks.js) + video (frame overlay) viewers · interactive SAM-click/DINO/INSID3 transports · COCO/YOLO/PAGE-XML export jobs · tag↔annotation unification + relations/links UI | 0,4,1 | needed |
| **7 · Full AL loop · gold · team ops** (decoupled follow-on) | retrain trigger + holdout eval gate + promote-on-improvement · build the gold QC-gate · eval framework (TP/FP/FN, mAP/IoU, eval-patches) · land the apps/media MFE reusing `@lance/ui` + the 3-service split · projects/assignees/WAL/plugins/video-tracking | prior | mixed |

## Notable findings

- **The find plane is genuinely strong** — search is already type-agnostic (a new embedding column
  is searchable with zero UI edits), the atlas embedding-interaction is real WebGPU. For the merge it
  stays but repoints at the catalog `/query`.
- **A real regression:** the full `AnnotationTable.svelte` (columns/sort review table, commit
  `6afcbc7`) existed in the pre-monorepo `frontend/src` but was NOT carried into `apps/media` — the
  annotate plane there has only the sidebar list. (Restore in Phase 1.)
- **The read→annotate loop is broken end-to-end:** the read plane produces selections (hitKey sets +
  scope SQL) but `/annotate` is demo-keyed and the individual viewer is video-only. (Phase 5.)
