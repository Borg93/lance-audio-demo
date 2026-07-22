# AI auto-labeling + the active-learning relabel loop

Distilled from a multi-agent study of **X-AnyLabeling** (AI-assisted labeling) and
**ActiveLabelingSystem** (active-learning loop), mapped to lance-media. See also
[RA_ANNO_MERGE.md](RA_ANNO_MERGE.md).

## Verdict

lance-media is already the right shape — the studies confirm the thesis, not reshape
it. The three planes are in place: **compute** (ratch silver stages as lance-ray +
vLLM jobs = a distributed, real version of ALS's ShadowTrainer/batch inference);
**store** (the `annotations` Lance table already has `status` /`source`/`confidence`/
`reviewer`, served as Arrow IPC, written via local-first → `merge_insert`); **query**
(type-driven agnostic search already ranks `WHERE status='prediction' ORDER BY
confidence ASC` with zero edits once the column is populated).

**Auto-labeling belongs in BATCH, not the app.** Both tools prove whole-image
detectors / VLM / HTR / ASR are batchable; only prompt-interactive SAM is not. And
**search + viewer as the relabel judge** is a strict upgrade over both tools' linear
"next-unchecked" walk — corpus-wide, re-queryable every round, backed by real model
confidence + vector diversity.

## The two planes: view+analyze (read) vs annotate (write)

The platform splits by read-vs-write — CQRS over the Lance catalog:

- **VIEW + ANALYZE — read-heavy.** *Look* (individual pages · gallery scroll · table)
  and *find* (search · filter · **embedding-interaction**, FiftyOne-style). A governed
  **read-only** catalog client (query/blobs/describe), cacheable, scales independently;
  home of the WebGPU atlas/embedder + the PixiJS viewer. **Its output is a `Selection`.**
- **ANNOTATE — write-heavy.** *Mutate* annotations via `LabelOp`s over a Selection.
  Interactive → local-first overlay → Save (`merge_insert` = 1 atomic version); batch →
  silver derivers (lance-ray). Replace-protects-humans, versioned, governed.

`Selection` is the ENTIRE bridge read→write (and the viewer/search-vs-annotator service
seam): the read plane forms it, the write plane consumes it.

## The write plane is mode-agnostic — 4 orthogonal axes, not a flat "review queue"

The `annotations` schema stays MODE-BLIND (records `source` + `status` + `confidence`,
never "manual"/"bulk"). Every labeling action is a `LabelOp` over four axes
(`frontend/apps/media/src/lib/labeling/`):

| Axis | Values | Formed by |
|---|---|---|
| **Selection** (target) | `one` · `picked` · `query` · `all` | browse → one/picked · search/embedding-interact → query/all |
| **Producer** (who) | `human` · `model` · `propagate` · `judge` | the producer registry (config-driven model catalog) |
| **Op** (what) | `set` · `verdict` · `predict` · `propagate` · `judge` | — |
| **Execution** (where) | `interactive` (local-first→Save) · `batch` (silver deriver) | — |

The **three modes are regions** in (Producer × Execution):

| Mode | Producer | Execution | Selection | Tool analog |
|---|---|---|---|---|
| **Manual** | human | interactive | one · picked | Label Studio ("apply to selection") |
| **AI-assisted** | model · propagate | interactive (or quick batch) | one · picked · all-from-small | X-AnyLabeling · **INSID3** few-shot · SAM click |
| **Bulk / auto / judge** | model · judge · propagate | **batch** | query · all | ActiveLabelingSystem loop · FiftyOne bulk-tag |

- **INSID3** (visinf/INSID3) = the `propagate` producer: training-free in-context
  segmentation on a frozen DINOv3 backbone — 1–few exemplar masks → propagate to a
  selection/all ("apply to all from small data"). DINOv3 features are the SAME embedding
  space the analyze plane interacts with, so exemplar-pick and propagate share a space.
- **AI-as-Judge** is batch, not interactive — it *looks at data* (scores/verifies existing
  predictions), never at a person's screen; it feeds confidence/uncertainty.

**Scaffold status:** `labeling/types.ts` (Selection/Producer/Op/Execution/LabelOp) +
`labeling/producers.ts` (typed registry: human, sam-click, insid3, grounding-dino, htr,
vlm-judge, embed-propagate) DONE; the controller's `apply(op)` routes the **manual** path
for real (manual = human·verdict·interactive·one — proving the annotator isn't coupled to
the review flow) and returns typed `queued`/`unsupported` for batch + interactive-assist
producers (their predict/decode transport + batch-deriver enqueue are the follow-ups).

## Auto-labeling = a silver deriver per model family

`htr` / `ocr` / `asr` / `detect-segment` / `embed` run as **lance-ray + vLLM batch
jobs** over bronze, writing ROWS into `annotations` with `status="prediction"`,
`source="model:<name>@<version>"`, real `confidence`, `uncertainty`, `model_version`.

- **Never clobber humans:** `merge_insert("id")` re-run REPLACES prior *predictions*
  but is predicated `WHERE source LIKE 'model:%' AND status='prediction'` — reviewed/
  accepted/locked rows survive (X-AnyLabeling `replace`/`locked`, ALS status guard).
- **Store the richest primitive** (mask/embedding), project to polygon/bbox at review
  time — no re-inference.
- **Interactive assist is the narrow exception — exactly one call: click-to-segment.**
  Encode-once-in-batch / decode-per-click: the batch SAM/DINO stage persists the image
  encoder embedding to a Lance column; the annotator's point/box prompt decodes that
  cached embedding (in-browser ONNX or a light endpoint). Everything else is a queued
  mini-batch, surfaced async, tagged by media id + Lance version, dropped on mismatch.

## The loop: SEARCH → JUDGE → RETRAIN → RE-PREDICT

| Step | HAVE | NEED |
|---|---|---|
| **SEARCH** (what to relabel) | type-driven search + result cache; the 4 AL strategies are queries: uncertainty `ORDER BY uncertainty DESC`, least-confidence `ORDER BY confidence ASC`, diversity = vector-ANN near-dup/max-min over `embedding` (beats ALS's Jaccard), balanced `GROUP BY label` | populate `uncertainty`/`confidence` (DONE for the demo); a review-queue search mode |
| **JUDGE/RELABEL** | PixiJS+Arrow annotator, local-first → `merge_insert`, status lifecycle, one-key accept-with-suggested-label | a review-queue view consuming a search result set (not a linear walk) |
| **RETRAIN** | ratch stages as lance-ray jobs (= ALS ShadowTrainer, real); training set = `WHERE status='accepted'` ∪ replay (`ORDER BY uncertainty DESC LIMIT k`) — Lance IS the dataset | a retrain trigger (count-gate OR time / uncertainty-drift / class-imbalance, from Lance version deltas) + a before/after eval gate on a frozen holdout |
| **RE-PREDICT** | re-run the batch stage on the new model version, rewriting predictions | model artifacts in HF Hub; a config repoint recorded as a lance-ns lineage edge |

## Adopt / Skip

**Adopt:** batch-predict-then-review split (server-side); a materialized
`uncertainty`/`confidence` column (the one load-bearing idea) so "what next" is a sort,
not a recompute — using REAL scores (CTC/beam/objectness/logit-entropy); the 4 query
strategies as search modes; encode-once/decode-per-click SAM; replace-protects-humans
merge policy; a config-driven model catalog (X-AnyLabeling YAML-per-model → silver-stage
descriptor with capability flags); a holdout eval gate before promotion; a compare-
versions viewer panel (Lance version diff).

**Skip:** all PyQt-desktop mechanics (QThread/ONNX-in-process/downloads → async job +
SSE); the god-file pattern (LabelingWidget 7118 lines, ModelManager's 98-branch
if/elif → registry membership, not per-model branches); sidecar-JSON storage +
format-converter-as-store (Lance is native; COCO/YOLO are export JOBS); ALS's hand-rolled
DatasetVersioner + ReplayBuffer + symlink registry (Lance versioning + lance-ns catalog
do it natively); ALS's synthetic single-box softmax "uncertainty"; the mocked ALS
training path.

## Schema (status: partly DONE)

**Done** (`annotations` table + `annotate.py` `_EMPTY_SCHEMA` + `seed_annotations.py`):
`confidence`, `uncertainty`, `source`, `model_version` added and round-tripping; the
review-queue query is proven (`status='prediction' ORDER BY uncertainty DESC`).

**Still to add when the loop lands:** `created_at`/`updated_at` (recency/replay/audit);
optional `margin` / `logits` (real distributional uncertainty); an `encoder_embedding`
reference (decode-per-click); `trained_in_version` (exclude already-trained rows from
replay).

## Open questions (user decisions)

1. **Identity key** — backend keys `doc_id/speech_id/chunk_id/frame_idx`; ra-anno's
   engine schema keys `page_id/dataset_id`. Pick one canonical key before predictions
   land at scale.
2. **Auto-accept policy** — QA-sampling auto-accept (high-confidence → accepted, small
   held-back fraction) vs human review of ALL predictions. Decides whether `confidence`
   gates status transitions or is advisory.
3. **Interactive decode transport** — in-browser ONNX SAM decoder vs a light endpoint
   (ships the encoder embedding to the client, or keeps it server-side).
4. **Do we own fine-tuning now,** or is the loop initially predict + relabel + re-predict
   (with an external model), training deferred? (Memory frames training as later.)
5. **Retrain trigger home** — a lance-ns catalog event/schedule vs an app trigger.
6. **Per-media uncertainty definition** — HTR (CTC/token), ASR (beam), detection
   (objectness/entropy) each need a real, comparable-scale score for cross-corpus ranking.
