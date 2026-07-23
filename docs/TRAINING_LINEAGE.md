# Training + sampling provenance — capture · store · propagate · view (DESIGN)

> How we capture *what was actually trained on*, its *params*, and the *data snapshot* —
> and how that propagates to the user and is viewed. Grounded in this repo's OpenLineage
> emission (`src/ratch/lineage.py`, `services/common/lancekit/lineage_emit.py`), the `annotations`
> schema (`services/annotator/annotations/schema.py` `EMPTY_SCHEMA`), and the AL loop (`docs/ACTIVE_LABELING.md`).
> Cross-checked against OpenLineage-for-ML + GreptimeDB conventions.

## Thesis: three complementary systems, cross-linked by shared keys

Training provenance is **not one system**. It is three layers, each answering a different
question, stitched by three join keys. We **invent nothing for the snapshot** (Lance already
time-travels) and **nothing for the graph shape** (`build_run_event` already emits spec-2-0-2).
We (a) widen the facet **payload**, (b) add a few provenance **columns**, and (c) make our
read/annotate app the **lens** that walks a prediction back to the exact rows a model saw.

| System | Question it answers | What it is |
|---|---|---|
| **Lance version** | *which exact data?* | the immutable **SNAPSHOT** — `checkout_version(v).to_table(filter=predicate)` is bit-exact replay. No copy, no export: the training set is `dataset@v` + a WHERE-predicate. |
| **OpenLineage** | *what produced this / can I reproduce it?* | the discrete provenance **GRAPH** — `dataset@v → training-run(params, sampling) → model@v → predictions(source=model:@v)`, one connected DAG. |
| **GreptimeDB** | *how good was it / how did it evolve?* | the continuous metrics **TIME-SERIES** — loss/eval/drift/sampling-volume per step, via OTLP. |

**Join keys (the whole trick):**
- `runId` — joins the lineage graph ↔ the GreptimeDB metrics.
- `model_version` (string, already on every prediction row) — joins a prediction ↔ the model node ↔ its producing run.
- `dataset_version` (int, `int(ds.version)`) — joins the Lance snapshot ↔ the input dataset facet ↔ the GreptimeDB tag.

Hard boundaries: **no training curves in lineage facets; no what-trained-on-what edges in GreptimeDB.**

## Capture — "what was actually trained on"

The training set is never snapshotted separately — it's a **WHERE-query at a pinned version**.
A training run does three captures:

1. **PIN the input.** Read `int(ds.version)` of the `annotations` table at job start. The
   training set = `SELECT … WHERE status='accepted' [∪ replay ORDER BY uncertainty DESC LIMIT k]
   AND trained_in_version < current`, evaluated at that version.
2. **STAMP the rows.** After the model commits (and passes the eval gate), `merge_insert('id')`
   writes `trained_in_version = <pinned input version>` onto every consumed row. This single
   column closes the loop: the next retrain's `AND trained_in_version < current` skips
   already-consumed rows (monotonic replay), and `WHERE trained_in_version = <v>` is the reverse
   index that reconstructs *exactly which rows fed model v*.
3. **EMIT the graph edge + STREAM the metrics** (below).

Because `annotations` already carries `source`/`status`/`confidence`/`uncertainty`/`model_version`
and every Save is one atomic Lance version, "what was trained on" is fully reconstructable by
time-travel + the `trained_in_version` stamp — **nothing is duplicated.**

## Snapshot — three carriers fully determine the input

`dataset@v` + predicate + seed = bit-exact replay: `lance.dataset(uri).checkout_version(v).to_table(filter=predicate)` + seed.

- **`DatasetVersionDatasetFacet.datasetVersion = int(ds.version)`** on `inputs[]`. **Current gap:**
  inputs are `[{namespace,name}]` only, and `measure_stage` (lineage.py) runs on the *output* URI —
  so we must add a **measure-on-INPUT** path emitting `{version, rowCount, SchemaDatasetFacet}` on inputs.
- **selection-predicate facet** — the WHERE/vector query that curated the subset (`dataset@v` alone
  under-determines it). Core `sql` run-facet or a custom `ratch_selection`.
- **`random_seed`** inside the params facet — shuffle/sampling determinism.
- **Model artifact = its own OUTPUT Dataset** (HF Hub URI, `namespace=models`, its own
  `DatasetVersionDatasetFacet` = model hash/rev, `format`) with columnLineage edges from
  training-set columns → model. It is **not** run through `measure_stage` (which assumes a
  written Lance table with `count_rows`).

## Params — one custom RUN facet (+ standard facets)

`run.facets` is `errorMessage`-only today. Add **`ratch_trainingConfig`** (there is no ratified
core hyperparameters/MLModel facet as of 2026):

```
ratch_trainingConfig {
  _producer, _schemaURL,
  hyperparameters { lr, batch_size, epochs, optimizer, weight_decay },
  random_seed,
  sampling { strategy: uncertainty|least-confidence|diversity|balanced, k, replay_fraction, holdout },
  base_model  # HF repo@rev
}
```

Plus standard `nominalTime` + `ParentRunFacet` (nest the run under its pipeline/experiment parent).
These params live **inside the RunEvent JSON** — Marquez/the catalog versions them immutably
against `runId`, so that's the durable, reproducible config record. **Discrete config in the graph,
time-series in GreptimeDB, joined by `runId`.**

## GreptimeDB — the continuous side (via OTLP)

Two tables, tagged for the join:

```
training_metrics(ts, TAGS run_id, model_version, dataset_version, split;
                 FIELDS loss, lr, grad_norm, throughput, eval_map, iou, precision, recall, f1)
al_sampling(ts,     TAGS round, strategy, class;
                 FIELDS labels_added, mean_uncertainty, entropy_p50, entropy_p95, class_count, psi_drift)
```

Ingested via OTLP / Prometheus remote-write; viewed in Grafana (PromQL/SQL). The model node
deep-links to "this model's training curves" by `run_id`.

## Propagate + view — click a prediction → the rows it trained on

**Propagation:** the training run stamps `trained_in_version`, emits the RunEvent
(`dataset@v → model@v`, with the params/selection facets), streams metrics to GreptimeDB keyed by
`runId`/`model_version`. RE-PREDICT then writes prediction rows with `source=model:<name>@<version>`
+ `model_version=<version>` (columns that already round-trip). Every prediction is **self-describing**.

**Viewing — three lenses, and we own the app that stitches them:**
- **(a) Provenance columns in-app** — the read plane already ranks `status='prediction' ORDER BY
  uncertainty DESC`; a prediction hit-card shows `source`/`model_version`/`confidence`; clicking
  `model_version` resolves the model node.
- **(b) Lineage graph** (Marquez/catalog at merge) — walk `dataset@v → training-run → model@v →
  predictions`, inspect facets (schema, version, params, selection, columnLineage); our app
  deep-links by `runId`/`model_version`.
- **(c) Compare-versions panel** (OURS, already shipped — `annotation_versions`) — extend so a
  version a training run touched surfaces its `model_version` + a link to the run.
- **(d) GreptimeDB dashboards** (Grafana at merge) — loss/eval curves + per-round sampling stats.

Net UX: **click a prediction → its model → the training run (params + live metrics) → the exact
input dataset version → time-travel to the rows it trained on.**

## Ours (pre-merge) vs the merge

**OURS — buildable in this repo now:**
1. **Provenance COLUMNS** on `EMPTY_SCHEMA` (`services/annotator/annotations/schema.py` —
   add while the table is still empty; `seed_annotations.py` imports it, so one edit
   propagates): `trained_in_version` (int), `created_at`/`updated_at` (ts),
   `margin` (f32) + `logits` (list<f32>), `encoder_embedding` (list<f32>/blob ref).
2. **Lineage facet SHAPE** — extend `lineage.py`/`build_run_event`: measure-on-input + input
   `DatasetVersionDatasetFacet`, `ratch_trainingConfig` + `ratch_selection` run facets, the
   model-artifact output path, `nominalTime`/`ParentRun`, the multi-input columnLineage fix —
   byte-identical to lance-ns constants so it drops into the merged builder seam.
3. **`trained_in_version` stamp policy + the replay-query contract**.
4. **Provenance UI** — extend the shipped compare-versions panel to surface `source`/`model_version`
   + deep-link prediction → model → run; read-plane hit-card provenance chips.
5. **The OTLP metric name/tag contract** (`training_metrics`/`al_sampling`) so any trainer emits the agreed shape.

**MERGE — lance-ns / Ray / infra (NOT built here):** the retrain COMPUTE (lance-ray + vLLM Ray
Data fine-tune jobs); model ARTIFACTS in HF Hub + registry; the retrain TRIGGER (count/drift/
class-imbalance from Lance version deltas); the frozen-holdout EVAL GATE; the governed
write + real lineage TRANSPORT (catalog-routed `merge_insert` → OpenLineage over Dapr/NATS +
OpenFGA + gold QC; swap our `build_run_event` for lance-ns's via `emit_stage_lineage(builder=)`);
the GreptimeDB deploy + Grafana + OTel Collector/Alloy.

## Open questions (decisions before predictions land at scale)

- **Canonical identity key** — `doc_id/speech_id/chunk_id/frame_idx` vs engine `page_id/dataset_id`
  (`docs/LANCE_NS_HANDOFF.md` question 1). Provenance rows key on it.
- **Do we fine-tune now, or is the near-term loop predict+relabel+re-predict with an external
  model** (training deferred)? Decides whether *we* emit `ratch_trainingConfig` or only at merge.
- **`trained_in_version` write timing** — stamp on gated COMPLETE (only if promoted), so a
  failed/ungated run never consumes rows out of replay (recommended).
- **Per-media uncertainty definition** — HTR CTC/token · ASR beam · detection objectness/entropy,
  each needs a real comparable-scale score for cross-corpus ranking + `al_sampling`.
- **Selection-predicate home** — core `sql` facet vs custom `ratch_selection` — must match whatever
  lance-ns standardizes so merged/standalone events stay byte-identical.
- **Auto-accept policy** — does `confidence` gate `status` transitions, and does replay include
  auto-accepted rows?
- **Model artifact** — plain output Dataset vs a first-class registry entity (namespace/naming) —
  avoid inventing a facet lance-ns will override.
