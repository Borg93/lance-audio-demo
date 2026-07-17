# lance-media × lance-ns lakehouse — does the silver/gold layer work with us?

> **Short answer: yes, cleanly — and the fit is unusually good, because lance-ns
> already built the exact seams we need.** Our batch pipeline drops into the
> medallion **media lane** as silver derivers + gold aggregations; our
> search/viewer app becomes a read-only **application layer** over the catalog.
> There is exactly **one** real design decision (single-table derivers vs our
> multi-table fan-out, §4) and a set of governance wires we gain for free.
>
> Grounded in a full read of `Borg93/lance-ns@419101f` (not the local checkout):
> `docs/{MEDALLION,RASK-INTEGRATION,DATA-CONTRACT}.md`,
> `services/medallion/{mover,producer}.py` +
> `services/medallion/services/{derivers,compute,ingest,quality,ray_submit}.py`,
> `services/catalog/api/v1/endpoints/data.py`, `services/common/{sources,sinks}.py`,
> and `frontend/apps/web`.

---

## At a glance

Two things live in this repo — **compute** and an **app** — and they attach to
the lance-ns lakehouse at two different points. The batch pipeline plugs into the
medallion cascade as **silver/gold derivers** (★ = proven, see §4); the app is a
**read-only client** of the catalog, which *is* the query engine.

```mermaid
flowchart LR
    analyst(["Analyst"]):::user

    subgraph media["lance-media — THIS repo"]
        rmedia["rmedia pipeline<br/>Ray + vLLM<br/><i>compute only</i>"]:::mine
        app["Search / Viewer app<br/>SvelteKit + WebGPU<br/><i>analysis frontend</i>"]:::mine
    end

    subgraph ns["lance-ns — the LAKEHOUSE (= the query engine)"]
        catalog["Catalog REST API<br/>/query · /blobs · /describe"]:::engine
        subgraph cascade["medallion cascade — Dapr movers (raw → bronze → silver → gold)"]
            direction LR
            bronze["bronze<br/>raw media blobs"]:::data
            derivers["silver derivers<br/>★ OUR stages plug in<br/>audio + fan-out PROVEN"]:::proven
            silver["silver-media<br/>embeddings · frames ·<br/>speaker-turns"]:::data
            gold["gold-media<br/>★ OUR aggregations<br/>atlas · topics · KG"]:::proven
            bronze --> derivers --> silver --> gold
        end
        catalog -. reads .-> silver
        catalog -. reads .-> gold
    end

    rask["rask — deploy platform: KubeRay · Dapr · rustfs · CNPG · gateway · MFEs<br/>hosts lance-ns + lance-media; our app replaces its viewer + search MFE"]:::host

    analyst --> app
    rmedia ==>|"ingest + run as movers"| bronze
    app ==>|"read only · governed · no S3 creds"| catalog
    rask -. hosts .-> ns
    rask -. hosts .-> media

    classDef user fill:#0d3b2e,stroke:#46f9b8,color:#eafff5
    classDef mine fill:#12345a,stroke:#6aa9ff,color:#eaf2ff
    classDef engine fill:#3a2a4d,stroke:#b388ff,color:#f3ecff
    classDef data fill:#2a2438,stroke:#b388ff,color:#efe9ff
    classDef proven fill:#0d3b2e,stroke:#46f9b8,color:#eafff5,stroke-width:3px
    classDef host fill:#3a2418,stroke:#ff9457,color:#ffe9d9
```

**In one line:** our *compute* becomes lakehouse silver/gold; our *app* reads that
data back through the catalog; rask hosts everything. No merge is performed here —
this is the map plus the de-risking proof.

---

## 1. The three repos and who owns "the query engine"

| repo | role | is it the query engine? |
|---|---|---|
| **lance-ns** | the **lakehouse**: Lance Namespace REST catalog over pylance `DirectoryNamespace` on S3/rustfs + the event-driven medallion cascade + lineage (→ AGE) + OpenFGA governance + compaction. Ships its **own** frontend (`apps/web` — a lineage/catalog explorer). | **YES.** The catalog's `POST /v1/table/{id}/query`, `/count_rows`, `GET /{id}/blobs` (`data.py:512,570,578`) + Lance's own indexes are the query/data-access engine. |
| **rask** | the **deployment platform**: operators (CloudNativePG, rustfs-operator, KubeRay+Kueue, NATS, Dapr, Traefik), the service-kit bricks, the SvelteKit MFEs, the gateway. | no — it hosts. |
| **lance-media** (this repo) | (a) **batch compute** — the rmedia Ray pipeline; (b) a **search/viewer app** — the media-content frontend. | **no — and that's the point.** We are *compute* + an *analysis frontend*, not a catalog. |

lance-ns's own `docs/RASK-INTEGRATION.md` already plans to **contribute the
lakehouse into rask** (using rask's operators). So all three converge on rask.
Your framing — *"our batch processing is just for compute; this is the search
application layer to look at the data in the lakehouse; not our query engine;
this replaces the viewer + search MFE in rask"* — is exactly the boundary
lance-ns already draws. We slot into two named seams it left open.

---

## 2. The medallion cascade, and where we plug in

lance-ns runs **raw → bronze → silver → gold** as Dapr movers over NATS
JetStream (`docs/MEDALLION.md`). The movers are **one generic binary**
(`medallion.mover:app`) differing only by `MEDALLION_*` env; each reads its
upstream Lance dataset at the triggered version, runs a **compute seam**
(`read → transform → write → version`), emits a `DERIVED_FROM` OpenLineage edge,
and fires the next trigger. There is already a **media lane**:

```
POST /ingest-media ─▶ lance-ray head ─▶ bronze-media$objects (blob-v2, fmt 2.2)
                                         │  medallion.media
                                         ▼
                              media-to-silver mover ─▶ silver-media$features
                              (derivers.py: image → thumbnail + embedding)   (terminal today)
```

The multimodality lives in **one file**: `services/medallion/services/derivers.py`.
`_DERIVERS: tuple[(content_probe, Deriver)]` is content-dispatched, and its
docstring is an open invitation:

> *"an audio deriver slots into `_DERIVERS` later … Adding a media type = one
> probe + one deriver here; the platform and the chart do not change."*

**That is our merge surface.** Our rmedia stage registry is a much richer set of
derivers than the single `is_image → thumbnail+embedding` one that ships:

| rmedia stage | medallion mapping | layer |
|---|---|---|
| `extract_frames` (video → frame JPEGs) | a **video deriver** (new `_DERIVERS` entry) — but it *fans out rows* (see §4) | silver |
| `text_embedding`, `frame_embedding`, `caption`, `caption_embedding` | derivers that **add columns** to the carried table — the exact `Deriver = (table, payloads) -> table` shape | silver |
| `diarize` (→ speaker_turns), `voiceprint` (→ speaker_embeddings) | **audio derivers** — also row-fan-out (§4) | silver |
| atlas projections, topic tree, KG | **silver → gold aggregations** — a *new gold media stage* (media lane is silver-terminal today) | **gold (net-new)** |

So: **silver = our per-item derivers; gold = our global aggregations.** The gold
media stage does not exist in lance-ns yet — our atlas/topics/KG would be the
first media gold datasets, written with the embedded `lineage` JSONB column that
the gold mover contract requires (`RASK-INTEGRATION.md` § lance-ray seam).

---

## 3. What we already match (the seams line up)

We independently built to the same contracts, so these need **no change**:

- **Ingest seam.** lance-ns: `SourceAdapter.iter_objects() -> SourceObject{uri, data}`
  (`services/common/sources.py`, adapters `LocalDirSource`/`S3Source`). Ours:
  `src/rmedia/ingest/sources.py` — same `SourceAdapter` + content-sniffed MIME.
  Our documents-table ingest ≙ their `ingest_to_bronze` (blob-v2 at fmt 2.2,
  `source_uri` provenance, `enable_stable_row_ids=True`).
- **Blob-v2 at 2.2.** Our `transcripts_v2.lance` is already
  `data_storage_version=2.2` with `lance.blob.v2` columns — the format the whole
  media lane assumes.
- **Blob flow rule.** lance-ns notes lance-ray exposes blob-v2 as plain
  `LargeBinary`, so blob stages re-attach `blob_field` via `read_blobs`/`blob_array`
  on write — **identical** to our invariants §7.10/§7.11 (heavy blobs never
  transit Ray blocks; `take_blobs`→`BlobFile` streaming; driver commits).
- **The compute seam is our driver.** Their `compute.py` "fake-Ray"
  `read → stamp → write → version` is the in-process stand-in for *"a distributed
  Ray Data job (lance-ray on KubeRay)"* — which is precisely our
  `read_lance → map_batches(actor pool) → driver commit`.

---

## 4. The one real design decision — single-table derivers vs our fan-out DAG

The medallion `Deriver` signature is **column-additive on one table**:
`Callable[[pa.Table, list[bytes]], pa.Table]` — it adds columns (thumbnail,
embedding) and carries the rest forward (`ARTIFACT_COLUMNS` classifies
DERIVED-vs-IDENTITY for column-level lineage). It **cannot express a stage that
emits a new table with a different row cardinality.**

Our pipeline is a **multi-table DAG**:
- `chunks` (N rows) → `extract_frames` → `chunk_frames` (M rows, new table)
- `documents` → `diarize` → `speaker_turns` (new table) → `voiceprint` → `speaker_embeddings`

Three ways to reconcile (pick at merge time):
1. **Sub-lanes** — model each fan-out as its own medallion lane
   (`silver-frames`, `silver-turns`, `silver-voiceprints`), each a
   generic-mover hop keyed off the media trigger. Most faithful to the current
   platform; more movers/values.
2. **Extend the deriver contract** to `(table, payloads) -> dict[str, pa.Table]`
   (a stage may emit sibling tables). One platform change, then our whole
   registry drops in as derivers. Cleanest for us; a real (small) platform PR.
3. **Composite silver deriver** — our pipeline runs its internal DAG inside one
   `media-to-silver` step and writes the fan-out tables itself, presenting a
   single silver "features" output to the cascade. Fewest platform changes; hides
   the sub-graph from medallion lineage (we'd emit our own sub-edges).

The column-additive stages (embeddings, captions) fit model 1/2/3 unchanged;
only the row-fan-out stages force the choice. **Recommendation: model 2** — it is
the smallest change that keeps every stage first-class in lineage/quality, and it
matches the docstring's "one probe + one deriver" ethos.

> **PROVEN 2026-07-16 (both halves) — `docs/proofs/lance-ns-media-derivers.patch`.**
> Against a fresh `lance-ns@419101f` clone, two lance-media stages were wired in
> and run through the platform's **real** compute seam on real Lance datasets:
> - **Silver deriver (audio).** An `is_audio` probe + `derive_voiceprint` added to
>   `_DERIVERS` (~40 lines): an audio blob flows bronze→silver through the
>   *unchanged* `transform_stage` and gains a content-derived `voiceprint` column,
>   blob carried at 2.2, `source_uri` + `stage` provenance intact, the column
>   recorded on the lineage WROTE facet. It also corrected `is_audio` to
>   content-verify (matching their `is_image`), fixing a test that had used a fake
>   WAV header as "junk media."
> - **Fan-out (model 2).** A parallel `_FANOUT_DERIVERS` registry + `derive_fanout`
>   + an additive `transform_stage_fanout`: a video blob emits a **new `frames`
>   sibling table** (1 video → N frame rows, each keyed to its parent by
>   `source_rowid` + `id`), blob-v2 at 2.2 — the exact shape `extract_frames` /
>   `diarize` / `voiceprint` need. **Zero change** to the mover, cascade, lineage,
>   or gates; it composes as a second registry alongside the column-additive one.
>
> Result: **166/166** of their medallion/media/blob/cascade/compute unit tests
> pass (their full suite + the two new proofs). Model 2 is not just recommended —
> it's demonstrated to drop in additively.

---

## 5. Governance we gain for free (today we have none)

By becoming medallion movers, our stages inherit lance-ns's three enforcement
points (`docs/DATA-CONTRACT.md`) at zero code cost to us:

- **Version handshake** — read upstream at the triggered version, write a new
  version, emit `DatasetVersionDatasetFacet`. *"The manifest is the schema; the
  version is the handshake."* Our descriptor's vector/identity/search bindings
  become the consumer **`requiredColumns`** declaration the quality gate asserts
  landed (turns a runtime "missing column" stall into a pre-promotion block).
- **OpenFGA gates** — `can_create_table` (writer) on silver, `can_promote`
  (validator) on gold, checked as the mover's own service identity. Our pipeline
  has no authz today; it gets ReBAC.
- **Quality gate** — `row_count_positive` + `not_null(key)` + `blob_resolves`
  (a 1-byte probe that catches a dangling external `media_blob` at promotion, not
  at first playback — directly relevant to our external-URI videos).
- **Lineage → AGE** — every hop is a `DERIVED_FROM` edge; our
  chunks→frames→embeddings→atlas DAG becomes a queryable provenance graph
  (`GET /datasets/<gold>/upstream`). rask has **zero** lineage today; we'd arrive
  already wired into it.
- **Creds** — workload identity (KubeRay projected SA) + short-TTL table-scoped
  creds via `POST /v1/table/{id}/credentials`. **No durable secret on compute** —
  which resolves the S3-wiring open question from `RASK_LANDING.md §4` in the
  lakehouse's favor: compute vends creds per-table rather than holding static S3
  keys.

---

## 6. The application layer — our search/viewer app over the catalog

Your other half — *"the search application layer to analyze and look at the data
in the lakehouse … replaces the viewer + search MFE"* — maps to the catalog's
**read surface**, governed at reader tier:

| our app needs | catalog endpoint | notes |
|---|---|---|
| schema/descriptor discovery | `POST /v1/table/{id}/describe` | feeds our runtime introspection (vector cols via `list_size`, blob cols, indexes) |
| FTS / vector / hybrid search | `POST /v1/table/{id}/query` (Arrow IPC) | Lance indexes are the engine; we pass predicates + `vector_column_name` |
| row counts, facets | `POST /v1/table/{id}/count_rows` | |
| media / frame bytes | `GET /v1/table/{id}/blobs?column=&row=&version=` | **Range-capable (206/416)**, governed `can_read_data` — our player/atlas fetch here **with no S3 creds** (browser-safe), replacing today's direct `lance.dataset` blob open |

This is the resolution of the local-filesystem gap in `RASK_LANDING.md §4`: the
app stops opening datasets directly and instead reads through the governed
catalog — the app never needs `storage_options` at all; the **catalog** holds
storage, the app holds a bearer token.

Two important distinctions:
- **We are complementary to lance-ns's `apps/web`, not a duplicate.** Their
  frontend is a **governance/lineage explorer** (routes: `lineage`, `tables`,
  `warehouses`, `models`) — it shows the *platform*. Ours shows the *content*
  (media search, playback, the WebGPU atlas, topic tree, KG). Same monorepo shape
  (SvelteKit 2 + Svelte 5 + turborepo + bun + a shared `packages/ui`), so ours
  lands as a **sibling app** (`apps/media`) or a rask MFE, reusing the shared UI.
- **Our descriptor is the semantic render-contract on top of the structural data
  contract.** lance-ns governs the physical dataset (manifest/version/lineage/FGA);
  our descriptor governs how a corpus *renders* (identity, display, search modes,
  atlas spaces, capabilities). They compose — descriptor discovery is a `/describe`
  call plus our declared半 (the `lance_media.descriptor` schema-metadata key).

---

## 7. Verdict + sequence

**Will silver/gold work with this? Yes — now demonstrated, not just argued.**
Silver = our per-item derivers; gold = our global aggregations (net-new
media-gold stage). The seams (ingest, blob-v2/2.2, blob-flow, compute) already
match by independent convergence; the governance (version/FGA/quality/lineage/
creds) is a free upgrade; and the one real design choice — the fan-out deriver
contract — is **proven to drop in additively** (§4: audio silver deriver + video
fan-out both run through their real compute seam; 166/166 of their tests green;
patch in `docs/proofs/lance-ns-media-derivers.patch`).

Merge sequence (all *after* the lance-ns→rask fold-in, none in this repo now):
1. **Land rmedia as media-lane derivers** — contribute `modalities/` derivers +
   the stage registry into `services/medallion/services/derivers.py` (+ the §4
   fan-out contract). The generic mover, cascade, lineage, FGA, quality stay put.
2. **Add the media-gold stage** — atlas/topics/KG as `silver→gold` movers writing
   gold datasets with the embedded lineage JSONB.
3. **Submit as a KubeRay job** — our `read_lance→map_batches→commit` becomes the
   real Ray Data job behind the agnostic Jobs-REST submit seam
   (`ray_submit.py`); honor the 4-point lance-ray seam contract.
4. **Re-point the app at the catalog** — swap direct `lance.dataset` opens for
   `/query` + `/blobs` + `/describe`; land it as `apps/media` (or a rask MFE)
   reusing `packages/ui`; descriptor discovery via `/describe` + the metadata key.
5. **Declare consumer columns** — feed our descriptor's search/identity bindings
   into the gold mover's `requiredColumns` so the quality gate guards them.

The batch pipeline stays *just compute*; the catalog stays *the query engine*;
our app stays *the analysis frontend* — which is exactly the boundary you set.
