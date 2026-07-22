# Modular architecture plan — Lance · Ray · DuckDB, low coupling

> The **build plan** for the model in [WHATS_LEFT §0](WHATS_LEFT.md): *Lance
> stores, Ray evolves the columns, DuckDB (lance extension) explores, the schema is dynamic.*
> Where WHATS_LEFT lists the bets and [COMPARISON_LIGHTLY.md](COMPARISON_LIGHTLY.md)
> says why, **this doc is about module boundaries** — the seams and interfaces
> that let each layer change without dragging the others. Goal stated plainly:
> **low coupling, high modularity.** Every "today" line below is grounded in the
> current `ratch` source. For the exhaustive, ranked list of the ~30 hardcoding
> chokepoints this plan removes, see [DYNAMISM_BLOCKERS.md](DYNAMISM_BLOCKERS.md).

---

## 0. Does this fit the current stack? Yes — the seams already exist

The good news from auditing the code: `ratch` is *already* structured for this.
The data-evolution shape and the swap-points are in place; the plan mostly
**formalizes** them rather than rewriting.

| Seam already present | Where | Why it matters |
|---|---|---|
| Column-creation core is **client-free** | `src/ratch/features/engine.py` (`upsert_scan_column` / `upsert_blob_column`, pure fns over a path + a `compute` callable) | A Ray/Ray Data driver can fan these out unchanged — the docstring even says so. |
| Enrichment is a **registry**, one entry per column | `src/ratch/features/columns.py` (`FEATURES: dict[str, Feature]`, each `Feature{name, table, run}`) | Adding/altering an enrichment is local; the CLI is a thin loop over it. |
| Model clients are **injected `Protocol`s** | `EmbeddingClient` / `CaptionClient` / `SummarizeClient`; tests inject fakes | The compute logic doesn't depend on vLLM — swap the impl (HTTP server → Ray Serve → Ray actor) without touching column logic. |
| Online model access is a **factory by URL** | `backend/clients.py` (`ensure_embedder`/`ensure_reranker`, URL from `Settings`, 503 on failure) | Point it at a Ray Serve ingress via `MEDIA_EMBED_URL` — no search-code change. |
| One **transport**, shared online+offline | `src/ratch/vllm/base.py` (`VLLMTransport`: httpx POST + threadpool fan-out) | The online query path and offline backfill already hit the same server the same way. |
| Search wire is **open dicts** | `backend/search/*` return `list[dict[str, Any]]` (`qb.to_list()`) | New columns can ride the payload without a typed-model change (the blocker is the `_HIT_COLUMNS` `SELECT`, not the envelope — see §5 of WHATS_LEFT). |
| **FTS-only** path is GPU-free | deferred `ratch.vllm` imports in `backend/clients.py` + `state.py` | The "no Ray, no GPU" deployment already works structurally. |

So the work is: (1) make storage object-store + refresh-aware; (2) put **Ray
Serve** behind the client `Protocol` (online) and **Ray Data** behind the engine
(offline); (3) turn `FEATURES` into an enrichment **DAG**; (4) add the dynamic
**schema** seam; (5) add a **DuckDB (`lance` extension)** OLAP module. Each is a module with
a contract — none reaches into another's internals.

---

## 1. The layers (modules) and their contracts

The whole point of low coupling: a layer depends only on the **interface** of the
layer below, never its implementation.

```
            ┌─────────────────────────────────────────────────────┐
   UI  ───▶ │  API + dynamic frontend   (renders from /api/schema) │
            └───────────────┬──────────────────────┬──────────────┘
                            │ search (ANN/FTS)      │ OLAP (SQL)
            ┌───────────────▼──────┐   ┌────────────▼─────────────┐
            │  Query module        │   │  OLAP module             │
            │  (rerank, hybrid)    │   │ (DuckDB lance extension) │
            └───────────────┬──────┘   └────────────┬─────────────┘
   Model serving            │ EmbeddingClient/Reranker (Protocol)  │
   ┌────────────────────┐   │                                      │
   │ Ray Serve + vLLM   │◀──┴──────────────┐                       │
   │ (embed/rerank/cap) │                  │ same Protocol         │
   └─────────▲──────────┘                  │                       │
             │ calls                        │                      │
   ┌─────────┴───────────────┐   ┌──────────┴──────────┐           │
   │ Enrichment / evolution  │   │ Online enrichment   │           │
   │ Ray Data (RayJob, bulk) │   │ (embed new row now) │           │
   │ walks the FEATURES DAG  │   └──────────┬──────────┘           │
   └─────────┬───────────────┘              │ add_columns          │
             └──────────────┬───────────────┴──────────────────────┘
                ┌───────────▼───────────────────────────────────┐
                │  Storage module — Lance on S3                  │
                │  (schema = live registry; handle + refresh)    │
                └───────────────────────────────────────────────┘
   Orchestration/events (§8) trigger enrichment + handle refresh out-of-band.
```

| Module | Responsibility | Interface it exposes (the contract) | Today → change |
|---|---|---|---|
| **Storage (Lance)** | hold media+vectors+FTS; be the schema registry | `open(uri) → handle`, `schema()`, `refresh()`, `add_columns()`, `scan(cols, filter)` | `state.py` opens local paths, pinned. Add **S3 URIs** (SDK `storage_options=`) + `checkout_latest()` refresh; wrap behind a `DatasetRegistry`. The SDK already backs every piece: **namespaces** (`create_namespace`/`namespace_path=`), **field-metadata roles** (`update_field_metadata`), refresh (`checkout_latest`), maintenance (`optimize`) — the work is wiring, not new format features. |
| **Model serving** | run embed/rerank/caption models | the `EmbeddingClient` / `RerankClient` / `CaptionClient` **Protocols** | impl is HTTP→vLLM today. Add a **Ray Serve** ingress (URL swap) + a **Ray-actor** impl for in-job batch. Logic above is untouched. |
| **Enrichment (data-evolution)** | create columns from columns | a `Stage{name, inputs, outputs, depends_on, compute}` descriptor + a driver that walks the DAG | `FEATURES` registry + `engine.py` exist. Add `inputs/outputs/depends_on` to `Feature`; add a **Ray Data driver** beside the CLI driver. |
| **Query** | ANN/FTS/hybrid/rerank search | `search(spec) → list[dict]`, **modes derived from column roles** | exists (`backend/search`) but **capability is column-bound**: `SearchMode` is a closed enum and each mode names a column (`_search_semantic`→`text_embedding`, etc.). Decouple from `_HIT_COLUMNS` *and* from the mode enum → generic `vector:<col>` / `fts:<col>` dispatch keyed on schema roles. |
| **OLAP** | analytical SQL over Lance (optional) | `sql(query) → arrow`, `facets(field)` | **new**, optional: the official **`lance` DuckDB extension** (`lance_vector_search`/`lance_fts`/`lance_hybrid_search`, `ATTACH … (TYPE lance)`). Concurrency is **Lance MVCC** (data stays Lance), not DuckDB's single-writer. **Retrieval already uses the LanceDB SDK** — this module is for SQL/OLAP only. |
| **Schema** | describe current columns + **capabilities** | `GET /api/schema → [{name, type, role, label}]` where `role` drives *what you can do* (search/filter/play), not just display | extend `/api/columns` to read `dataset.schema()` live + roles; both search dispatch and UI affordances derive from it. |
| **API + UI** | serve data; render generically | OpenAPI contract (build-time) + runtime `/api/schema` | generate TS client from OpenAPI; render table/detail/filters from schema. |
| **Orchestration** | trigger enrichment + refresh | events ("media added", "stage done") | **new** (§8): replaces shell/Makefile chaining. |

---

## 2. Online vs offline — one model-serving layer, two drivers

This is the distinction you flagged ("Ray Serve with vLLM is for online columns
like embeddings"). The model-serving layer is **one thing**; what differs is who
drives it and at what latency:

- **Ray Serve + vLLM = the model-serving layer** (autoscaling deployments for
  embed / rerank / caption). It serves **online columns** — low-latency, one item
  at a time:
  - **query embedding** at search time (`backend/search` → `EmbeddingClient`), and
  - **on-demand enrichment**: a new media row arrives → embed it *now* so it's
    searchable immediately, writing the `text_embedding` column for that row.
- **Ray Data (RayJob) = the offline bulk driver** for the data-evolution DAG —
  backfilling a column across the *whole* corpus (e.g. a new embedding space over
  145k rows). It can either **call the same Ray Serve endpoint** or hold its own
  **batch actors** (warm model per replica) — your call per stage.

Both paths go through the **same `EmbeddingClient` Protocol** and write via the
**same `add_columns` engine**. So "online embedding" and "offline embedding" are
two drivers over one seam — no duplicated model code, and a stage can move
between online and offline without rewriting its `compute`.

---

## 3. The key interfaces (where low coupling actually lives)

**(a) Model client — already a Protocol; just add impls.**
```python
class EmbeddingClient(Protocol):
    def embed_text(self, texts: list[str]) -> np.ndarray: ...
    def embed_image(self, jpegs: list[bytes]) -> np.ndarray: ...
# impls: VLLMEmbeddingClient (HTTP→server today) · RayServeEmbeddingClient (URL→Serve)
#        · RayActorEmbeddingClient (in-job, warm model). Callers never branch on which.
```

**(b) Enrichment stage — promote `Feature` into a DAG node.**
```python
class Stage(BaseModel):          # extends today's Feature
    name: str
    table: str
    inputs: list[str]            # columns it reads   ← NEW (makes the DAG explicit)
    outputs: list[str]           # columns it writes  ← NEW
    depends_on: list[str] = []   # stage names        ← NEW
    run: Callable[[Path, Options, Progress], int]   # unchanged
# A driver topologically sorts STAGES, runs only stages whose outputs are
# missing/NULL for the new rows (incremental), idempotently. CLI = one driver;
# Ray Data = another; an event handler = a third. The stages don't know which.
```

**(c) Storage handle — add refresh so new columns appear without restart.**
```python
class DatasetRegistry(Protocol):
    def open(self, name: str) -> LanceHandle: ...   # name, not a filesystem path
    def schema(self, name: str) -> FieldSchema: ...  # live → drives /api/schema
    def refresh(self) -> None: ...                   # checkout_latest(), cheap
# state.py builds this once (S3 uri + storage_options); an event or TTL calls refresh().
```

**(d) OLAP — a thin, separate module (no coupling to search/enrichment).**
```python
class Olap(Protocol):
    def sql(self, q: str) -> pa.Table: ...    # DuckDB lance extension; concurrency = Lance MVCC
    def facets(self, field: str) -> list[Facet]: ...
```

---

## 4. Concrete decoupling refactors (mapped to files)

1. **`DatasetRegistry`** wraps the `db / "x.lance"` path-building scattered across
   `state.py`, `ingest/ingest.py`, `features/columns.py`, `cli/*`. One module owns
   table addressing + S3 + refresh → unlocks §3 (namespacing) and the no-restart
   schema (§5). *Highest decoupling win.*
2. **Split the `FEATURES` `run` closures** (`columns.py`): they currently both
   *build the client from a URL* and *call the column fn*. Separate "describe the
   stage (inputs/outputs)" from "provide a client" from "drive it" — so a Ray
   driver reuses the stage without the CLI's client-building.
3. **Derive the search `SELECT` from the schema**, retiring `_HIT_COLUMNS`
   (`backend/search/constants.py`) — decouples search from a frozen column list.
   *And go further:* retire the closed `SearchMode` enum (`backend/search/spec.py`)
   in favour of **role-derived dispatch** — `vector:<col>` for every `embedding`
   column, `fts:<col>` for every `fts` column — so a new column is searchable with
   zero backend/frontend edits. This is the "stop knowing too much" refactor: the
   query layer knows *roles*, not column names.
4. **`/api/schema`** reads `DatasetRegistry.schema()` live (extends
   `backend/system/router.py` `/api/columns`); the frontend renders from it.
5. **OLAP module** is brand-new and stands alone — the DuckDB `lance` extension scanning the same
   Lance tables; nothing else depends on it (the UI calls it for stats/§12).
6. **Ray Serve deployment** is config: `MEDIA_EMBED_URL` → Serve ingress; the
   `backend/clients.py` factory already injects it.

None of these requires touching another module's internals — that's the test of
whether the boundary is right.

---

## 5. Sequencing (low-risk → foundational)

1. **`DatasetRegistry` + `/api/schema` + handle refresh** — the schema/storage
   seam. Unblocks dynamic UI (§5) and S3, low risk, no GPU.
2. **Derive `SELECT` from schema; generate TS client from OpenAPI** — finishes the
   dynamic-UI loop (§5).
3. **Stage DAG**: add `inputs/outputs/depends_on` to `Feature`; a topological
   driver (still CLI). Pure refactor, no Ray yet.
4. **Ray Data driver** behind the engine (offline backfill) + **Ray Serve** behind
   the client Protocol (online) — §1, on KubeRay.
5. **DuckDB `lance`-extension OLAP module** (§3, optional) → analytics/charts (§12).
6. **Events** (§8) wire enrichment triggers + handle refresh together.

The throughline: **every step adds an implementation behind an interface that
already exists or is introduced once** — so the system stays decoupled and each
layer (storage, serving, enrichment, query, OLAP, schema, UI) evolves on its own.
