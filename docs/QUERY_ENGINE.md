# Query engine — future direction (TODO, NOT now)

> A distributed, Arrow-native query engine over Lance-on-object-storage. **Later infra**
> — it does NOT block or replace the current app work. Captured so the idea is durable and
> the boundary with what we build now is explicit.

## The vision — Quack + Lance + ADBC + firnflow

| Piece | What it gives |
|---|---|
| **Quack-Cluster** (kristianaryanto/Quack-Cluster) | serverless **distributed SQL**: per-worker embedded **DuckDB** executes query fragments; **Ray Actors** fan out across workers; reads files directly from object storage (S3/GCS), **no ETL** |
| **Lance extension** | so DuckDB reads **Lance** tables (versioned, vector + FTS indices), not just Parquet/CSV |
| **ADBC / quack** (adbc-drivers/quack) | an **Arrow Database Connectivity** driver (C++) for DuckDB → **zero-copy Arrow** result transfer to clients |
| **firnflow** | a tiered **foyer** RAM/NVMe cache over object storage ("S3 cost, RAM speed") with **version-based invalidation** |

= *"distributed DuckDB-on-Lance-on-S3, Arrow-native, foyer-cached, Ray-fanned."* A **Job-1 query engine** (execution), per docs/SEARCH split.

## Job 1 (this engine) vs Job 2 (stays ours)

- **Job 1 — QUERY EXECUTION (this engine):** scan / filter / aggregate over Lance-on-S3, distributed (Ray), cached (foyer), Arrow-native (ADBC). Data infrastructure.
- **Job 2 — RETRIEVAL ORCHESTRATION (our search backend):** GPU **embed** → vector-ANN → **fuse** (RRF/hybrid) → GPU **rerank** → descriptor-driven **shape** → the read→annotate `Selection`. Application/product logic.

## What it replaces / doesn't

- **Replaces:** raw Lance scan/filter **execution** in our backend + the data/query **cache** tier. Our backend delegates execution *down*.
- **Does NOT replace:** the Job-2 orchestration — Quack-Cluster is **SQL only, no ML/vector/embeddings** (per its README). The GPU embed+rerank, Lance's native vector-ANN, descriptor modes, hit-shaping, and the `Selection` bridge stay ours. **So the search backend is not redundant — it slots on top.**
- **Open sub-question:** vector-ANN — DuckDB+lance can *scan* Lance, but Lance's native IVF/PQ index is the fast ANN path; a Quack-style SQL engine handles the structured/analytical + FTS side, while vector retrieval may stay a Lance-native call or a DuckDB UDF. Decide at spike time.

## Relation to lance-ns

- **lance-ns** = the lakehouse / **catalog** (governance: OpenFGA, OpenLineage, medallion, the catalog contract).
- **This engine** = the distributed SQL **execution** layer.
- They **compose**: the engine executes; lance-ns's catalog governs + fronts it. Building this is an **alternative to using lance-ns's built-in query execution** — the fork to decide later: reuse lance-ns's engine, or slot a Quack-style engine under its catalog.

## Cache ownership (recap, see LANCE_NS_INTEGRATION.md §6a)

| Cache | Caches | Owner |
|---|---|---|
| Object-byte / query (**foyer**) | S3 fragment bytes + query-execution results | **this engine / lakehouse** |
| **Result cache (ours)** | the GPU-orchestrated output (embed+search+fuse+rerank) — the engine can't cache what it doesn't run | **our backend** (dict → redis at replica scale) |
| Frontend | rendered responses, client state | **browser / HTTP / CDN** |

## Phased TODO (later — post-merge / decoupled)

1. **Spike:** DuckDB + lance-extension reading our Lance tables (single node), results to the backend via **ADBC** (Arrow, zero-copy). Prove SQL-over-Lance works.
2. **Distribute:** add the Quack-Cluster pattern — Ray Actors fan out scan/filter over partitions.
3. **Cache:** add the firnflow-style **foyer** RAM/NVMe tier + version-based invalidation.
4. **Vector:** decide the vector-ANN path (Lance-native call vs DuckDB UDF) alongside SQL/FTS.
5. **Governance fork:** standalone engine vs under lance-ns's catalog (OpenFGA/OpenLineage).
6. **Wire:** point the search backend's *execution* calls at the engine; keep Job-2 orchestration.

## Not now

Post-merge / decoupled infra. Current priority stays the **pre-merge provenance** (columns +
stamping + viewing) + the shipped labeling/read platform. This doc exists so the vision is
captured without derailing that.
