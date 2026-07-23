# LanceDB SDK audit — what we use, what to adopt, what to skip

*2026-07-21 · lancedb 0.34.0 · pylance 7.0.0 · audited against the lancedb Python API
reference (sync + async). Method: 3 read-only inventory agents (every `lance` /
`lancedb` / `lance_namespace` call in backend + src + scripts) + 4 adversarial probes,
one per adoption candidate, each grounded in file:line evidence.*

## The three-layer usage map (deliberate, keep it)

| Layer | API | Where | Why |
|---|---|---|---|
| **Data plane** (media bytes, annotations, atlas, graph, introspection) | raw **pylance** `lance.dataset(uri, storage_options=…)` | `lancekit/{reader,writer,introspect,descriptor}`, `media_api/*` | `_rowid`-addressed `take_blobs` (lazy seekable BlobFile), `checkout_version`, `merge_insert` builder, schema metadata — the blob/rowid capabilities the lancedb Table layer wraps more thinly than we need |
| **Search plane** | **lancedb** `connect` → `open_table`, `MatchQuery`/`PhraseQuery`, rerankers, `create_index(config=FTS()/IvfPq())` | `lancekit/registry.py`, `search_api/*`, `ratch` ingest/engine | The query-builder surface (FTS/hybrid/rerank) is exactly what the SDK is for |
| **Catalog seam** | **lance_namespace** REST client (`QueryTableRequest`, `merge_insert_into_table`, `DeleteFromTableRequest`) | `lancekit/{reader,writer}` catalog transports | The lance-ns merge contract — predicates are *strings on this wire* (constrains everything below) |

## Probe verdicts

### 1. `read_consistency_interval` / `checkout_latest` / `Session` — **SKIP** (with one local fix)

Every data read opens fresh per request (`table_dataset` → `lance.dataset`;
`handle.db.open_table` constructs a new Table at the latest manifest) — already
equivalent to strong consistency, and a request's handle is a consistent snapshot for
its whole life (load-bearing: `chunk_frame`'s `to_table` rowids must agree with the
subsequent `take_blobs`). The remembered *"500s on /api/chunk-frame after a rebuild
until restart"* was the **pre-registry** app pinning `chunks_ds` at startup (commit
`a97ce9f`) — the registry refactor already removed that mechanism. Adopting
`read_consistency_interval` would mean re-introducing cached Table handles just to have
something for the interval to refresh.

**Real residual pain (ours, not the SDK's):** the registry caches the *descriptor*
(schema/row_count/version frozen at first `get()`) forever — `DatasetHandle.refresh_descriptor`
exists with **zero callers**, and `search_api/target.py:116` acknowledges the stale race
in a comment. → **Settled (shipped):** drift-sync landed as `DatasetHandle.sync_table_info` when a per-request open
observes a newer table version than the cached `TableInfo`. `Session` (shared
index/metadata cache across opens) is the one SDK feature with genuine upside for the
open-per-request pattern — adopt only if per-request open cost ever shows in profiles.

### 2. Typed `Expr` (`col`/`lit`/`isin`) — **SKIP the builder, fix the strings**

Expr solves the wrong layer here: the writer seam's `delete(predicate: str)` **is the
lance-ns REST wire contract** (`DeleteFromTableRequest(predicate=str)`), reads use
pylance `filter=str`, and two deliberate raw-SQL pass-throughs (scope `where`, search
`raw`) can't be Expr-represented — so Expr would flatten through `to_sql()` at every
seam, adding a dialect-compat surface to a codebase that already documents a planner
bug where exact predicate shape is load-bearing (`media.py` frame_idx workaround).

The probe *did* find real string-layer defects (none currently exploitable — the
default `doc_key_pattern ^[A-Za-z0-9_-]{1,64}$` admits no quotes):

- **7 independent quoting implementations**, with `_sql_quote` existing three times
  under **two different contracts** (full-literal in `annotations/commit.py` vs
  bare-escape in `search_api/filters.py` + `voice_service.py`).
- **`validate_doc_key` conflates validation with escaping** — it returns the
  SQL-*escaped* key, and `save.py`/`tags.py` stamp that escaped value into stored row
  identity and deterministic tag ids. Latent data corruption the moment a descriptor
  uses a permissive doc-key pattern (which the function's docstring advertises).
- `ratch/modalities/av/cluster.py:73` interpolates a filesystem path into a `LIKE`
  filter with **no escaping** (never fires today — callers pass fixed names).

→ **Settled (shipped):** the shared pure-string predicate helper lives in `services/common/lancekit/predicate.py`
(`quote_literal` / `eq` / `isin` / `and_`) used by all 7 sites — works identically for
pylance `filter=`, `ds.delete`, and the catalog REST predicate; make `validate_doc_key`
validation-only (return the RAW key; escape only at SQL-render time); fix the
`cluster.py` LIKE. ~25 lines, mechanical adoption.

### 3. `fetch_blob_files` / `fetch_blobs` — **SKIP** (already have the capability)

The media plane already serves everything from Blob V2 via lazy seekable handles:
`ds.take_blobs(column, ids=[rowid])` → `seek(start)` → 1 MiB chunked
`StreamingResponse`, full HTTP Range (206/416, suffix ranges, RFC 9110 malformed-header
fallback — tested), `_rowid` addressing that survives compaction, `storage_options`
threaded so ranged S3 serving works. `fetch_blob_files` is a Table-layer wrapper over
the *same* pylance BlobFile machinery, and would need new row-id plumbing while
breaking the NotFoundError/dangling-URI guards written against `LanceDataset`. The only
real blob item is **data-side and already built**: run `ratch materialize-blobs`
(external `file://` URIs → managed) for the S3 dataset — the known re-ingest item.

### 4. Versions / tags / branches / `optimize` — **SPLIT**

- **`list_versions`/`checkout` migration: skip.** Our `ds.versions()` /
  `ds.checkout_version()` wrappers are thin and deliberate; the genuinely hand-rolled
  parts (per-unit annotation count at each version; 404 translation) are app logic no
  SDK provides.
- **Version GC: REAL gap — adopt `cleanup_old_versions`/`optimize` as maintenance.**
  The annotations table commits 1–2 versions *per Save* and nothing ever prunes any
  table the app writes (`WHATS_LEFT.md` §2 says so). `/versions` materializes the full
  manifest list before capping — slows linearly with save count. **Constraint:** the
  compare-versions audit feature is served *from* retained old versions, so GC needs a
  retention window ≥ the audit horizon. → **Settled (shipped):** extended via the `ratch maintain`
  CLI / a `maintain` make target; at merge this belongs to lance-ns (catalog owns table
  maintenance), so keep it a thin scheduled call, not app machinery.
- **Version tags: adopt WITH GC** (pylance `ds.tags` — no lancedb migration needed).
  Named review milestones ("batch-1-reviewed") are exempt from cleanup — the durable
  audit spine once intermediate saves get pruned. Explicitly distinct from
  `annotations/tags.py` **row** tags (labels ON chunks) — different concept, zero overlap.
- **Branches: skip (speculative).** No workflow wants isolated writable forks; branches
  would break the linear-version assumptions in the optimistic-concurrency handshake
  (`base_version` vs one `ds.version`), the `X-Annotations-Version` header, and the
  compare-versions diff.

## Backlog summary

Items 1–5 of the original backlog (predicate helper, drift-sync, version GC +
tags, materialize-blobs, describe_indices migration) all shipped — detail in git
history. The one live item:

- **[watch] `Session` shared cache** — only if per-request open cost profiles hot.
