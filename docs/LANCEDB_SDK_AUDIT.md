# LanceDB SDK audit — what we use, what to adopt, what to skip

*2026-07-21 · lancedb 0.34.0 · pylance 7.0.0 · audited against the lancedb Python API
reference (sync + async). Method: 3 read-only inventory agents (every `lance` /
`lancedb` / `lance_namespace` call in backend + src + scripts) + 4 adversarial probes,
one per adoption candidate, each grounded in file:line evidence.*

## The three-layer usage map (deliberate, keep it)

| Layer | API | Where | Why |
|---|---|---|---|
| **Data plane** (media bytes, annotations, atlas, graph, introspection) | raw **pylance** `lance.dataset(uri, storage_options=…)` | `lancekit/{reader,writer,introspect,descriptor}`, `media_api/*` | `_rowid`-addressed `take_blobs` (lazy seekable BlobFile), `checkout_version`, `merge_insert` builder, schema metadata — the blob/rowid capabilities the lancedb Table layer wraps more thinly than we need |
| **Search plane** | **lancedb** `connect` → `open_table`, `MatchQuery`/`PhraseQuery`, rerankers, `create_index(config=FTS()/IvfPq())` | `lancekit/registry.py`, `search_api/*`, `rmedia` ingest/engine | The query-builder surface (FTS/hybrid/rerank) is exactly what the SDK is for |
| **Catalog seam** | **lance_namespace** REST client (`QueryTableRequest`, `merge_insert_into_table`, `DeleteFromTableRequest`) | `lancekit/{reader,writer}` catalog transports | The lance-ns merge contract — predicates are *strings on this wire* (constrains everything below) |

## Adopted now (this commit)

- **lancedb 0.33.0 → 0.34.0** (user-directed). 615 backend tests + full E2E green.
- **Unified index API** — all 4 legacy call sites migrated, deprecation warnings gone:
  `create_fts_index(kwargs…)` → `create_index(col, config=FTS(…))` in
  `rmedia/core/engine.py::ensure_fts_index`, `rmedia/ingest/ingest.py` (ingest +
  `reindex_fts`), `tests/test_search_api_service.py`; legacy
  `create_index(metric=…, vector_column_name=…)` → `create_index(col,
  config=IvfPq(…))` in `ensure_vector_index` (metric now `Literal["l2","cosine","dot"]`).

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
in a comment. → **P1 backlog: wire `refresh_descriptor`** when a per-request open
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
- `rmedia/modalities/av/cluster.py:73` interpolates a filesystem path into a `LIKE`
  filter with **no escaping** (never fires today — callers pass fixed names).

→ **P1 backlog: one shared pure-string predicate helper in `backend/lancekit`**
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
real blob item is **data-side and already built**: run `rmedia materialize-blobs`
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
  retention window ≥ the audit horizon. → **P2 backlog:** extend the `rmedia compact`
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

## Backlog summary (order of value)

1. **[P1] Shared predicate helper in lancekit — DONE** (`lancekit/predicate.py`:
   `quote_literal/eq/ne/isin/and_`; ten sites migrated — the sweep found three more than
   the probe's seven — `validate_doc_key` validation-only, `cluster.py` LIKE fixed).
   Migration note: for quote-containing keys under a *permissive* descriptor pattern,
   stored identity/tag-id bytes change from escaped to raw (the correct bytes); the
   default pattern admits no quotes, so shipped datasets are byte-identical.
2. **[P1] Descriptor drift-sync — DONE** (`DatasetHandle.sync_table_info`, replacing the
   dead `refresh_descriptor`): per-table re-introspection on observed version drift at
   both open seams (`table_dataset`, `resolve_target`); copy-on-write, best-effort,
   stampede-guarded.
3. **[P2] Annotations version GC — DONE** (`rmedia maintain` / `make maintain`:
   `cleanup_old_versions` with `--older-than-days` retention, tagged versions + latest
   always survive via `error_if_tagged_old_versions=False`) + **version tags — DONE**
   (`rmedia tag NAME [--version N] [--delete|--list]` — review milestones exempt from
   pruning; a pruned version is a clean 404 through `/versions` checkout).
4. **[P2] `rmedia materialize-blobs` — DONE 2026-07-21** (parity_new: media_blob 376 MB +
   thumbnails now managed; re-synced to MinIO; 65 MB blob streamed over S3 + full
   annotations write plane verified against the S3-backed backend).
5. **[P3] `describe_indices()` migration — DONE 2026-07-21** (`introspect.py`: type from
   the type_url tail — BTree/Bitmap/Inverted unchanged, vector indexes now the generic
   `Vector` instead of `IVF_PQ`; columns from `field_names`; display-only downstream,
   frontend validates an opaque string).
6. **[watch] `Session` shared cache** — only if per-request open cost profiles hot.
