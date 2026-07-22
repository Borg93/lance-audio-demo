# OpenLineage emission for ratch stages — design (DRAFT)

Status: **first cut, opt-in, not wired into the driver.** `src/ratch/lineage.py`
builds the pieces; nothing calls it yet. This doc is the plan for wiring it in.

## Why

lance-ns's lakehouse is OpenLineage-native (spec **2-0-2**): every medallion hop
emits a `RunEvent`, consumed into an Apache-AGE provenance graph (see
`LANCE_NS_INTEGRATION.md §5a`). When our stages merge in as derivers, their runs
should appear in that graph — chunks → frames → embeddings → atlas as a queryable
`DERIVED_FROM` DAG with field-to-field column lineage.

## The contract we must honour (verified against lance-ns source)

A deriver **does not emit**. The mover harness does:

```
transform_stage(from_uri, to_uri, stage) ──▶ WriteResult
    { version, row_count, size_bytes, fields, column_map }
        │
        ▼
build_run_event(operation, inputs, output, version, row_count, size_bytes,
                schema_fields=fields, column_map=column_map, …) ──▶ RunEvent (JSON)
        │
        ▼
outbox.publish_lineage_with_outbox(...)   # Dapr CloudEvent on the lineage topic
```

- `services/common/openlineage.py` pins the spec constants (`SCHEMA_URL`,
  `run_id_for` = deterministic `uuid5`, the facet `_schemaURL`s, `schema_facet`,
  `column_lineage_facet`, `ErrorMessageRunFacet`). One module so no emitter drifts.
- `column_map: list[(out_field, in_field, subtype)]` — carried columns are
  `IDENTITY`, derived artifacts `TRANSFORMATION`. This becomes the `columnLineage`
  facet. On the in-process path lance-ns **declares** it; on the Ray path it
  **reconstructs** it from the two on-disk schemas (`measure_stage`). The two must
  agree.
- `run_id_for(seed)` is deterministic so an at-least-once redelivery MERGEs onto
  one `:Run` instead of duplicating it.

## What is genuinely OURS to produce

Two things, both pure functions the harness needs from us:

1. **`column_map(stage)`** — our `Stage` model already carries `read_columns` /
   `key_columns` / `output_columns` / `blob_column`, which map 1:1 onto the edges:
   - read/key columns carried → `(col, col, IDENTITY)`
   - each output column → `(out, primary_input, TRANSFORMATION)` where
     `primary_input` = the blob column (BLOB/APPEND) or the first read column (SCAN)
   - an APPEND_ROWS key minted downstream (e.g. `frame_idx`) → `TRANSFORMATION`
2. **`WriteResult`** via `measure_stage(uri)` — version, exact rows, on-disk bytes,
   and blob/vector-aware `facet_fields(schema)`. Byte-for-byte the shape lance-ns's
   `measure` returns.

`src/ratch/lineage.py` implements both, mirroring their field names so their
`build_run_event` consumes our `WriteResult` unchanged.

## Two modes, one seam

`emit_stage_lineage(..., builder=?, sink=?)`:

| Mode | `builder` | `sink` | Use |
|---|---|---|---|
| **Merged** | lance-ns `medallion.schemas.events.build_run_event` | their Dapr outbox publish | production; they own the wire event + transport |
| **Standalone** | `None` → our `build_run_event` mirror | a `LineageSink` (file/stdout/HTTP) | pre-merge Ray/CLI runs that want lineage now |

The mirror (`build_run_event` in `lineage.py`) is deliberately minimal and pinned
to the same constants, so a standalone event and a merged event describe the same
run identically. **Do not extend the mirror past their facets** — at merge, pass
their builder and delete ours.

## Wiring plan (not yet done)

1. **Emit point.** After a stage's Lance write commits (in `core/engine.py` /
   `core/driver.py`, once per stage run), call `emit_stage_lineage(...)`. Blocking
   Lance IO → already off the hot path in the driver.
2. **Config gate.** `RATCH_LINEAGE=off|stdout|dir:<path>|http:<url>` (default
   `off`) selects the sink; `off` skips measure+build entirely (zero overhead),
   exactly like the search cache's `MEDIA_SEARCH_CACHE_SIZE=0`.
3. **Namespaces.** Standalone: `job_namespace="ratch"`, dataset namespaces
   `bronze/silver/gold`. Merged: the harness supplies its real namespaces.
4. **START/COMPLETE/FAIL.** Draft emits COMPLETE (and FAIL with an
   `errorMessage` facet). A START before the compute is a small addition when the
   emit point moves to wrap the whole stage run.

## Open questions

- **event_time** is a parameter (no wall-clock baked in) so runs stay reproducible
  in tests; the driver passes `datetime.now(UTC)`. Confirm lance-ns's expectation.
- **inputs for APPEND_ROWS.** `extract_frames` reads `chunks` but its parent for
  lineage is arguably `documents` (media) too. The draft names the single declared
  source table; multi-input edges are a follow-on.
- **column_map for fan-out.** The `__{name}` sibling-table fan-out (model 2, the
  proven patch) needs its child tables named as distinct outputs — one RunEvent per
  sibling, or one with multiple outputs. Decide with the harness.
