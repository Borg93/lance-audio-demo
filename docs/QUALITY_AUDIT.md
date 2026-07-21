# Code-quality audit — annotation platform (2026-07-21)

> A 61-agent find→verify audit (architecture · TypeScript/Svelte-5 · Python · E2E lenses,
> judged against the house rules: engine-not-Svelte, no-god-files, thin wrappers,
> registry-driven extension, plain-Pydantic/sync-def/domain-exception Python).
> **57 findings, all verified factually real; 45 judged worth fixing.**
> This file is the durable backlog — items are line-anchored as of commit 5f30ef8.

## Verdict in one paragraph

The LAYERING is right (engine plain-TS → one runes facade → thin views; registries for
modality/producer/column extension) and the platform is E2E-proven — but two structural
debts dominate: **`AnnotatorController` (975 lines, ~14 responsibilities) absorbs every
new write-plane feature** instead of delegating transports/projection to plain TS, and
**`annotate.py` (528 lines, 7 responsibilities) does the same on the backend** with the
annotations schema **triplicated** (backend `_EMPTY_SCHEMA` / `seed_annotations.py` /
engine `schema.ts` — already drifting). A 750-line dead parallel `AnnotationStore` made
"which implementation is real?" genuinely ambiguous. The rest is disciplined-but-real:
duplicated rituals (annotations fetch ×4, commit choreography ×2, E2E helpers ×3),
stringly-typed seams, and a handful of dead exports.

## Applied immediately (commit that follows this file)

- ✅ **Deleted the dead `AnnotationStore` (711 lines) + `transport.ts`** — the runes
  controller IS that layer; `buildBatchTable` (its one used export) → `engine/store/batch.ts`.
- ✅ **Deleted 4 stale ad-hoc E2E harnesses** (annotate/debug/evidence/smoke-all.mjs —
  historical phase proofs, preserved at their proof commits in git history).
- ✅ **Deleted the aspirational `SpatialGeom`/`TemporalFacet`/`Annotation` types**
  (referenced nowhere; the modality-spanning model lives in the SCHEMA, not a parallel type).
- ✅ **One tool registry** — `viewer/tool-defs.ts` now drives the toolbar, the shell
  keymap, and cv/drawing flags (was 3 hand-maintained copies that could drift).
- ✅ **`_job_id` includes the scope CONTENT** (sorted keys / WHERE) — a same-size but
  different selection is a different job (was a real idempotency defect) + `Literal`
  types for `JobScope.level`/`JobRequest.op`; frontend `BatchJob.op: Op`.

## The backlog — prioritized

### P1 — structural (do as dedicated refactors, E2E suite is the net)

1. **Split `AnnotatorController`** (annotator.svelte.ts:111, high/god-file ×2 lenses):
   extract (a) `lib/labeling/annotations-client.ts` — the save/reload/assist HTTP
   transports + payload assembly (the pattern already exists: history.ts/jobs.ts/
   tag-writer.ts are plain-TS fetch modules); (b) a row-projection module (the pure
   `_raw`/`_field`/`_num` + AnnoRow mapping over (Table, overrides)); (c) one shared
   InsertRow factory (3 inline 16-field literals must stay in sync today). The facade
   keeps: reactive state mirroring, selection, undo/redo, LabelOp dispatch.
2. **Split `annotate.py`** (high/god-file): models+schema / wire-serving / save / tags /
   versions modules; extract the shared commit choreography (version check → merge →
   delete → re-open → emit_save → SaveResult) used verbatim by save_annotations AND
   add_tags; `_tag_rows` should derive defaults from `NewAnnotation` not a 20-key literal.
3. **One schema source of truth** (high/duplication): `_EMPTY_SCHEMA` vs
   `seed_annotations.py` vs engine `schema.ts` — generate or assert-equal in CI/test;
   they already disagree (seed lacks nothing today only because we hand-synced twice).
4. **Shared annotations-load helper** (high/duplication): the fetch→ok→version-header→
   tableFromIPC ritual is copy-pasted in ImageViewer/AudioViewer/VideoViewer/_reload.

### P2 — seams + types

- **[medium/layering] Move the dataset-access kernel out of the viewer group — DONE** —
  `validate_doc_key`/`chunk_key_filter` → `lancekit/keys.py`, `table_dataset` →
  `lancekit/registry.py`, `DatasetParam` → `backend/deps.py`; all nine importers rewired.
  `annotations/` + `assist` now import zero viewer modules — the annotator-service lift
  touches no viewer code.

- **[medium/duplication]** `lancekit/writer.py:90` — LocalCatalogWriteTransport (90-108) is byte-identical to LanceTableWriter (42-59) — same three methods, same bodies — and RestCatalogWriteTransport.merge_upsert/merge_insert_only (137-151 vs 153-166) duplicate the Arrow-IPC-file serialization differing only in the when_matched_update_all flag.
- **[medium/dead-code]** `lancekit/writer.py:77` — CatalogTableWriter stores self._id = table_id but never uses it, and the class as a whole is a no-op pass-through: CatalogWriteTransport's protocol is structurally identical to TableWriter, so every method just forwards.
- **[medium/coupling]** `media_api/annotate.py:267` — All four annotate routes (and assist.py:68) hardcode the identity arity as /{doc_id}/{speech_id}/{chunk_id} path params and the tuple (speech_id, chunk_id) at lines 293, 344, 399, 420, while identity_values/chunk_key_filter are deliberately arity-generic off the descriptor.
- **[medium/layering-violation]** `media_api/jobs.py:110` — _remote/_remote_status call resp.raise_for_status(), letting raw httpx.HTTPStatusError/transport errors escape the router (same pattern in assist.py:153); backend/core/handlers.py has no httpx mapping, so an upstream 4xx/5xx surfaces as an opaque 500.
- **[medium/cohesion]** `src/lib/components/saved-views.svelte:36` — Hand-rolled dropdown (a bare `open` boolean + absolute-positioned div, 34-75) instead of the bits-ui Popover used by the four sibling popover components (filter-popover, search-settings, help-popover, status-badge).
- **[medium/dead-code]** `src/lib/viewer/PixiCanvas.svelte:47` — setContext("pixi", pixiCtx) has zero getContext consumers anywhere in src, and the `zoom`/`panX`/`panY` bindables, `colorFn`, `annotationStyle` props and `children` snippet (11-31, 144-146) are never passed by either mount site (ImageViewer, VideoViewer pass only `onready`).
- **[medium/duplication]** `src/lib/viewer/annotator.svelte.ts:447` — Three inline 16-field InsertRow literals that must stay in sync: addTemporalSegment (447-464), the assist() prediction loop (567-585), and _buildInsert (601-621) each spell out every zeroed spatial/temporal field.
- **[medium/duplication]** `src/lib/viewer/annotator.svelte.ts:815` — save() repeats the index→id resolution mapping `id: this._raw(t, "id", index) ?? String(index)` three times (edits 815-817, geometry 819-826, temporal 827-831), and the six-line pending-state reset block appears verbatim twice (409 branch 861-868 vs success branch 872-878).
- **[medium/type-safety]** `src/lib/viewer/annotator.svelte.ts:729` — apply() casts arbitrary LabelOp payload keys with `field as EditableField` before calling updateField — the Record<string,string> payload is never validated against the editable-field set.
- **[medium/layering-violation]** `src/lib/viewer/layout/AnnotatorShell.svelte:52` — onKeydown (52-122) is 70 lines of keyboard-routing policy — modifier handling, tool-arming guards (spatialTool/cvTool/canDraw), drawing-vs-review key ownership, focusable-element exceptions — living in a .svelte file.
- **[medium/defect]** `src/routes/annotate/+page.svelte:22` — The route parses window.location.search once in component-init (`if (browser) { new URLSearchParams(...) }`) instead of deriving from SvelteKit's reactive `page.url`.
- **[medium/cohesion]** `src/lib/engine/interaction/InteractionManager.ts:56` — InteractionManager accretes per-tool concerns: the CV lifecycle hardcodes one tool's name into a public callback type (onCvToolReady?: (tool: "magnetic")), owns imageSource/cvInitialized state (L62-64), and exposes brush-only accessors (brushOptions getter L133, setBrushOptions L198) alongside tools, editors, selection, alt-click cycling, and pointer plumbing.
- **[medium/layering-violation]** `src/lib/labeling/review-selection.svelte.ts:8` — Directory-level import cycle: labeling/review-selection.svelte.ts imports $lib/viewer/types (MediaUnit, MediaKind) while viewer/annotator.svelte.ts imports labeling/{types,producers,jobs,history} — labeling and viewer each depend on the other.
- **[medium/coupling]** `src/lib/viewer/annotator.svelte.ts:555` — The assist endpoint is derived by string surgery on the save URL — `url.replace("/api/annotations/", "/api/assist/")` — coupling the controller to exact backend route naming.
- **[medium/coupling]** `src/lib/viewer/layout/AnnotatorShell.svelte:26` — The shell derives viewer capability by special-casing a modality name — `const spatial = $derived(unit.kind !== 'audio')` — and the keyboard guard repeats the same inference (controller.ctx !== null checks, L78-81), contradicting registry.ts's 'adding a modality = one registry entry' contract.
- **[medium/layering-violation]** `src/lib/viewer/layout/AnnotatorShell.svelte:52` — A ~70-line keyboard controller (onKeydown, L52-122) — modifier chords, tool hotkeys with three-clause eligibility guards, drawing-key forwarding rules, review hotkeys — lives inline in the shell .svelte instead of a plain TS keymap module.
- **[medium/coupling]** `src/lib/viewer/types.ts:41` — ViewerProps.controller is typed as the concrete AnnotatorController class rather than a narrow interface — the modality decoupling contract's seam is the entire 975-line runes facade.

### P3 — E2E suite hygiene

- **[medium/duplication]** `e2e/annotator.e2e.mjs:210` — The suite-runner shell — newPage + pageErrors collector, try/catch(crashed)/finally { browser.close, re-seed with warning, finish } — is repeated near-verbatim in all three suites (annotator 24-27 + 210-224, temporal 19-22 + 179-193, read-plane 14-17 + 74-88), ~20 lines x3, plus the identical closing 'no page-level JS errors' assertion x3.
- **[medium/defect]** `e2e/annotator.e2e.mjs:75` — 26 fixed page.waitForTimeout sleeps remain across the three suites (annotator 7, temporal 9, read-plane 6, plus tool()'s 200ms) — ~15s of unconditional dead time per run — even though the suites already built the right pattern (countAbove polling) for the count case.
- **[medium/coupling]** `e2e/lib.mjs:14` — All three suites mutate the SAME demo unit (shared KEY, lib.mjs:14) and the same annotations table via seed(); temporal.e2e.mjs even re-seeds mid-run (line 121). Nothing prevents or documents that the suites must run serially — parallel execution (e.g. a future turbo/CI matrix, or someone running two suites in two terminals) would interleave seeds and corrupt every count assertion.
- **[medium/coupling]** `e2e/read-plane.e2e.mjs:34` — Read-plane SavedViews assertions are entirely text/attribute-coupled — button:has-text("Views") (34), input[placeholder*="Save current view"] (38), li hasText 'e2e-view' (45,52), button[title="Delete view"] (52) — because saved-views.svelte has zero data-testid attributes; annotator suites also lean on '[title="Annotation count"]' (annotator:30, temporal:25), getByText('Back to list') (annotator:124), and the structural 'div:has(> div > wave), div.bg-card:has(canvas)' (temporal:57).
- **[medium/duplication]** `e2e/temporal.e2e.mjs:24` — count()/countAbove() are copy-pasted verbatim from annotator.e2e.mjs (lines 29-43) into temporal.e2e.mjs (lines 24-36), ~15 lines duplicated including the fragile '[title="Annotation count"]' selector.
- **[medium/coupling]** `package.json:13` — test:e2e is a fail-fast &&-chain: a single FAIL exit(1) in annotator.e2e.mjs skips temporal and read-plane entirely, so one broken tool hides the state of the other two planes.
- **[low/cohesion]** `e2e/read-plane.e2e.mjs:19` — All three suites are 100% happy-path; cheap negative-path checks are missing: a bogus key (/annotate?keys=nonexistent) rendering an error status instead of a crash, Escape cancelling an in-progress shape (count unchanged), and the assist bar surfacing a failure when /api/assist errors.
- **[low/duplication]** `e2e/temporal.e2e.mjs:97` — The save-POST idiom (waitForResponse('/api/annotations/' + method POST).then(r=>r.ok()).catch(false) then Control+s) appears three times: annotator.e2e.mjs 194-198, temporal.e2e.mjs 97-101 and 156-160.
- **[low/coupling]** `e2e/temporal.e2e.mjs:14` — Test fixtures live in the production static dir (static/e2e/tone.wav 353KB + clip.mp4 136KB), so ~490KB of test media is copied verbatim into every vite production build and served at /e2e/* in deployed apps.
- **[low/python-style]** `package.json:16` — format/format:check run 'oxfmt src' only, so e2e/ is never formatted — visible drift already exists: annotator.e2e.mjs's suite() body dedents to column 0 from line 84 to 206 while still inside the function, making the file read as if the suite ends at line 83.

### Noted, not worth fixing (12 items)

Taste-level or working-idiom findings the verify pass judged not worth the churn —
kept in the workflow journal (wf_9e7f0ca2-1ee) for reference.

## Full finding list (all 57, line-anchored at 5f30ef8)

- **[high/duplication]** `media_api/annotate.py:146` — The annotation schema has three hand-maintained sources of truth that already disagree: _EMPTY_SCHEMA here, SCHEMA in scripts/seed_annotations.py:23, and the engine's frontend/src/lib/engine/schema.ts — and the comment at line 144 claims 'Kept in one place', which is false.
- **[high/god-file]** `media_api/annotate.py:1` — annotate.py (528 lines) has accreted seven responsibilities: 9 Pydantic request/response models, the schema contract (_EMPTY_SCHEMA), Arrow-IPC wire serving (GET), version history/time-travel (GET versions + _iso + _checkout), the save flush (POST), the tag batch (POST /tags + tag_id + _tag_rows), and utilities (_sql_quote, _ipc_stream, row builders).
- **[high/duplication]** `src/lib/viewer/ImageViewer.svelte:15` — The annotations fetch/decode ritual — fetch(annotationsUrl) → res.ok guard → Number(res.headers.get('X-Annotations-Version')) → tableFromIPC(new Uint8Array(await res.arrayBuffer())) — is copy-pasted four times: ImageViewer 15-18, VideoViewer 40-43, AudioViewer 42-46, and annotator.svelte.ts _reload 891-894.
- **[high/god-file]** `src/lib/viewer/annotator.svelte.ts:111` — AnnotatorController is a 975-line class with ~9 responsibilities: engine mirroring, selection, review-queue ordering, undo/redo, LabelOp dispatch, temporal edits, layer grouping, AI-assist HTTP, and Save/reload HTTP.
- **[high/type-safety]** `src/lib/viewer/annotator.svelte.ts:232` — `(r as unknown as Record<string, string>)[col]` in the groups $derived defeats the compiler check that would have caught a real mismatch: STRING_FIELD_CANDIDATES (line 101) offers "reviewer" as a group-by column, but AnnoRow (41-54) has no `reviewer` field.
- **[high/coupling]** `src/lib/engine/interaction/InteractionManager.ts:72` — Adding a 10th tool is not additive: the tool set is hardcoded as 8 repeated construct-wire-register blocks in the constructor (L72-115), plus the Tool string union (pixi/types.ts:22), TOOL_KEYS in AnnotatorShell.svelte:40, TOOLS in AnnotatorToolbar.svelte:45, and per-tool special cases in setTool (magnetic, L166) — at least 5 files per new tool.
- **[high/dead-code]** `src/lib/engine/store/AnnotationStore.ts:192` — AnnotationStore (711 lines) and its AnnotationTransport seam (store/transport.ts, 44 lines) are exported from the engine's public API (engine/index.ts:35) but never instantiated anywhere; only buildBatchTable is used. Comments reference binding files that don't exist (src/lib/stores/annotations.svelte.ts, httpAnnotationTransport.ts).
- **[high/god-file]** `src/lib/viewer/annotator.svelte.ts:111` — AnnotatorController is a 975-line class with ~14 responsibilities: engine attach/mirroring, selection, tool+brush state, layer grouping, Arrow row projection (_raw/_field/_num), undo/redo, insert/delete/geometry/temporal edit queues, optimistic Arrow append, save/reload HTTP, assist HTTP, LabelOp dispatch, review-queue navigation, zoom passthrough, and hex color utils.
- **[medium/duplication]** `lancekit/writer.py:90` — LocalCatalogWriteTransport (90-108) is byte-identical to LanceTableWriter (42-59) — same three methods, same bodies — and RestCatalogWriteTransport.merge_upsert/merge_insert_only (137-151 vs 153-166) duplicate the Arrow-IPC-file serialization differing only in the when_matched_update_all flag.
- **[medium/dead-code]** `lancekit/writer.py:77` — CatalogTableWriter stores self._id = table_id but never uses it, and the class as a whole is a no-op pass-through: CatalogWriteTransport's protocol is structurally identical to TableWriter, so every method just forwards.
- **[medium/duplication]** `media_api/annotate.py:480` — save_annotations and add_tags duplicate the entire commit choreography verbatim: the base_version ConflictError check (394-397 vs 480-483, identical f-string), delete-by-id predicate building (434-436 vs 503-505), the touched==0 early return, and the reopen-fresh + emit_save + logger + SaveResult tail (438-459 vs 507-522).
- **[medium/duplication]** `media_api/annotate.py:242` — _tag_rows hand-writes a 20-key literal dict of annotation-row defaults (x=0.0, status='accepted', source='human', mask='', ...) that duplicates NewAnnotation's field defaults a hundred lines above.
- **[medium/coupling]** `media_api/annotate.py:267` — All four annotate routes (and assist.py:68) hardcode the identity arity as /{doc_id}/{speech_id}/{chunk_id} path params and the tuple (speech_id, chunk_id) at lines 293, 344, 399, 420, while identity_values/chunk_key_filter are deliberately arity-generic off the descriptor.
- **[medium/type-safety]** `media_api/jobs.py:38` — Closed vocabularies are typed as bare str with the values in trailing comments: JobScope.level ('chunks' | 'scope' | 'corpus'), JobRequest.op ('predict' | 'propagate' | 'judge'), JobResult.status and backend (lines 58-59).
- **[medium/defect]** `media_api/jobs.py:71` — _job_id's docstring promises 'a re-submit of the same op+scope+exemplars is idempotent' but the hash basis uses only _scope_size (a count, -1 for scope/corpus) — scope.keys, scope.where, and prompt are all omitted, so two different selections of equal size (or any two WHERE-scoped jobs) collide onto the same job_id.
- **[medium/layering-violation]** `media_api/jobs.py:110` — _remote/_remote_status call resp.raise_for_status(), letting raw httpx.HTTPStatusError/transport errors escape the router (same pattern in assist.py:153); backend/core/handlers.py has no httpx mapping, so an upstream 4xx/5xx surfaces as an opaque 500.
- **[medium/duplication]** `e2e/annotator.e2e.mjs:210` — The suite-runner shell — newPage + pageErrors collector, try/catch(crashed)/finally { browser.close, re-seed with warning, finish } — is repeated near-verbatim in all three suites (annotator 24-27 + 210-224, temporal 19-22 + 179-193, read-plane 14-17 + 74-88), ~20 lines x3, plus the identical closing 'no page-level JS errors' assertion x3.
- **[medium/defect]** `e2e/annotator.e2e.mjs:75` — 26 fixed page.waitForTimeout sleeps remain across the three suites (annotator 7, temporal 9, read-plane 6, plus tool()'s 200ms) — ~15s of unconditional dead time per run — even though the suites already built the right pattern (countAbove polling) for the count case.
- **[medium/dead-code]** `e2e/debug.mjs:1` — Four pre-lib.mjs ad-hoc harnesses live beside the committed suites — annotate.mjs, debug.mjs, evidence.mjs, smoke-all.mjs — unreferenced by any package.json script, undocumented (TESTING.md says 'three suites'), each re-declaring its own chromium launch args (some divergent: debug.mjs uses swiftshader, annotate.mjs omits --ignore-gpu-blocklist) and a positional-argv BASE/CHROME convention lib.mjs replaced with env vars.
- **[medium/coupling]** `e2e/lib.mjs:14` — All three suites mutate the SAME demo unit (shared KEY, lib.mjs:14) and the same annotations table via seed(); temporal.e2e.mjs even re-seeds mid-run (line 121). Nothing prevents or documents that the suites must run serially — parallel execution (e.g. a future turbo/CI matrix, or someone running two suites in two terminals) would interleave seeds and corrupt every count assertion.
- **[medium/coupling]** `e2e/read-plane.e2e.mjs:34` — Read-plane SavedViews assertions are entirely text/attribute-coupled — button:has-text("Views") (34), input[placeholder*="Save current view"] (38), li hasText 'e2e-view' (45,52), button[title="Delete view"] (52) — because saved-views.svelte has zero data-testid attributes; annotator suites also lean on '[title="Annotation count"]' (annotator:30, temporal:25), getByText('Back to list') (annotator:124), and the structural 'div:has(> div > wave), div.bg-card:has(canvas)' (temporal:57).
- **[medium/duplication]** `e2e/temporal.e2e.mjs:24` — count()/countAbove() are copy-pasted verbatim from annotator.e2e.mjs (lines 29-43) into temporal.e2e.mjs (lines 24-36), ~15 lines duplicated including the fragile '[title="Annotation count"]' selector.
- **[medium/coupling]** `package.json:13` — test:e2e is a fail-fast &&-chain: a single FAIL exit(1) in annotator.e2e.mjs skips temporal and read-plane entirely, so one broken tool hides the state of the other two planes.
- **[medium/cohesion]** `src/lib/components/saved-views.svelte:36` — Hand-rolled dropdown (a bare `open` boolean + absolute-positioned div, 34-75) instead of the bits-ui Popover used by the four sibling popover components (filter-popover, search-settings, help-popover, status-badge).
- **[medium/dead-code]** `src/lib/viewer/PixiCanvas.svelte:47` — setContext("pixi", pixiCtx) has zero getContext consumers anywhere in src, and the `zoom`/`panX`/`panY` bindables, `colorFn`, `annotationStyle` props and `children` snippet (11-31, 144-146) are never passed by either mount site (ImageViewer, VideoViewer pass only `onready`).
- **[medium/duplication]** `src/lib/viewer/annotator.svelte.ts:447` — Three inline 16-field InsertRow literals that must stay in sync: addTemporalSegment (447-464), the assist() prediction loop (567-585), and _buildInsert (601-621) each spell out every zeroed spatial/temporal field.
- **[medium/duplication]** `src/lib/viewer/annotator.svelte.ts:815` — save() repeats the index→id resolution mapping `id: this._raw(t, "id", index) ?? String(index)` three times (edits 815-817, geometry 819-826, temporal 827-831), and the six-line pending-state reset block appears verbatim twice (409 branch 861-868 vs success branch 872-878).
- **[medium/type-safety]** `src/lib/viewer/annotator.svelte.ts:729` — apply() casts arbitrary LabelOp payload keys with `field as EditableField` before calling updateField — the Record<string,string> payload is never validated against the editable-field set.
- **[medium/duplication]** `src/lib/viewer/layout/AnnotatorShell.svelte:40` — The tool↔hotkey table exists twice: TOOL_KEYS in AnnotatorShell (40-51) and the `key` fields of TOOLS in AnnotatorToolbar (45-56) — two hand-maintained parallel lists.
- **[medium/layering-violation]** `src/lib/viewer/layout/AnnotatorShell.svelte:52` — onKeydown (52-122) is 70 lines of keyboard-routing policy — modifier handling, tool-arming guards (spatialTool/cvTool/canDraw), drawing-vs-review key ownership, focusable-element exceptions — living in a .svelte file.
- **[medium/defect]** `src/routes/annotate/+page.svelte:22` — The route parses window.location.search once in component-init (`if (browser) { new URLSearchParams(...) }`) instead of deriving from SvelteKit's reactive `page.url`.
- **[medium/cohesion]** `src/lib/engine/interaction/InteractionManager.ts:56` — InteractionManager accretes per-tool concerns: the CV lifecycle hardcodes one tool's name into a public callback type (onCvToolReady?: (tool: "magnetic")), owns imageSource/cvInitialized state (L62-64), and exposes brush-only accessors (brushOptions getter L133, setBrushOptions L198) alongside tools, editors, selection, alt-click cycling, and pointer plumbing.
- **[medium/layering-violation]** `src/lib/labeling/review-selection.svelte.ts:8` — Directory-level import cycle: labeling/review-selection.svelte.ts imports $lib/viewer/types (MediaUnit, MediaKind) while viewer/annotator.svelte.ts imports labeling/{types,producers,jobs,history} — labeling and viewer each depend on the other.
- **[medium/duplication]** `src/lib/viewer/AudioViewer.svelte:42` — The annotations-load protocol (fetch → check res.ok → read X-Annotations-Version header → tableFromIPC(new Uint8Array(...))) is copy-pasted four times: ImageViewer.svelte:15-18, VideoViewer.svelte:40-43, AudioViewer.svelte:42-46, and annotator.svelte.ts _reload L891-894.
- **[medium/coupling]** `src/lib/viewer/annotator.svelte.ts:555` — The assist endpoint is derived by string surgery on the save URL — `url.replace("/api/annotations/", "/api/assist/")` — coupling the controller to exact backend route naming.
- **[medium/coupling]** `src/lib/viewer/layout/AnnotatorShell.svelte:26` — The shell derives viewer capability by special-casing a modality name — `const spatial = $derived(unit.kind !== 'audio')` — and the keyboard guard repeats the same inference (controller.ctx !== null checks, L78-81), contradicting registry.ts's 'adding a modality = one registry entry' contract.
- **[medium/layering-violation]** `src/lib/viewer/layout/AnnotatorShell.svelte:52` — A ~70-line keyboard controller (onKeydown, L52-122) — modifier chords, tool hotkeys with three-clause eligibility guards, drawing-key forwarding rules, review hotkeys — lives inline in the shell .svelte instead of a plain TS keymap module.
- **[medium/coupling]** `src/lib/viewer/types.ts:41` — ViewerProps.controller is typed as the concrete AnnotatorController class rather than a narrow interface — the modality decoupling contract's seam is the entire 975-line runes facade.
- **[low/python-style]** `lancekit/writer.py:138` — pyarrow is imported at TYPE_CHECKING only, forcing runtime-local `import pyarrow as pa_` (aliased, twice) inside RestCatalogWriteTransport methods.
- **[low/python-style]** `media_api/annotate.py:354` — _iso(ts: object) duck-types via getattr(ts, 'isoformat', None) + callable() instead of narrowing with isinstance(ts, datetime).
- **[low/defect]** `media_api/annotate.py:348` — annotation_versions does an N+1 time-travel scan: one ds.checkout_version + filtered to_table per listed version (up to limit=200 sequential full checkouts) inside a sync route.
- **[low/coupling]** `media_api/assist.py:148` — _remote hand-builds the cross-router URL string f"/api/chunk-frame/{doc_id}/{speech_id}/{chunk_id}" — a stringly-typed dependency on media.py's route template (media.py:243).
- **[low/cohesion]** `e2e/read-plane.e2e.mjs:19` — All three suites are 100% happy-path; cheap negative-path checks are missing: a bogus key (/annotate?keys=nonexistent) rendering an error status instead of a crash, Escape cancelling an in-progress shape (count unchanged), and the assist bar surfacing a failure when /api/assist errors.
- **[low/duplication]** `e2e/temporal.e2e.mjs:97` — The save-POST idiom (waitForResponse('/api/annotations/' + method POST).then(r=>r.ok()).catch(false) then Control+s) appears three times: annotator.e2e.mjs 194-198, temporal.e2e.mjs 97-101 and 156-160.
- **[low/coupling]** `e2e/temporal.e2e.mjs:14` — Test fixtures live in the production static dir (static/e2e/tone.wav 353KB + clip.mp4 136KB), so ~490KB of test media is copied verbatim into every vite production build and served at /e2e/* in deployed apps.
- **[low/python-style]** `package.json:16` — format/format:check run 'oxfmt src' only, so e2e/ is never formatted — visible drift already exists: annotator.e2e.mjs's suite() body dedents to column 0 from line 84 to 206 while still inside the function, making the file read as if the suite ends at line 83.
- **[low/type-safety]** `src/lib/labeling/jobs.ts:21` — BatchJob.op is typed `string` with the real contract relegated to a comment (`// predict | propagate | judge`) even though the `Op` union already exists in labeling/types.ts and is imported-adjacent.
- **[low/dead-code]** `src/lib/labeling/producers.ts:119` — interactiveProducers() and batchProducers() (119-124) are exported but unreferenced anywhere in src; likewise jobStatus() in labeling/jobs.ts:55.
- **[low/type-safety]** `src/lib/saved-views.svelte.ts:24` — load() guards only Array.isArray before `parsed as SavedView[]` — element shape (name/dataset/spec) is unvalidated localStorage input.
- **[low/runes-misuse]** `src/lib/viewer/AudioViewer.svelte:38` — One-shot mount work (fetch annotations → controller.attachData) runs in a $effect with a fire-and-forget async IIFE, no cleanup/AbortController, relying on the shell's {#key} remount for once-per-unit semantics.
- **[low/dead-code]** `src/lib/viewer/annotator.svelte.ts:504` — propagate() is a public controller method with no caller in any component — the INSID3 few-shot entry point exists only as API surface.
- **[low/duplication]** `src/lib/viewer/annotator.svelte.ts:524` — _appendInsert's `if (t && arrow)` and `else if (t)` branches (522-536) duplicate the concat/buildBatchTable/table-assign/count/_reapplyOverrides sequence, differing only in the two arrow.load/sync calls.
- **[low/duplication]** `src/lib/viewer/annotator.svelte.ts:669` — _queuePos() (669-672) re-implements the position lookup that the queuePos $derived (260-265) already computes — two definitions of 'where is the selection in the review queue'.
- **[low/duplication]** `src/lib/viewer/annotator.svelte.ts:104` — numToHex/hexToNum re-implement engine/utils/color.ts colorToHex/hexToColor with subtly different behavior (the annotator pair masks with & 0xffffff and strips an optional '#'; the engine pair does neither).
- **[low/naming]** `src/lib/viewer/annotator.svelte.ts:557` — The `saving` flag is overloaded to mean both 'Save in flight' and 'assist request in flight' (assist() sets it at 557), and `saveError` doubles as the assist error channel.
- **[low/duplication]** `src/lib/viewer/annotator.svelte.ts:104` — numToHex/hexToNum re-implement engine utils/color.ts colorToHex/hexToColor with subtly different semantics (masking to 0xffffff, padStart, optional '#'), in a file that already imports from $lib/engine.
- **[low/dead-code]** `src/lib/viewer/types.ts:47` — SpatialGeom, TemporalFacet, and the composed Annotation type (L47-82) are referenced nowhere; the shapes actually flowing through the system are AnnoRow + InsertRow (controller) and CommitShape (engine), with a third shape_type vocabulary mapped ad hoc ('rect' → 'rectangle' at annotator.svelte.ts:604).
