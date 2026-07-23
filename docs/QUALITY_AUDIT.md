# Code-quality audit — annotation platform (2026-07-21)

> A 61-agent find→verify audit (architecture · TypeScript/Svelte-5 · Python · E2E lenses,
> judged against the house rules: engine-not-Svelte, no-god-files, thin wrappers,
> registry-driven extension, plain-Pydantic/sync-def/domain-exception Python).
> **57 findings, all verified factually real; 45 judged worth fixing.**
> This file is the durable backlog — items are line-anchored as of commit 5f30ef8.
> (Anchor paths predate the services split: `media_api/` → `services/annotator/`,
> `backend/` → `services/`, frontend `src/lib/` → `frontend/apps/{media,annotator}/src/lib/`
> + `frontend/packages/`, `e2e/` → `frontend/apps/media/e2e/`.)

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

## The backlog — prioritized

*(Swept 2026-07-21 for the merge-handoff goal: every remaining item below is
**deferred** with a one-line reason inline. Cleared items live in git history.)*

### P1 — structural — cleared in b63405f + b9a2e44 (schema single-sourced, god-files split; details in git history)

### P2 — seams + types

- **[medium/duplication]** `lancekit/writer.py:90` — LocalCatalogWriteTransport (90-108) is byte-identical to LanceTableWriter (42-59) — same three methods, same bodies — and RestCatalogWriteTransport.merge_upsert/merge_insert_only (137-151 vs 153-166) duplicate the Arrow-IPC-file serialization differing only in the when_matched_update_all flag.  — **deferred:** collapses at merge when the catalog transport becomes the only writer; unifying pre-merge would couple the two seams the split keeps apart.
- **[medium/dead-code]** `lancekit/writer.py:77` — CatalogTableWriter stores self._id = table_id but never uses it, and the class as a whole is a no-op pass-through: CatalogWriteTransport's protocol is structurally identical to TableWriter, so every method just forwards.  — **deferred:** same merge-time collapse as above; the pass-through IS the seam the lance-ns writer drops into.
- **[medium/coupling]** `media_api/annotate.py:267` — All four annotate routes (and assist.py:68) hardcode the identity arity as /{doc_id}/{speech_id}/{chunk_id} path params and the tuple (speech_id, chunk_id) at lines 293, 344, 399, 420, while identity_values/chunk_key_filter are deliberately arity-generic off the descriptor.  — **deferred:** superseded in part by the annotations/ package + lancekit/keys.py (the *logic* is arity-generic); the ROUTE shape awaits the lance-ns route contract (handoff question).
- **[medium/layering-violation]** `media_api/jobs.py:110` — _remote/_remote_status call resp.raise_for_status(), letting raw httpx.HTTPStatusError/transport errors escape the router (same pattern in assist.py:153); backend/core/handlers.py has no httpx mapping, so an upstream 4xx/5xx surfaces as an opaque 500.  — **deferred:** needs an httpx→domain-error mapping in core/handlers as its own reviewed change; upstream faults currently surface as 500 (correct class, opaque detail).
- **[medium/cohesion]** `src/lib/components/saved-views.svelte:36` — Hand-rolled dropdown (a bare `open` boolean + absolute-positioned div, 34-75) instead of the bits-ui Popover used by the four sibling popover components (filter-popover, search-settings, help-popover, status-badge).  — **deferred:** cosmetic consistency; the read-plane E2E now exercises this dropdown, so the bits-ui swap should ride a UI-focused change, not this goal.
- **[medium/dead-code]** `src/lib/viewer/PixiCanvas.svelte:47` — setContext("pixi", pixiCtx) has zero getContext consumers anywhere in src, and the `zoom`/`panX`/`panY` bindables, `colorFn`, `annotationStyle` props and `children` snippet (11-31, 144-146) are never passed by either mount site (ImageViewer, VideoViewer pass only `onready`).  — **deferred:** dead-prop pruning on the canvas is safe but touches both viewer mount sites; batch with the next viewer change.
- **[medium/type-safety]** `src/lib/viewer/annotator.svelte.ts:729` — apply() casts arbitrary LabelOp payload keys with `field as EditableField` before calling updateField — the Record<string,string> payload is never validated against the editable-field set.  — **deferred:** wants a typed EDITABLE_FIELDS guard shared with the backend contract; small but contract-adjacent (handoff).
- **[medium/layering-violation]** `src/lib/viewer/layout/AnnotatorShell.svelte:52` — onKeydown (52-122) is 70 lines of keyboard-routing policy — modifier handling, tool-arming guards (spatialTool/cvTool/canDraw), drawing-vs-review key ownership, focusable-element exceptions — living in a .svelte file.  — **deferred:** engine-not-svelte extraction (keymap module) sized M; tool-defs.ts already carries the data half.
- **[medium/defect]** `src/routes/annotate/+page.svelte:22` — The route parses window.location.search once in component-init (`if (browser) { new URLSearchParams(...) }`) instead of deriving from SvelteKit's reactive `page.url`.  — **deferred:** works because the annotate route re-mounts per navigation today; derive from page.url when the shell goes multi-unit.
- **[medium/cohesion]** `src/lib/engine/interaction/InteractionManager.ts:56` — InteractionManager accretes per-tool concerns: the CV lifecycle hardcodes one tool's name into a public callback type (onCvToolReady?: (tool: "magnetic")), owns imageSource/cvInitialized state (L62-64), and exposes brush-only accessors (brushOptions getter L133, setBrushOptions L198) alongside tools, editors, selection, alt-click cycling, and pointer plumbing.  — **deferred:** per-tool capability registry is the right shape; schedule with the next engine tool addition.
- **[medium/layering-violation]** `src/lib/labeling/review-selection.svelte.ts:8` — Directory-level import cycle: labeling/review-selection.svelte.ts imports $lib/viewer/types (MediaUnit, MediaKind) while viewer/annotator.svelte.ts imports labeling/{types,producers,jobs,history} — labeling and viewer each depend on the other.  — **deferred:** dissolves when viewer/types graduates to a shared package (labeling↔viewer both importing leaf types is the symptom, not the disease).
- **[medium/coupling]** `src/lib/viewer/layout/AnnotatorShell.svelte:26` — The shell derives viewer capability by special-casing a modality name — `const spatial = $derived(unit.kind !== 'audio')` — and the keyboard guard repeats the same inference (controller.ctx !== null checks, L78-81), contradicting registry.ts's 'adding a modality = one registry entry' contract.  — **deferred:** wants a `spatial` capability flag on the viewer registry entry; schedule with the next modality addition.
- **[medium/layering-violation]** `src/lib/viewer/layout/AnnotatorShell.svelte:52` — A ~70-line keyboard controller (onKeydown, L52-122) — modifier chords, tool hotkeys with three-clause eligibility guards, drawing-key forwarding rules, review hotkeys — lives inline in the shell .svelte instead of a plain TS keymap module.  — **deferred:** duplicate of the keymap-extraction item above (one extraction serves both).
- **[medium/coupling]** `src/lib/viewer/types.ts:41` — ViewerProps.controller is typed as the concrete AnnotatorController class rather than a narrow interface — the modality decoupling contract's seam is the entire 975-line runes facade.  — **deferred:** narrow the seam interface when a second controller implementation exists; today it would be a 40-method interface mirroring one class.

### P3 — E2E suite hygiene — **deferred as a class:** the suites are green ×N and shared
hygiene (runner shell, sleeps→polls, testids, serial-only note, fixture dir) is one
dedicated pass; not worth destabilizing mid-goal. The serial-only constraint is now
documented here: the three suites share the demo unit and MUST run serially.
(Sleep/selector tallies in the bullets below predate the 2026-07-21 additions.)

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

*(The original 57-finding line-anchored archive was removed 2026-07-23 — it is
reconstructable from git history at `5f30ef8`; the live backlog above is complete.)*
