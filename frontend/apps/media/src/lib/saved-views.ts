/**
 * Saved views — named read-plane selections (a FiftyOne-class curation convenience).
 *
 * A "view" is just a named `SearchSpec` (query + mode + filters + topic + where), so a
 * reviewer can return to "the low-confidence HTR lines" or "the ambiguous speaker
 * cluster". Pure logic (no Svelte, no storage) so it's testable; the runes store
 * (saved-views.svelte.ts) wraps this over localStorage. See +page.svelte's `spec`/runSearch.
 */
import type { SearchSpec } from "$lib/api";

export interface SavedView {
  name: string;
  /** Dataset id the view belongs to ("" = the default DB) — filters are dataset-specific. */
  dataset: string;
  spec: SearchSpec;
}

/** Drop the EPHEMERAL, non-reproducible parts of a spec before saving: an uploaded image
 *  (a `File` can't serialize) and its derived query vector (`qVec` — large + tied to that
 *  upload). A saved view captures the reproducible text/filter/mode query. */
export function stripEphemeral(spec: SearchSpec): SearchSpec {
  const clean: SearchSpec = { ...spec };
  delete clean.image;
  delete clean.qVec;
  return clean;
}

/** Insert or overwrite a view by (name, dataset), newest first. */
export function upsertView(views: readonly SavedView[], view: SavedView): SavedView[] {
  const rest = views.filter((v) => !(v.name === view.name && v.dataset === view.dataset));
  return [view, ...rest];
}

/** Remove the view matching (name, dataset). */
export function removeView(
  views: readonly SavedView[],
  name: string,
  dataset: string,
): SavedView[] {
  return views.filter((v) => !(v.name === name && v.dataset === dataset));
}

/** The views belonging to a dataset ("" = default). */
export function viewsForDataset(views: readonly SavedView[], dataset: string): SavedView[] {
  return views.filter((v) => v.dataset === dataset);
}
