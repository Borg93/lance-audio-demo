/**
 * Persisted column-visibility sets for the search page's tables — load/save
 * helpers kept out of +page.svelte so the merge semantics live (and read) in
 * one place.
 *
 * Two flavours:
 *   • merged ({cols, known}) — for column sets that grow over app versions.
 *     `known` records which columns existed when the user last toggled, so a
 *     column ADDED to the app later starts visible (when it's in the default
 *     set) instead of being hidden forever by a verbatim restore. The old
 *     plain-array format needed a storage-key bump per added column
 *     (v2→v3 caption, v3→v4 score) precisely because it lacked this.
 *   • plain — a verbatim string array for fixed column sets.
 *
 * All storage access is best-effort (private mode / quota → fall back to
 * defaults, never throw).
 */
import { z } from 'zod';

const SavedColsSchema = z.object({ cols: z.array(z.string()), known: z.array(z.string()) });

export function loadMergedCols(args: {
  storageKey: string;
  /** Every column key the app currently defines. */
  allKeys: string[];
  /** Columns visible for a fresh user — also what NEW columns merge from. */
  defaults: string[];
  /** Optional one-shot migration source: a legacy plain-array key. */
  legacyKey?: string;
  /** Keys to append during legacy migration (columns newer than the legacy era). */
  legacyAppend?: string[];
}): string[] {
  const { storageKey, allKeys, defaults, legacyKey, legacyAppend = [] } = args;
  try {
    const raw = localStorage.getItem(storageKey);
    if (raw) {
      const saved = SavedColsSchema.parse(JSON.parse(raw));
      const fresh = allKeys.filter((k) => !saved.known.includes(k) && defaults.includes(k));
      return [...saved.cols, ...fresh];
    }
    if (legacyKey) {
      const legacy = localStorage.getItem(legacyKey);
      if (legacy) {
        const cols = z.array(z.string()).parse(JSON.parse(legacy));
        const migrated = [...cols, ...legacyAppend.filter((k) => !cols.includes(k))];
        // Stamp the merged record NOW: without a `known` baseline a user who
        // never toggles again would re-migrate forever and miss every column
        // added after the legacy era — the exact failure this format fixes.
        persistMergedCols(storageKey, migrated, allKeys);
        return migrated;
      }
    }
  } catch {
    /* fall through to defaults */
  }
  return [...defaults];
}

export function persistMergedCols(storageKey: string, cols: string[], allKeys: string[]): void {
  try {
    localStorage.setItem(storageKey, JSON.stringify({ cols, known: allKeys }));
  } catch {
    /* best-effort */
  }
}

export function loadCols(storageKey: string, defaults: string[]): string[] {
  try {
    const raw = localStorage.getItem(storageKey);
    if (raw) return z.array(z.string()).parse(JSON.parse(raw));
  } catch {
    /* fall through to defaults */
  }
  return [...defaults];
}

export function persistCols(storageKey: string, cols: string[]): void {
  try {
    localStorage.setItem(storageKey, JSON.stringify(cols));
  } catch {
    /* best-effort */
  }
}
