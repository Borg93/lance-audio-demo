/**
 * The read→annotate bridge: a selection of media units to review, stepped through in
 * the annotator. The READ plane (atlas lasso / search) forms it — deep-linking to
 * `/annotate?keys=k1,k2,…` where each key is the descriptor key-path
 * (`doc/speech/chunk`) — and the annotator route opens it here + navigates. This is
 * the `Selection` bridge made concrete (see labeling/types.ts).
 */
import type { MediaKind, MediaUnit } from "$lib/viewer/types";

/** Build an annotator unit from a key-path (`doc/speech/chunk`) — the same path the
 *  chunk-frame + annotations endpoints take. Modality defaults to image/document
 *  (our corpus); a descriptor-driven kind is a small follow-up. */
export function unitFromKey(key: string, kind: MediaKind = "image"): MediaUnit {
  return {
    kind,
    key,
    imageUrl: `/api/chunk-frame/${key}`,
    annotationsUrl: `/api/annotations/${key}`,
  };
}

class ReviewSelection {
  units = $state<MediaUnit[]>([]);
  index = $state(0);

  get active(): MediaUnit | null {
    return this.units[this.index] ?? null;
  }
  get total(): number {
    return this.units.length;
  }

  /** Open a selection of key-paths for review (from the read plane). */
  openKeys(keys: string[]): void {
    this.units = keys.filter(Boolean).map((k) => unitFromKey(k));
    this.index = 0;
  }
  go(i: number): void {
    if (i >= 0 && i < this.units.length) this.index = i;
  }
  clear(): void {
    this.units = [];
    this.index = 0;
  }
}

/** Singleton — written by the read plane (via the route's `?keys=`), read by the
 *  annotator route + its PageNav. */
export const reviewSelection = new ReviewSelection();
