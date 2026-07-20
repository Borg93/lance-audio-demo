/**
 * The workflow's tag store — tags keyed by chunk identity (`hitKey`), so a chunk
 * can be tagged inline anywhere it appears (the results list) AND in bulk by a
 * Tagger node, with the tags persisting alongside the graph and flowing into
 * Export. A small focused runes class composed into the graph singleton.
 */
import type { Hit } from "$lib/api";
import { hitKey } from "$lib/utils";

/** Anything carrying the dataset's identity key fields — `hitKey` reads them
 *  through the active DatasetView, so a bare Row is enough. */
type ChunkRef = Hit;

export class WorkflowTags {
  /** tags keyed by `hitKey` (the dataset's identity key). */
  byChunk = $state<Record<string, string[]>>({});

  /** Current tags for a chunk (empty array when untagged). */
  forHit(hit: ChunkRef): string[] {
    return this.byChunk[hitKey(hit)] ?? [];
  }

  /** Toggle one tag on one chunk (inline tagging from a results row). */
  toggle(hit: ChunkRef, tag: string): void {
    const t = tag.trim();
    if (!t) return;
    const k = hitKey(hit);
    const cur = this.byChunk[k] ?? [];
    const next = cur.includes(t) ? cur.filter((x) => x !== t) : [...cur, t];
    this.byChunk = { ...this.byChunk, [k]: next };
  }

  /** Stamp `tags` (union, add-only) onto every chunk — the Tagger node's bulk
   *  write and the inline add path. */
  addTo(hits: readonly ChunkRef[], tags: readonly string[]): void {
    const add = tags.map((t) => t.trim()).filter(Boolean);
    if (!add.length || !hits.length) return;
    const next = { ...this.byChunk };
    for (const h of hits) {
      const k = hitKey(h);
      const cur = next[k] ?? [];
      next[k] = [...cur, ...add.filter((t) => !cur.includes(t))];
    }
    this.byChunk = next;
  }

  /** Hits with their current tags attached from the store — used at export time
   *  so a `tags` column reflects both inline and Tagger-node tags. */
  withTags(hits: readonly Hit[]): Hit[] {
    return hits.map((h) => ({ ...h, tags: this.forHit(h) }));
  }

  /** Replace the whole store (rehydrate from persistence). */
  hydrate(byChunk: Record<string, string[]>): void {
    this.byChunk = byChunk;
  }

  /** A plain serialisable snapshot of the store (for persistence). */
  snapshot(): Record<string, string[]> {
    return { ...this.byChunk };
  }

  /** Clear all tags (full graph reset). */
  reset(): void {
    this.byChunk = {};
  }
}
