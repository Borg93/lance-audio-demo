/**
 * Shared bidirectional cross-filter state for the atlas ⇄ search workspace.
 *
 * One singleton store, two independent Sets over the SAME point universe (the
 * loaded atlas points), keyed by point INDEX (not the `doc|speech|chunk`
 * string — index Sets avoid stringify churn on the hot recolour path):
 *
 *   • `filteredIds` — the SEARCH + facet result set projected to point indices
 *     (`null` ⇒ no active search; the whole corpus is in scope).
 *   • `selectedIds` — what the user picked ON THE MAP (lasso / box / cluster /
 *     legend / click).
 *
 * The map recolour reads BOTH Sets per point: a search miss (`filteredIds`)
 * ghosts a point hardest, a selection miss (`selectedIds`) dims it less — so a
 * lasso AFTER a search shows the hits inside the lasso, and a search AFTER a
 * lasso re-highlights within the selection — bidirectional lock-step.
 *
 * Renderer-agnostic: holds plain Sets + scalars, no EmbeddingView types. The
 * map component derives its per-point RGBA alpha from the two Sets; the page
 * feeds `setFilteredFromHits` and reads `selectedIds`.
 */

import type { Hit, AtlasSpace } from '$lib/api';
import { hitKey } from '$lib/utils';

/** Colour channel for the scatter (legend + per-point recolour). `topic` =
 *  chunk broad topic (topic_l2); `doc_topic` = per-video topic. */
export type ColorBy = 'cluster' | 'language' | 'topic' | 'doc_topic' | 'doc' | 'none';

/** Build a `'doc|speech|chunk' → point index` map once per loaded space. */
export function buildKeyIndex(
    docs: readonly string[],
    doc: readonly number[],
    speech_id: readonly number[],
    chunk_id: readonly number[],
): Map<string, number> {
    const map = new Map<string, number>();
    for (let i = 0; i < doc.length; i++) {
        const dc = doc[i];
        const d = dc !== undefined ? docs[dc] : undefined;
        const s = speech_id[i];
        const c = chunk_id[i];
        if (d === undefined || s === undefined || c === undefined) continue;
        map.set(`${d}|${s}|${c}`, i);
    }
    return map;
}

// `hitKey` lives in `$lib/utils` (single source); re-exported here so atlas
// consumers keep importing it alongside `buildKeyIndex`. Its `doc|speech|chunk`
// shape must stay in lock-step with `buildKeyIndex` above.
export { hitKey };

class CrossFilter {
    /** Point indices the user picked on the map (untruncated — drives dimming). */
    selectedIds = $state<Set<number>>(new Set());
    /** Point indices of the current search+facet results; null ⇒ no active filter. */
    filteredIds = $state<Set<number> | null>(null);
    /** Which projection the map is showing; the page may read it to fetch per space. */
    space = $state<AtlasSpace>('text');
    /** Colour channel for the scatter. */
    colorBy = $state<ColorBy>('cluster');
    /** Cluster ids the user has hidden via the legend (recoloured to background). */
    hiddenClusters = $state<Set<number>>(new Set());
    /** Hovered point index (for the analysis popover); null ⇒ nothing hovered. */
    hovered = $state<number | null>(null);

    /** `doc|speech|chunk → index` for the CURRENT space (rebuilt on space swap). */
    keyToIndex = $state.raw<Map<string, number>>(new Map());

    get hasSelection(): boolean {
        return this.selectedIds.size > 0;
    }
    get hasFilter(): boolean {
        return this.filteredIds !== null;
    }

    // ── setters ──────────────────────────────────────────────────────────────

    /** Project search Hit[] to point indices via the current key→index map. */
    setFilteredFromHits(hits: readonly Hit[]): void {
        const map = this.keyToIndex;
        const next = new Set<number>();
        for (const h of hits) {
            const i = map.get(hitKey(h));
            if (i !== undefined) next.add(i);
        }
        this.filteredIds = next;
    }

    /** Clear the search→map filter (back to the whole corpus). */
    clearFilter(): void {
        this.filteredIds = null;
    }

    /** Replace the map selection with an explicit FULL index set (no truncation). */
    setSelectedIndices(idx: readonly number[]): void {
        this.selectedIds = new Set(idx);
    }

    clearSelection(): void {
        if (this.selectedIds.size > 0) this.selectedIds = new Set();
    }

    /** Select every point in a cluster — a first-class lasso-equivalent. */
    selectCluster(clusterId: number, clusters: readonly number[]): void {
        const next = new Set<number>();
        for (let i = 0; i < clusters.length; i++) {
            if (clusters[i] === clusterId) next.add(i);
        }
        this.selectedIds = next;
    }

    setSpace(s: AtlasSpace): void {
        this.space = s;
    }

    /** Toggle a cluster's visibility on the map (reassigns the Set so reactivity fires). */
    toggleClusterHidden(id: number): void {
        const next = new Set(this.hiddenClusters);
        if (next.has(id)) next.delete(id);
        else next.add(id);
        this.hiddenClusters = next;
    }

    /** Un-hide every cluster. */
    showAllClusters(): void {
        if (this.hiddenClusters.size > 0) this.hiddenClusters = new Set();
    }

    /** Reset everything tied to the loaded universe (called on a space swap). */
    resetForSpace(keyToIndex: Map<string, number>): void {
        this.keyToIndex = keyToIndex;
        this.selectedIds = new Set();
        this.filteredIds = null;
        this.hiddenClusters = new Set();
        this.hovered = null;
    }
}

/** The singleton, imported by both the page and AtlasMap so they lock-step. */
export const crossFilter = new CrossFilter();
