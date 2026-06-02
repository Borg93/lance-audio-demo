<script lang="ts">
  /**
   * Atlas — a custom embedding map, read straight from our Lance backend (no
   * DuckDB / Mosaic / parquet). The scatter is Apple's lightweight EmbeddingView
   * renderer; everything around it is ours and reuses the search UI:
   *
   *   • points + colours        ← GET /api/atlas/points   (Lance scan)
   *   • lasso/box → results     ← POST /api/atlas/chunks   (Lance scan, by key)
   *   • click a point → play    ← GET /api/atlas/chunk/..  (Lance scan)
   *   • results table           ← <HitTable>   (same as search)
   *   • playback                ← <PlayerPane> (same as search)
   *
   * Layout mirrors the search page: scatter | player on top, the selected-chunk
   * results table across the bottom.
   */
  import { onMount } from 'svelte';
  import { EmbeddingView } from 'embedding-atlas/svelte';
  import type { DataPoint, Point, Rectangle } from 'embedding-atlas';
  import {
    getAtlasPoints,
    getAtlasChunk,
    getAtlasChunks,
    type AtlasPoints,
    type Hit,
  } from '$lib/api';
  import ResizableSplit from '$lib/components/resizable-split.svelte';
  import PlayerPane from '$lib/components/player-pane.svelte';
  import HitTable from '$lib/components/hit-table.svelte';
  import { Button } from '$lib/components/ui';
  import { Loader2, Lasso } from 'lucide-svelte';

  let error = $state<string | null>(null);
  let pts = $state.raw<AtlasPoints | null>(null);
  let x = $state.raw<Float32Array | null>(null);
  let y = $state.raw<Float32Array | null>(null);

  let colorBy = $state<'cluster' | 'none'>('cluster');
  let pointSize = $state(0); // 0 = auto

  // selection state
  let active = $state.raw<Hit | null>(null); // single chunk → player
  let tableHits = $state.raw<Hit[]>([]); // lasso/box selection → table
  let tableLoading = $state(false);
  let selectionCount = $state(0);
  let rangeSel = $state.raw<Rectangle | null>(null);

  // scatter size (EmbeddingView needs explicit width/height)
  let mapW = $state(0);
  let mapH = $state(0);

  const TABLE_COLS = ['thumbnail', 'namn', 'start', 'end', 'text'];

  // theme follows the app's theme button (toggles `.dark` on <html>)
  let isDark = $state(document.documentElement.classList.contains('dark'));
  $effect(() => {
    const obs = new MutationObserver(() => {
      isDark = document.documentElement.classList.contains('dark');
    });
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
    return () => obs.disconnect();
  });
  const colorScheme = $derived<'light' | 'dark'>(isDark ? 'dark' : 'light');

  onMount(async () => {
    try {
      const p = await getAtlasPoints();
      pts = p;
      x = Float32Array.from(p.x);
      y = Float32Array.from(p.y);
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    }
  });

  // ── colouring + cluster legend ──────────────────────────────────────────
  type Coloring = { category: Uint8Array | null; colors: string[] | null };
  const coloring = $derived.by((): Coloring => {
    const cl = pts?.cluster;
    if (colorBy !== 'cluster' || !cl) return { category: null, colors: null };
    let maxC = 0;
    for (const c of cl) if (c > maxC) maxC = c;
    const category = new Uint8Array(cl.length);
    for (let i = 0; i < cl.length; i++) {
      const c = cl[i]!;
      category[i] = c < 0 ? 0 : Math.min(255, c + 1);
    }
    const n = Math.min(256, maxC + 2);
    const colors = Array.from(
      { length: n },
      (_, i) => `hsl(${Math.round((i * 360) / Math.max(1, n))} 62% ${isDark ? 58 : 48}%)`,
    );
    colors[0] = isDark ? '#3a3a40' : '#cccccc'; // noise
    return { category, colors };
  });

  /** Per-cluster counts (the "distribution"), real clusters only, biggest first. */
  const legend = $derived.by((): { id: number; color: string; count: number }[] => {
    const cl = pts?.cluster;
    const colors = coloring.colors;
    if (!cl || !colors) return [];
    const counts = new Map<number, number>();
    for (const c of cl) if (c >= 0) counts.set(c, (counts.get(c) ?? 0) + 1);
    return [...counts.entries()]
      .map(([id, count]) => ({ id, color: colors[id + 1] ?? '#888', count }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 12);
  });

  const config = $derived({
    mode: 'points' as const,
    colorScheme,
    pointSize: pointSize > 0 ? pointSize : null,
    autoLabelEnabled: false,
  });

  // ── selection helpers ───────────────────────────────────────────────────
  function keyAt(i: number): [string, number, number] | null {
    const p = pts;
    if (!p) return null;
    const dc = p.doc[i];
    const doc = dc !== undefined ? p.docs[dc] : undefined;
    const s = p.speech_id[i];
    const c = p.chunk_id[i];
    if (doc === undefined || s === undefined || c === undefined) return null;
    return [doc, s, c];
  }

  function pointInPolygon(px: number, py: number, poly: Point[]): boolean {
    let inside = false;
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      const a = poly[i]!;
      const b = poly[j]!;
      if (a.y > py !== b.y > py && px < ((b.x - a.x) * (py - a.y)) / (b.y - a.y) + a.x) {
        inside = !inside;
      }
    }
    return inside;
  }

  function indicesInRange(range: Rectangle | Point[]): number[] {
    const xs = x;
    const ys = y;
    if (!xs || !ys) return [];
    const out: number[] = [];
    if (Array.isArray(range)) {
      for (let i = 0; i < xs.length; i++) {
        if (pointInPolygon(xs[i]!, ys[i]!, range)) out.push(i);
      }
    } else {
      for (let i = 0; i < xs.length; i++) {
        const xi = xs[i]!;
        const yi = ys[i]!;
        if (xi >= range.xMin && xi <= range.xMax && yi >= range.yMin && yi <= range.yMax) {
          out.push(i);
        }
      }
    }
    return out;
  }

  /** Nearest point to (qx, qy) — a linear scan over the typed arrays. */
  async function querySelection(qx: number, qy: number, unit: number): Promise<DataPoint | null> {
    const xs = x;
    const ys = y;
    if (!xs || !ys) return null;
    let best = -1;
    let bestD = Infinity;
    for (let i = 0; i < xs.length; i++) {
      const dx = xs[i]! - qx;
      const dy = ys[i]! - qy;
      const d = dx * dx + dy * dy;
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    }
    if (best < 0 || Math.sqrt(bestD) > unit * 12) return null;
    return { x: xs[best] ?? 0, y: ys[best] ?? 0, identifier: best };
  }

  async function onSelection(sel: DataPoint[] | null): Promise<void> {
    const dp = sel && sel.length ? sel[sel.length - 1] : null;
    const i = typeof dp?.identifier === 'number' ? dp.identifier : null;
    if (i == null) return;
    const key = keyAt(i);
    if (!key) return;
    try {
      active = await getAtlasChunk(key[0], key[1], key[2]);
    } catch {
      /* keep previous */
    }
  }

  async function onRangeSelection(range: Rectangle | Point[] | null): Promise<void> {
    rangeSel = range && !Array.isArray(range) ? range : null;
    if (!range) {
      selectionCount = 0;
      tableHits = [];
      return;
    }
    const idx = indicesInRange(range);
    selectionCount = idx.length;
    if (idx.length === 0) {
      tableHits = [];
      return;
    }
    const keys = idx
      .slice(0, 1000)
      .map((i) => keyAt(i))
      .filter((k): k is [string, number, number] => k !== null);
    tableLoading = true;
    try {
      tableHits = await getAtlasChunks(keys);
    } catch {
      tableHits = [];
    } finally {
      tableLoading = false;
    }
  }
</script>

{#if error}
  <div class="grid h-full place-items-center p-6 text-center text-sm text-destructive">
    Failed to load the embedding map: {error}
  </div>
{:else if !x || !y || !pts}
  <div class="flex h-full items-center justify-center gap-2 text-sm text-muted-foreground">
    <Loader2 class="size-4 animate-spin" /> Loading embedding map…
  </div>
{:else}
  <div class="grid h-full min-h-0 grid-rows-[1fr_auto]">
    <!-- top: scatter | player -->
    <ResizableSplit minLeft={420} minRight={320} initial={0.66}>
      {#snippet left()}
        <div bind:clientWidth={mapW} bind:clientHeight={mapH} class="relative h-full min-h-0 bg-background">
          {#if mapW > 0 && mapH > 0}
            <EmbeddingView
              data={{ x, y, category: coloring.category }}
              categoryColors={coloring.colors}
              width={mapW}
              height={mapH}
              pixelRatio={2}
              config={config}
              rangeSelection={rangeSel}
              querySelection={querySelection}
              onSelection={onSelection}
              onRangeSelection={onRangeSelection}
            />
          {/if}

          <!-- toolbar -->
          <div class="absolute left-3 top-3 flex items-center gap-2 rounded-md bg-card/85 px-2 py-1 text-[11px] shadow-sm backdrop-blur">
            <span class="text-muted-foreground">{pts.count.toLocaleString()} chunks</span>
            <span class="text-border">·</span>
            <span class="text-muted-foreground">color</span>
            <Button variant={colorBy === 'cluster' ? 'secondary' : 'ghost'} size="sm" onclick={() => (colorBy = 'cluster')}>
              Cluster
            </Button>
            <Button variant={colorBy === 'none' ? 'secondary' : 'ghost'} size="sm" onclick={() => (colorBy = 'none')}>
              None
            </Button>
            <span class="text-border">·</span>
            <span class="text-muted-foreground">size</span>
            <input type="range" min="0" max="8" step="0.5" bind:value={pointSize} class="w-20 accent-primary" />
          </div>

          <!-- cluster legend / distribution -->
          {#if colorBy === 'cluster' && legend.length}
            <div class="absolute right-3 top-3 max-h-[60%] w-44 overflow-y-auto rounded-md bg-card/85 p-2 text-[11px] shadow-sm backdrop-blur">
              <div class="mb-1 font-medium text-muted-foreground">Clusters · top {legend.length}</div>
              {#each legend as c (c.id)}
                <div class="flex items-center gap-2 py-0.5">
                  <span class="size-2.5 shrink-0 rounded-full" style:background={c.color}></span>
                  <span class="text-foreground">#{c.id}</span>
                  <span class="ml-auto font-mono text-muted-foreground">{c.count.toLocaleString()}</span>
                </div>
              {/each}
            </div>
          {/if}

          <div class="pointer-events-none absolute bottom-3 left-3 flex items-center gap-1.5 rounded-md bg-card/80 px-2 py-1 text-[11px] text-muted-foreground backdrop-blur">
            <Lasso class="size-3.5" /> drag to lasso · click a point to play
          </div>
        </div>
      {/snippet}

      {#snippet right()}
        <div class="h-full min-h-0 bg-muted/30">
          <PlayerPane hit={active} />
        </div>
      {/snippet}
    </ResizableSplit>

    <!-- bottom: results table for the current lasso/box selection -->
    <div class="flex h-72 min-h-0 flex-col border-t border-border bg-card/30">
      <div class="flex items-center gap-2 border-b border-border px-3 py-1.5 text-xs">
        <span class="font-medium text-foreground">Selection</span>
        {#if tableLoading}
          <Loader2 class="size-3.5 animate-spin text-muted-foreground" />
        {:else if selectionCount > 0}
          <span class="text-muted-foreground">
            {selectionCount.toLocaleString()} chunks
            {#if selectionCount > tableHits.length}<span class="text-muted-foreground/70">· showing {tableHits.length}</span>{/if}
          </span>
        {:else}
          <span class="text-muted-foreground">lasso a region of the map to list its chunks</span>
        {/if}
      </div>
      <div class="min-h-0 flex-1 overflow-auto">
        {#if tableHits.length}
          <HitTable hits={tableHits} active={active} visible={TABLE_COLS} query="" onselect={(h) => (active = h)} />
        {/if}
      </div>
    </div>
  </div>
{/if}
