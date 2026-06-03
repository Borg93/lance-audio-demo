<script lang="ts">
  /**
   * AtlasMap — the EVōC embedding map as a reusable child of the search page.
   *
   * Renders a custom WebGPU point scatter (<GpuScatter>) over Float32Array x/y
   * for the CURRENT projection space. The renderer draws instanced quads shaded
   * into round, premultiplied sprites — modern WebGPU only, no legacy fallback.
   *
   * The shared cross-filter store fuses into a per-point RGBA buffer (the
   * shader reads true per-point alpha, so "dim" is just a low alpha — no
   * reserved colour slot, no 256-category cap). Bidirectional:
   *
   *   • search/filter → `crossFilter.filteredIds` dims non-matching points;
   *   • lasso/cluster on the map → `crossFilter.selectedIds` fills the table
   *     (via the page) and can seed a search.
   *
   * Everything is Lance-native: points ← GET /api/atlas/points, selection detail
   * ← POST /api/atlas/chunks, single playback ← GET /api/atlas/chunk.
   */
  import GpuScatter from './gpu-scatter.svelte';
  import {
    getAtlasPoints,
    getAtlasChunk,
    getAtlasChunks,
    getAtlasStatus,
    type AtlasPoints,
    type AtlasSpace,
    type Hit,
  } from '$lib/api';
  import { crossFilter, buildKeyIndex, type ColorBy } from './cross-filter.svelte';
  import AtlasTooltip from './AtlasTooltip.svelte';
  import { Button, Select, type SelectOption } from '$lib/components/ui';
  import { Loader2, Lasso, X, Eye, EyeOff, Hand } from 'lucide-svelte';

  let {
    active = $bindable(null),
    onSeedSearch,
    onSelectionHits,
  }: {
    active?: Hit | null;
    /** Promote the current map selection to a search (by stable `_rowid`). */
    onSeedSearch?: (rowids: number[]) => void;
    /** Surface the lasso/box selection's hits to the page (drives HitTable). */
    onSelectionHits?: (hits: Hit[], total: number) => void;
  } = $props();

  // The WebGPU scatter reads TRUE per-point alpha, so the old 256-slot /
  // categoryColors palette (and its grey-padding monochrome bug) is gone: we
  // build a per-point RGBA buffer directly. We still bound the number of
  // DISTINCT cluster/language hues only to keep the legend sane.
  const MAX_DISTINCT = 512;
  // How many legend rows to render (the legend box already scrolls). The visual
  // space has ~2000 mostly-tiny clusters, so we show the largest N and note the
  // rest rather than capping at an arbitrary handful.
  const LEGEND_LIMIT = 200;

  let error = $state<string | null>(null);
  let loading = $state(true);
  let pts = $state.raw<AtlasPoints | null>(null);
  let x = $state.raw<Float32Array | null>(null);
  let y = $state.raw<Float32Array | null>(null);
  let grid = $state.raw<SpatialGrid | null>(null); // uniform bucket index for hover
  let visualBuilt = $state(false); // whether atlas_img_* exists (gates the toggle)

  let pointSize = $state(0); // 0 = auto
  let selectionCount = $state(0);
  let tableLoading = $state(false);
  let mode = $state<'lasso' | 'pan'>('lasso');

  // hover popover
  let hoverIndex = $state<number | null>(null);
  let hoverX = $state(0);
  let hoverY = $state(0);
  let mapEl = $state<HTMLDivElement | null>(null);

  // scatter size
  let mapW = $state(0);
  let mapH = $state(0);

  // theme follows the app's theme button (toggles `.dark` on <html>)
  let isDark = $state(document.documentElement.classList.contains('dark'));
  $effect(() => {
    const obs = new MutationObserver(() => {
      isDark = document.documentElement.classList.contains('dark');
    });
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
    return () => obs.disconnect();
  });

  // ── uniform spatial grid (built once per space) for O(1)-ish hover picking ──
  // Linear-scanning ~145k points on every pointermove is the dominant lag.
  // Instead we bucket point indices into a GRID×GRID lattice over the x/y
  // extent once, then a hover only scans the target cell + its neighbour ring.
  type SpatialGrid = {
    cells: number[][]; // row-major GRID*GRID buckets of point indices
    cols: number;
    rows: number;
    minX: number;
    minY: number;
    invW: number; // 1 / cellWidth
    invH: number; // 1 / cellHeight
  };
  const GRID = 256;

  function buildGrid(xs: Float32Array, ys: Float32Array): SpatialGrid {
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    for (let i = 0; i < xs.length; i++) {
      const xi = xs[i]!;
      const yi = ys[i]!;
      if (xi < minX) minX = xi;
      if (xi > maxX) maxX = xi;
      if (yi < minY) minY = yi;
      if (yi > maxY) maxY = yi;
    }
    if (!Number.isFinite(minX)) {
      minX = 0;
      minY = 0;
      maxX = 1;
      maxY = 1;
    }
    const spanX = maxX - minX || 1;
    const spanY = maxY - minY || 1;
    const invW = GRID / spanX;
    const invH = GRID / spanY;
    const cells: number[][] = Array.from({ length: GRID * GRID }, () => []);
    for (let i = 0; i < xs.length; i++) {
      const cx = Math.min(GRID - 1, Math.max(0, ((xs[i]! - minX) * invW) | 0));
      const cy = Math.min(GRID - 1, Math.max(0, ((ys[i]! - minY) * invH) | 0));
      cells[cy * GRID + cx]!.push(i);
    }
    return { cells, cols: GRID, rows: GRID, minX, minY, invW, invH };
  }

  // ── load points for a space + (re)build the shared key→index map ──────────
  async function load(space: AtlasSpace): Promise<void> {
    loading = true;
    error = null;
    try {
      const p = await getAtlasPoints(space);
      pts = p;
      const xs = Float32Array.from(p.x);
      const ys = Float32Array.from(p.y);
      x = xs;
      y = ys;
      grid = buildGrid(xs, ys);
      crossFilter.resetForSpace(buildKeyIndex(p.docs, p.doc, p.speech_id, p.chunk_id));
      selectionCount = 0;
      onSelectionHits?.([], 0);
    } catch (e) {
      pts = null;
      x = null;
      y = null;
      grid = null;
      error = e instanceof Error ? e.message : String(e);
    } finally {
      loading = false;
    }
  }

  // Probe which spaces exist (gates the Visual toggle), then load the current one.
  $effect(() => {
    let cancelled = false;
    (async () => {
      try {
        const status = await getAtlasStatus(crossFilter.space);
        if (!cancelled) visualBuilt = status.spaces?.visual ?? false;
      } catch {
        /* leave visualBuilt as-is */
      }
    })();
    return () => {
      cancelled = true;
    };
  });

  // Load whenever the chosen space changes (initial mount + Text/Visual toggle).
  $effect(() => {
    const space = crossFilter.space;
    void load(space);
  });

  async function switchSpace(space: AtlasSpace): Promise<void> {
    if (space === crossFilter.space) return;
    if (space === 'visual' && !visualBuilt) return; // gated
    crossFilter.setSpace(space);
  }

  // ── colour helpers (no dep): HSL/hex → {r,g,b} bytes ──────────────────────
  type Rgb = { r: number; g: number; b: number };

  function hslToRgb(h: number, s: number, l: number): Rgb {
    // h in [0,360), s/l in [0,1]
    const c = (1 - Math.abs(2 * l - 1)) * s;
    const hp = (((h % 360) + 360) % 360) / 60;
    const xCol = c * (1 - Math.abs((hp % 2) - 1));
    let r1 = 0;
    let g1 = 0;
    let b1 = 0;
    if (hp < 1) [r1, g1, b1] = [c, xCol, 0];
    else if (hp < 2) [r1, g1, b1] = [xCol, c, 0];
    else if (hp < 3) [r1, g1, b1] = [0, c, xCol];
    else if (hp < 4) [r1, g1, b1] = [0, xCol, c];
    else if (hp < 5) [r1, g1, b1] = [xCol, 0, c];
    else [r1, g1, b1] = [c, 0, xCol];
    const m = l - c / 2;
    return {
      r: Math.round((r1 + m) * 255),
      g: Math.round((g1 + m) * 255),
      b: Math.round((b1 + m) * 255),
    };
  }

  function hexToRgb(hex: string): Rgb {
    let h = hex.trim().replace('#', '');
    if (h.length === 3) h = h[0]! + h[0]! + h[1]! + h[1]! + h[2]! + h[2]!;
    const n = Number.parseInt(h, 16);
    if (!Number.isFinite(n)) return { r: 0, g: 0, b: 0 };
    return { r: (n >> 16) & 255, g: (n >> 8) & 255, b: n & 255 };
  }

  /** Ranked hue for index i of n distinct categories (matches old `hsl()`). */
  function hueRgb(i: number, n: number): Rgb {
    const hue = Math.round((i * 360) / Math.max(1, n));
    return hslToRgb(hue, 0.62, isDark ? 0.58 : 0.48);
  }
  /** CSS string for the same hue (legend swatches read this). */
  function hueCss(i: number, n: number): string {
    return `hsl(${Math.round((i * 360) / Math.max(1, n))} 62% ${isDark ? 58 : 48}%)`;
  }

  const NOISE_RGB = $derived(hexToRgb(isDark ? '#52525b' : '#cccccc'));
  const OTHER_RGB = $derived(hexToRgb(isDark ? '#71717a' : '#a1a1aa'));
  const ACCENT_RGB = $derived(hueRgb(1, 4));
  const BG_COLOR = $derived(isDark ? '#0a0a0a' : '#ffffff');

  const DIM_ALPHA = 30; // cross-filter-dimmed points: faint but present
  const NOISE_ALPHA = 70; // HDBSCAN noise: muted so coloured clusters dominate

  /**
   * THEME-INDEPENDENT size ranking. `slotOf` maps clusterId → its dense rank
   * (0..distinct-1, or -1 for the small-cluster tail beyond MAX_DISTINCT).
   * `distinct` is how many distinct cluster hues we emit.
   */
  const clusterRanking = $derived.by((): { slotOf: Map<number, number>; distinct: number } | null => {
    const cl = pts?.cluster;
    if (!cl) return null;
    const counts = new Map<number, number>();
    for (const c of cl) if (c >= 0) counts.set(c, (counts.get(c) ?? 0) + 1);
    const ranked = [...counts.entries()].sort((a, b) => b[1] - a[1]);
    const distinct = Math.min(ranked.length, MAX_DISTINCT);
    const slotOf = new Map<number, number>();
    ranked.forEach(([id], rank) => slotOf.set(id, rank < MAX_DISTINCT ? rank : -1));
    return { slotOf, distinct };
  });

  /**
   * Per-point RGBA buffer (length 4*count). Replaces the old category/
   * categoryColors pair — the shader reads true per-point alpha so dimming is
   * just a low alpha byte, removing the 256-slot palette limit entirely.
   * Recomputed when colorBy / space / filteredIds / selectedIds /
   * hiddenClusters / theme change — NOT on hover.
   */
  const rgba = $derived.by((): Uint8Array | null => {
    const p = pts;
    if (!p) return null;
    const n = p.count;
    const colorBy = crossFilter.colorBy;
    const buf = new Uint8Array(n * 4);

    if (colorBy === 'cluster' && p.cluster) {
      const r = clusterRanking;
      if (!r) return null;
      const { slotOf, distinct } = r;
      const cl = p.cluster;
      const hidden = crossFilter.hiddenClusters;
      const noise = NOISE_RGB;
      const other = OTHER_RGB;
      for (let i = 0; i < n; i++) {
        const o = i * 4;
        const c = cl[i]!;
        if (c >= 0 && hidden.has(c)) {
          buf[o + 3] = 0; // fully hidden
          continue;
        }
        let col: Rgb;
        let alpha: number;
        if (c < 0) {
          col = noise;
          alpha = NOISE_ALPHA;
        } else {
          const rank = slotOf.get(c) ?? -1;
          col = rank < 0 ? other : hueRgb(rank + 1, distinct + 1);
          alpha = 255;
        }
        if (!crossFilter.visible(i)) alpha = DIM_ALPHA;
        buf[o] = col.r;
        buf[o + 1] = col.g;
        buf[o + 2] = col.b;
        buf[o + 3] = alpha;
      }
      return buf;
    }

    if (colorBy === 'language' && p.language && p.languages) {
      const lang = p.language;
      const distinct = Math.min(MAX_DISTINCT, p.languages.length);
      const other = OTHER_RGB;
      for (let i = 0; i < n; i++) {
        const o = i * 4;
        const code = lang[i] ?? 0;
        const col = code < distinct ? hueRgb(code + 1, distinct + 1) : other;
        const alpha = crossFilter.visible(i) ? 255 : DIM_ALPHA;
        buf[o] = col.r;
        buf[o + 1] = col.g;
        buf[o + 2] = col.b;
        buf[o + 3] = alpha;
      }
      return buf;
    }

    // colorBy === 'none' (or the channel's data is absent): one accent hue.
    const accent = ACCENT_RGB;
    for (let i = 0; i < n; i++) {
      const o = i * 4;
      buf[o] = accent.r;
      buf[o + 1] = accent.g;
      buf[o + 2] = accent.b;
      buf[o + 3] = crossFilter.visible(i) ? 255 : DIM_ALPHA;
    }
    return buf;
  });

  const autoPointSize = $derived(pointSize > 0 ? pointSize : 3);

  // ── legends (clickable → selectCluster / language facet) ──────────────────
  // Legend swatches read the SAME hslToRgb hues the map uses, so a swatch can
  // never disagree with its points.
  const clusterLegend = $derived.by((): { id: number; color: string; count: number }[] => {
    const p = pts;
    const r = clusterRanking;
    if (!p?.cluster || !r) return [];
    const { slotOf, distinct } = r;
    const counts = new Map<number, number>();
    for (const c of p.cluster) if (c >= 0) counts.set(c, (counts.get(c) ?? 0) + 1);
    return [...counts.entries()]
      .map(([id, count]) => {
        const rank = slotOf.get(id) ?? -1;
        const color = rank < 0 ? (isDark ? '#71717a' : '#a1a1aa') : hueCss(rank + 1, distinct + 1);
        return { id, color, count };
      })
      .sort((a, b) => b.count - a.count)
      .slice(0, LEGEND_LIMIT);
  });

  /** Distinct non-noise cluster count + noise (unclustered) point count — for
   *  the legend's "showing X of Y" line and the noise summary row. */
  const clusterStats = $derived.by((): { total: number; noise: number } => {
    const cl = pts?.cluster;
    if (!cl) return { total: 0, noise: 0 };
    const ids = new Set<number>();
    let noise = 0;
    for (const c of cl) {
      if (c < 0) noise++;
      else ids.add(c);
    }
    return { total: ids.size, noise };
  });

  const languageLegend = $derived.by((): { code: number; label: string; color: string; count: number }[] => {
    const p = pts;
    if (!p?.language || !p.languages) return [];
    const lang = p.language;
    const labels = p.languages;
    const distinct = Math.min(MAX_DISTINCT, labels.length);
    const counts = new Map<number, number>();
    for (const code of lang) counts.set(code, (counts.get(code) ?? 0) + 1);
    return [...counts.entries()]
      .map(([code, count]) => ({
        code,
        label: labels[code] || '—',
        color: code < distinct ? hueCss(code + 1, distinct + 1) : isDark ? '#71717a' : '#a1a1aa',
        count,
      }))
      .sort((a, b) => b.count - a.count)
      .slice(0, LEGEND_LIMIT);
  });

  const languageTotal = $derived(pts?.languages?.length ?? 0);

  const colorOptions = [
    { value: 'cluster', label: 'Cluster' },
    { value: 'language', label: 'Language' },
    { value: 'none', label: 'None' },
  ] satisfies (SelectOption & { value: ColorBy })[];
  const isColorBy = (v: string): v is ColorBy => colorOptions.some((o) => o.value === v);
  let colorByValue = $state<string>('cluster');
  $effect(() => {
    // The Select binds a plain string; narrow it to the store's literal union
    // via the known options (type guard, no `as`).
    if (isColorBy(colorByValue)) crossFilter.colorBy = colorByValue;
  });

  // ── selection helpers (data-space) ────────────────────────────────────────
  type Pt = { x: number; y: number };

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

  function pointInPolygon(px: number, py: number, poly: Pt[]): boolean {
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

  function indicesInPolygon(poly: Pt[]): number[] {
    const xs = x;
    const ys = y;
    if (!xs || !ys) return [];
    const out: number[] = [];
    for (let i = 0; i < xs.length; i++) {
      if (pointInPolygon(xs[i]!, ys[i]!, poly)) out.push(i);
    }
    return out;
  }

  /** Nearest point index to data (qx, qy) via the spatial grid: scan the target
   *  cell and widening neighbour rings until a hit is found within the radius. */
  function nearestIndex(qx: number, qy: number): number | null {
    const xs = x;
    const ys = y;
    const g = grid;
    if (!xs || !ys || !g) return null;
    const unit = 1 / Math.max(g.invW, g.invH); // ~one grid cell in data units
    const maxR = unit * 12; // same pick radius as before
    const cx = Math.min(g.cols - 1, Math.max(0, ((qx - g.minX) * g.invW) | 0));
    const cy = Math.min(g.rows - 1, Math.max(0, ((qy - g.minY) * g.invH) | 0));
    let best = -1;
    let bestD = Infinity;
    for (let r = 0; r < g.cols; r++) {
      const x0 = Math.max(0, cx - r);
      const x1 = Math.min(g.cols - 1, cx + r);
      const y0 = Math.max(0, cy - r);
      const y1 = Math.min(g.rows - 1, cy + r);
      for (let gy = y0; gy <= y1; gy++) {
        for (let gx = x0; gx <= x1; gx++) {
          if (r > 0 && gx > x0 && gx < x1 && gy > y0 && gy < y1) continue; // interior already scanned
          const bucket = g.cells[gy * g.cols + gx]!;
          for (const i of bucket) {
            const dx = xs[i]! - qx;
            const dy = ys[i]! - qy;
            const d = dx * dx + dy * dy;
            if (d < bestD) {
              bestD = d;
              best = i;
            }
          }
        }
      }
      const ringEdge = r / Math.max(g.invW, g.invH);
      if (best >= 0 && ringEdge * ringEdge > bestD) break;
      if (ringEdge > maxR) break;
    }
    if (best < 0 || Math.sqrt(bestD) > maxR) return null;
    return best;
  }

  // ── load the table for a set of indices (1000-cap; full set stays in store) ─
  // Resolved by stable `_rowid` (one per point from /points) — the backend takes
  // exactly these rows, so even a max-size selection lists in a few hundred ms.
  async function loadTableForIndices(idx: number[]): Promise<void> {
    selectionCount = idx.length;
    if (idx.length === 0) {
      onSelectionHits?.([], 0);
      return;
    }
    const rowids = pts?.rowid;
    if (!rowids) {
      onSelectionHits?.([], idx.length);
      return;
    }
    const ids = idx
      .slice(0, 1000)
      .map((i) => rowids[i])
      .filter((r): r is number => r !== undefined);
    tableLoading = true;
    try {
      const hits = await getAtlasChunks(ids);
      onSelectionHits?.(hits, idx.length);
    } catch (e) {
      console.error('atlas: failed to load selection', e);
      onSelectionHits?.([], idx.length);
    } finally {
      tableLoading = false;
    }
  }

  // ── GpuScatter callbacks ──────────────────────────────────────────────────
  // hover → nearest index → tooltip + crossFilter.hovered (anchored at cursor)
  function onScatterHover(dataX: number, dataY: number): void {
    const i = nearestIndex(dataX, dataY);
    hoverIndex = i;
    crossFilter.hovered = i;
    if (i != null) {
      hoverX = lastPointerX;
      hoverY = lastPointerY;
    }
  }
  function onScatterHoverEnd(): void {
    hoverIndex = null;
    crossFilter.hovered = null;
  }

  // click a point → load full hit into the shared player
  async function onScatterPick(dataX: number, dataY: number): Promise<void> {
    const i = nearestIndex(dataX, dataY);
    if (i == null) return;
    const key = keyAt(i);
    if (!key) return;
    try {
      active = await getAtlasChunk(key[0], key[1], key[2]);
    } catch {
      /* keep previous */
    }
  }

  // lasso → FULL index selection (drives dimming) + capped table fetch
  async function onScatterLasso(polyDataXY: number[]): Promise<void> {
    const poly: Pt[] = [];
    for (let i = 0; i + 1 < polyDataXY.length; i += 2) {
      poly.push({ x: polyDataXY[i]!, y: polyDataXY[i + 1]! });
    }
    if (poly.length < 3) return;
    const idx = indicesInPolygon(poly);
    crossFilter.setSelectedIndices(idx); // untruncated — drives the dim recolour
    await loadTableForIndices(idx);
  }

  // legend click → select an entire cluster / language (a first-class lasso)
  async function pickCluster(id: number): Promise<void> {
    const cl = pts?.cluster;
    if (!cl) return;
    crossFilter.selectCluster(id, cl);
    await loadTableForIndices([...crossFilter.selectedIds]);
  }

  async function pickLanguage(code: number): Promise<void> {
    const lang = pts?.language;
    if (!lang) return;
    const idx: number[] = [];
    for (let i = 0; i < lang.length; i++) if (lang[i] === code) idx.push(i);
    crossFilter.setSelectedIndices(idx);
    await loadTableForIndices(idx);
  }

  function clearSelection(): void {
    crossFilter.clearSelection();
    selectionCount = 0;
    onSelectionHits?.([], 0);
  }

  function seedSearch(): void {
    const rowids = pts?.rowid;
    if (!rowids) return;
    const ids = [...crossFilter.selectedIds]
      .map((i) => rowids[i])
      .filter((r): r is number => r !== undefined);
    if (ids.length) onSeedSearch?.(ids);
  }

  // ── hover popover wiring ──────────────────────────────────────────────────
  // Plain (non-reactive) cursor coords: read synchronously when a hover lands,
  // never rendered directly — so a pointermove no longer triggers reactivity.
  let lastPointerX = 0;
  let lastPointerY = 0;
  let rafId = 0;
  function flushHover(): void {
    rafId = 0;
    if (hoverIndex != null) {
      hoverX = lastPointerX;
      hoverY = lastPointerY;
    }
  }
  function trackPointer(e: PointerEvent): void {
    const rect = mapEl?.getBoundingClientRect();
    if (!rect) return;
    lastPointerX = e.clientX - rect.left;
    lastPointerY = e.clientY - rect.top;
    if (hoverIndex != null && rafId === 0) {
      rafId = requestAnimationFrame(flushHover);
    }
  }
  $effect(() => () => {
    if (rafId !== 0) cancelAnimationFrame(rafId);
  });

  const legendMode = $derived(crossFilter.colorBy);
</script>

{#if error}
  <div class="grid h-full place-items-center p-6 text-center text-sm text-destructive">
    Failed to load the embedding map: {error}
  </div>
{:else}
  <!-- svelte-ignore a11y_no_static_element_interactions -->
  <div
    bind:this={mapEl}
    bind:clientWidth={mapW}
    bind:clientHeight={mapH}
    onpointermove={trackPointer}
    role="presentation"
    class="relative h-full min-h-0 bg-background"
  >
    {#if loading || !x || !y || !pts || !rgba}
      <div class="flex h-full items-center justify-center gap-2 text-sm text-muted-foreground">
        <Loader2 class="size-4 animate-spin" /> Loading embedding map…
      </div>
    {:else}
      {#if mapW > 0 && mapH > 0}
        <GpuScatter
          {x}
          {y}
          {rgba}
          pointSize={autoPointSize}
          width={mapW}
          height={mapH}
          background={BG_COLOR}
          onHover={onScatterHover}
          onHoverEnd={onScatterHoverEnd}
          onPick={onScatterPick}
          onLasso={onScatterLasso}
          {mode}
        />
      {/if}

      <AtlasTooltip pts={pts} index={hoverIndex} x={hoverX} y={hoverY} />

      <!-- toolbar -->
      <div class="absolute left-3 top-3 flex flex-wrap items-center gap-2 rounded-md bg-card/85 px-2 py-1 text-[11px] shadow-sm backdrop-blur">
        <span class="text-muted-foreground">{(pts.count ?? 0).toLocaleString()} chunks</span>
        <span class="text-border">·</span>
        <!-- Text / Visual space toggle -->
        <span class="text-muted-foreground">space</span>
        <Button
          variant={crossFilter.space === 'text' ? 'secondary' : 'ghost'}
          size="sm"
          onclick={() => switchSpace('text')}
        >
          Text
        </Button>
        <Button
          variant={crossFilter.space === 'visual' ? 'secondary' : 'ghost'}
          size="sm"
          disabled={!visualBuilt}
          title={visualBuilt ? 'Visual (frame embedding) map' : 'Not built yet — run `raudio feature atlas --space visual`'}
          onclick={() => switchSpace('visual')}
        >
          Visual
        </Button>
        <span class="text-border">·</span>
        <!-- lasso / pan mode -->
        <Button
          variant={mode === 'lasso' ? 'secondary' : 'ghost'}
          size="sm"
          onclick={() => (mode = 'lasso')}
          title="Lasso: drag to select a region"
        >
          <Lasso class="size-3.5" /> Lasso
        </Button>
        <Button
          variant={mode === 'pan' ? 'secondary' : 'ghost'}
          size="sm"
          onclick={() => (mode = 'pan')}
          title="Pan: drag to move (shift+drag or middle-drag pans in any mode; scroll to zoom)"
        >
          <Hand class="size-3.5" /> Pan
        </Button>
        <span class="text-border">·</span>
        <span class="text-muted-foreground">color</span>
        <Select bind:value={colorByValue} options={colorOptions} class="h-7 w-28" ariaLabel="Color by" />
        <span class="text-border">·</span>
        <span class="text-muted-foreground">size</span>
        <input type="range" min="0" max="8" step="0.5" bind:value={pointSize} class="w-16 accent-primary" />
      </div>

      <!-- legend / distribution (clickable → select) -->
      {#if legendMode === 'cluster' && clusterLegend.length}
        <div class="absolute right-3 top-3 max-h-[60%] w-52 overflow-y-auto rounded-md bg-card/85 p-2 text-[11px] shadow-sm backdrop-blur">
          <div class="mb-1 flex items-center gap-2">
            <span class="font-medium text-muted-foreground">Clusters · {clusterLegend.length} of {clusterStats.total}</span>
            {#if crossFilter.hiddenClusters.size > 0}
              <button
                type="button"
                class="ml-auto rounded px-1 py-0.5 text-primary hover:bg-secondary/50"
                onclick={() => crossFilter.showAllClusters()}
                title="Show all hidden clusters"
              >
                Show all
              </button>
            {/if}
          </div>
          {#each clusterLegend as c (c.id)}
            {@const hidden = crossFilter.hiddenClusters.has(c.id)}
            <div class="flex w-full items-center gap-1 rounded px-1 py-0.5 hover:bg-secondary/50" class:opacity-40={hidden}>
              <button
                type="button"
                class="flex flex-1 items-center gap-2 text-left"
                onclick={() => pickCluster(c.id)}
                title="Select cluster #{c.id}"
              >
                <span class="size-2.5 shrink-0 rounded-full" style:background={c.color}></span>
                <span class="text-foreground">#{c.id}</span>
                <span class="ml-auto font-mono text-muted-foreground">{c.count.toLocaleString()}</span>
              </button>
              <button
                type="button"
                class="shrink-0 rounded p-0.5 text-muted-foreground hover:text-foreground"
                onclick={() => crossFilter.toggleClusterHidden(c.id)}
                title={hidden ? 'Show cluster on map' : 'Hide cluster on map'}
              >
                {#if hidden}
                  <EyeOff class="size-3.5" />
                {:else}
                  <Eye class="size-3.5" />
                {/if}
              </button>
            </div>
          {/each}
          {#if clusterStats.noise > 0}
            <div class="mt-1 flex items-center gap-2 border-t border-border/60 px-1 pt-1 text-muted-foreground">
              <span class="size-2.5 shrink-0 rounded-full" style:background={isDark ? '#52525b' : '#cccccc'}></span>
              <span>noise (unclustered)</span>
              <span class="ml-auto font-mono">{clusterStats.noise.toLocaleString()}</span>
            </div>
          {/if}
          {#if clusterStats.total > clusterLegend.length}
            <div class="mt-1 px-1 text-muted-foreground/70">
              +{(clusterStats.total - clusterLegend.length).toLocaleString()} smaller clusters not shown
            </div>
          {/if}
        </div>
      {:else if legendMode === 'language' && languageLegend.length}
        <div class="absolute right-3 top-3 max-h-[60%] w-48 overflow-y-auto rounded-md bg-card/85 p-2 text-[11px] shadow-sm backdrop-blur">
          <div class="mb-1 font-medium text-muted-foreground">Languages · {languageLegend.length} of {languageTotal}</div>
          {#each languageLegend as l (l.code)}
            <button
              type="button"
              class="flex w-full items-center gap-2 rounded px-1 py-0.5 text-left hover:bg-secondary/50"
              onclick={() => pickLanguage(l.code)}
              title="Select language {l.label}"
            >
              <span class="size-2.5 shrink-0 rounded-full" style:background={l.color}></span>
              <span class="uppercase text-foreground">{l.label}</span>
              <span class="ml-auto font-mono text-muted-foreground">{l.count.toLocaleString()}</span>
            </button>
          {/each}
        </div>
      {/if}

      <!-- selection status + actions -->
      <div class="absolute bottom-3 left-3 flex items-center gap-2 rounded-md bg-card/85 px-2 py-1 text-[11px] shadow-sm backdrop-blur">
        {#if tableLoading}
          <Loader2 class="size-3.5 animate-spin text-muted-foreground" />
        {/if}
        {#if selectionCount > 0}
          <span class="text-foreground">{selectionCount.toLocaleString()} selected</span>
          {#if selectionCount > 1000}
            <span class="text-muted-foreground/70">· table shows 1000</span>
          {/if}
          <Button variant="ghost" size="sm" onclick={seedSearch} title="Use this selection as the search result set">
            Use as filter
          </Button>
          <Button variant="ghost" size="icon" onclick={clearSelection} title="Clear selection">
            <X class="size-3.5" />
          </Button>
        {:else}
          <Lasso class="size-3.5 text-muted-foreground" />
          <span class="text-muted-foreground">drag to lasso · click a legend to select · click a point to play · scroll to zoom</span>
        {/if}
      </div>
    {/if}
  </div>
{/if}
