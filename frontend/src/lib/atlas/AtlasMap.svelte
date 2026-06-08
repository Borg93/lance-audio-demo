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
  import { crossFilter, buildKeyIndex, hitKey, type ColorBy } from './cross-filter.svelte';
  import { buildGrid, nearestIndex, type SpatialGrid } from './atlas-grid';
  import { hexToRgb, hueRgb, buildHuePalette, type Rgb } from './atlas-colors';
  import { indicesInPolygon, type Pt } from './atlas-geometry';
  import AtlasTooltip from './AtlasTooltip.svelte';
  import AtlasLegend from './AtlasLegend.svelte';
  import {
    visibleIds,
    buildClusterLegend,
    clusterStats as buildClusterStats,
    buildCategoryLegend,
    categoryTitle as deriveCategoryTitle,
    type CategoryChannel,
    type ClusterRanking,
  } from './atlas-legend';
  import { Button, Select, type SelectOption } from '$lib/components/ui';
  import { Loader2, Lasso, X, Hand, Settings2 } from 'lucide-svelte';

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

  let error = $state<string | null>(null);
  let loading = $state(true);
  let pts = $state.raw<AtlasPoints | null>(null);
  let x = $state.raw<Float32Array | null>(null);
  let y = $state.raw<Float32Array | null>(null);
  let grid = $state.raw<SpatialGrid | null>(null); // uniform bucket index for hover
  let visualBuilt = $state(false); // whether atlas_img_* exists (gates the toggle)
  let captionBuilt = $state(false); // whether atlas_cap_* exists (gates the toggle)

  let pointSize = $state(0); // 0 = auto
  let filterAlpha = $state(8); // 0..120 — opacity of EXCLUDED points (search-filtered OR lasso-not-selected)
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

  // Spatial-grid hover picking (`buildGrid`/`nearestIndex`) lives in ./atlas-grid;
  // lasso geometry in ./atlas-geometry; colour math in ./atlas-colors — all pure.

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
        if (!cancelled) {
          visualBuilt = status.spaces?.visual ?? false;
          captionBuilt = status.spaces?.caption ?? false;
        }
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
    if (space === 'caption' && !captionBuilt) return; // gated
    crossFilter.setSpace(space);
  }

  // Colour math (hslToRgb / hexToRgb / hueRgb / hueCss / buildHuePalette) is in
  // ./atlas-colors; the derived swatch/point colours below pass the live theme.
  const NOISE_RGB = $derived(hexToRgb(isDark ? '#52525b' : '#cccccc'));
  const OTHER_RGB = $derived(hexToRgb(isDark ? '#71717a' : '#a1a1aa'));
  const ACCENT_RGB = $derived(hueRgb(1, 4, isDark));
  const BG_COLOR = $derived(isDark ? '#0a0a0a' : '#ffffff');

  const NOISE_ALPHA = 70; // HDBSCAN noise: muted so coloured clusters dominate

  /**
   * THEME-INDEPENDENT size ranking. `slotOf` maps clusterId → its dense rank
   * (0..distinct-1, or -1 for the small-cluster tail beyond MAX_DISTINCT).
   * `distinct` is how many distinct cluster hues we emit.
   */
  const clusterRanking = $derived.by((): ClusterRanking | null => {
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

  /** The factorized (codes, labels) pair backing a categorical colour mode.
   *  `cluster` is NOT here — it has its own noise/#id/hide semantics. */
  function channelFor(p: AtlasPoints | null, mode: ColorBy): CategoryChannel | null {
    if (!p) return null;
    if (mode === 'language' && p.language && p.languages)
      return { codes: p.language, labels: p.languages };
    if (mode === 'topic' && p.topic && p.topics) return { codes: p.topic, labels: p.topics };
    if (mode === 'doc_topic' && p.doc_topic && p.doc_topics)
      return { codes: p.doc_topic, labels: p.doc_topics };
    // `doc`/`docs` (the video ids) are always shipped, so colour-by-Video is
    // always available — one hue per source video.
    if (mode === 'doc') return { codes: p.doc, labels: p.docFiles ?? p.docs };
    return null;
  }

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

    // Read both cross-filter Sets once (tracks them as deps + avoids 145k method
    // calls). A point EXCLUDED by either the search filter or the lasso/cluster
    // selection fades to the slider-controlled `filterAlpha`, so the "Dimmed
    // opacity" slider governs how visible excluded points are in BOTH contexts.
    const fIds = crossFilter.filteredIds;
    const sel = crossFilter.selectedIds;
    const alphaFor = (i: number, base: number): number => {
      const filteredOut = fIds !== null && !fIds.has(i);
      const notSelected = sel.size > 0 && !sel.has(i);
      return filteredOut || notSelected ? filterAlpha : base;
    };

    if (colorBy === 'cluster' && p.cluster) {
      const r = clusterRanking;
      if (!r) return null;
      const { slotOf, distinct } = r;
      const palette = buildHuePalette(distinct, isDark); // distinct hues once, not per point
      const cl = p.cluster;
      const hidden = crossFilter.hiddenClusters;
      const noise = NOISE_RGB;
      const other = OTHER_RGB;
      for (let i = 0; i < n; i++) {
        const o = i * 4;
        const c = cl[i]!;
        // Noise (c < 0) shares one hide key (-1); clusters use their own id.
        if (hidden.has(c < 0 ? -1 : c)) {
          buf[o + 3] = 0; // fully hidden
          continue;
        }
        let col: Rgb;
        let base: number;
        if (c < 0) {
          col = noise;
          base = NOISE_ALPHA;
        } else {
          const rank = slotOf.get(c) ?? -1;
          col = rank < 0 ? other : palette[rank]!;
          base = 255;
        }
        const alpha = alphaFor(i, base);
        buf[o] = col.r;
        buf[o + 1] = col.g;
        buf[o + 2] = col.b;
        buf[o + 3] = alpha;
      }
      return buf;
    }

    const channel = channelFor(p, colorBy);
    if (channel) {
      const { codes, labels } = channel;
      const distinct = Math.min(MAX_DISTINCT, labels.length);
      const palette = buildHuePalette(distinct, isDark); // distinct hues once, not per point
      const other = OTHER_RGB;
      const noise = NOISE_RGB;
      for (let i = 0; i < n; i++) {
        const o = i * 4;
        const code = codes[i] ?? 0;
        const empty = (labels[code] ?? '') === ''; // unclustered/noise → muted grey
        const col = empty ? noise : code < distinct ? palette[code]! : other;
        buf[o] = col.r;
        buf[o + 1] = col.g;
        buf[o + 2] = col.b;
        buf[o + 3] = alphaFor(i, empty ? NOISE_ALPHA : 255);
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
      buf[o + 3] = alphaFor(i, 255);
    }
    return buf;
  });

  const autoPointSize = $derived(pointSize > 0 ? pointSize : 5);

  // Data-space coords of the active hit's point (the one in the player / table)
  // so <GpuScatter> can ring it — resolves via the shared key→index map.
  const activeMarkerXY = $derived.by((): [number, number] | null => {
    const a = active;
    const xs = x;
    const ys = y;
    if (!a || !xs || !ys) return null;
    const i = crossFilter.keyToIndex.get(hitKey(a));
    if (i === undefined) return null;
    const mx = xs[i];
    const my = ys[i];
    return mx !== undefined && my !== undefined ? [mx, my] : null;
  });

  // ── filter-aware legend counts ────────────────────────────────────────────
  // The legend math is PURE (./atlas-legend, unit-testable): these thin deriveds
  // resolve the reactive store reads (filtered/selected sets, colorBy, theme),
  // the channel + ranking, and hand them to the builders. The presentational
  // <AtlasLegend/> renders the rows — it computes nothing.
  const visible = $derived(
    visibleIds(crossFilter.filteredIds, crossFilter.hasSelection ? crossFilter.selectedIds : null),
  );
  const legendFiltered = $derived(visible !== null);
  const clusterRows = $derived(
    pts?.cluster && clusterRanking
      ? buildClusterLegend(pts.cluster, clusterRanking, visible, isDark)
      : [],
  );
  const clusterStatsValue = $derived(buildClusterStats(pts?.cluster, visible));
  const categoryChannel = $derived(channelFor(pts, crossFilter.colorBy));
  const categoryRows = $derived(buildCategoryLegend(categoryChannel, visible, isDark, MAX_DISTINCT));
  const categoryTotal = $derived(categoryChannel?.labels.length ?? 0);
  const categoryTitleValue = $derived(deriveCategoryTitle(crossFilter.colorBy));

  // Topic modes appear only once the columns are built (factorized into the
  // points payload); cluster + none are always available.
  const colorOptions = $derived.by((): (SelectOption & { value: ColorBy })[] => {
    const opts: (SelectOption & { value: ColorBy })[] = [{ value: 'cluster', label: 'Cluster' }];
    if (pts?.language) opts.push({ value: 'language', label: 'Language' });
    if (pts?.topic) opts.push({ value: 'topic', label: 'Topic' });
    if (pts?.doc_topic) opts.push({ value: 'doc_topic', label: 'Video topic' });
    opts.push({ value: 'doc', label: 'Video' });
    opts.push({ value: 'none', label: 'None' });
    return opts;
  });
  const ALL_COLOR_BY: ColorBy[] = ['cluster', 'language', 'topic', 'doc_topic', 'doc', 'none'];
  const isColorBy = (v: string): v is ColorBy => (ALL_COLOR_BY as string[]).includes(v);
  let colorByValue = $state<string>('cluster');
  $effect(() => {
    // The Select binds a plain string; narrow it to the store's literal union
    // via the known options (type guard, no `as`).
    if (isColorBy(colorByValue)) crossFilter.colorBy = colorByValue;
  });

  // Toolbar: the size/opacity sliders live in a ⚙ popover to keep the bar clean.
  let showSettings = $state(false);

  // Space tabs (DRY the segmented toggle); each disabled until its map is built.
  const spaceTabs = $derived.by(
    (): { value: AtlasSpace; label: string; disabled: boolean; title: string }[] => [
      {
        value: 'text',
        label: 'Text',
        disabled: false,
        title: 'Transcript-semantics map (text_embedding)',
      },
      {
        value: 'visual',
        label: 'Visual',
        disabled: !visualBuilt,
        title: visualBuilt
          ? 'Frame-image map (frame_embedding)'
          : 'Not built yet — run `raudio feature atlas --space visual`',
      },
      {
        value: 'caption',
        label: 'Caption',
        disabled: !captionBuilt,
        title: captionBuilt
          ? 'Frame-caption map (caption_embedding)'
          : 'Not built yet — run `raudio feature atlas --space caption`',
      },
    ],
  );

  // ── selection helpers (data-space) ────────────────────────────────────────
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

  // Pure geometry/grid helpers live in ./atlas-geometry + ./atlas-grid; these
  // thin wrappers bind them to the component's currently-loaded points/grid.
  function indicesInLasso(poly: Pt[]): number[] {
    return x && y ? indicesInPolygon(x, y, poly) : [];
  }
  function pickIndex(qx: number, qy: number): number | null {
    return grid && x && y ? nearestIndex(grid, x, y, qx, qy) : null;
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
      // No per-point rowid (e.g. a stale cached /points) — we can't fetch the
      // rows. Report an EMPTY selection (total 0, not idx.length) so the page
      // falls back to its search hits / empty-state rather than showing an empty
      // table under a misleading "N chunks" header. A cache-busted /points
      // (api.ts) restores rowids; this is just the belt-and-braces guard.
      console.warn('atlas: /points payload has no rowid — cannot list selection (stale cache?)');
      selectionCount = 0;
      onSelectionHits?.([], 0);
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
    const i = pickIndex(dataX, dataY);
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
    const i = pickIndex(dataX, dataY);
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
    const idx = indicesInLasso(poly);
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

  /** Select every point in a category (language / topic / doc_topic) → table +
   *  seedable search. The atlas analog of the Tree page's "show results". */
  async function pickCategory(code: number): Promise<void> {
    const ch = channelFor(pts, crossFilter.colorBy);
    if (!ch) return;
    const idx: number[] = [];
    for (let i = 0; i < ch.codes.length; i++) if (ch.codes[i] === code) idx.push(i);
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
          markerXY={activeMarkerXY}
        />
      {/if}

      <AtlasTooltip {pts} index={hoverIndex} x={hoverX} y={hoverY} />

      <!-- toolbar -->
      <div
        class="absolute left-3 top-3 flex flex-wrap items-center gap-1.5 rounded-md bg-card/85 px-2 py-1 text-[11px] shadow-sm backdrop-blur"
      >
        <span class="px-1 text-muted-foreground">{(pts.count ?? 0).toLocaleString()} pts</span>

        <!-- projection space (segmented) -->
        <div class="flex overflow-hidden rounded border border-border">
          {#each spaceTabs as t (t.value)}
            <button
              type="button"
              class="px-2 py-0.5 transition-colors disabled:opacity-40 {crossFilter.space ===
              t.value
                ? 'bg-secondary text-foreground'
                : 'text-muted-foreground hover:bg-secondary/50'}"
              disabled={t.disabled}
              title={t.title}
              onclick={() => switchSpace(t.value)}
            >
              {t.label}
            </button>
          {/each}
        </div>

        <!-- tool: lasso / pan (segmented, icon-only) -->
        <div class="flex overflow-hidden rounded border border-border">
          <button
            type="button"
            class="px-1.5 py-1 transition-colors {mode === 'lasso'
              ? 'bg-secondary text-foreground'
              : 'text-muted-foreground hover:bg-secondary/50'}"
            title="Lasso — drag to select a region"
            aria-label="Lasso select"
            onclick={() => (mode = 'lasso')}
          >
            <Lasso class="size-3.5" />
          </button>
          <button
            type="button"
            class="px-1.5 py-1 transition-colors {mode === 'pan'
              ? 'bg-secondary text-foreground'
              : 'text-muted-foreground hover:bg-secondary/50'}"
            title="Pan — drag to move (or shift / middle-drag in any mode; scroll to zoom)"
            aria-label="Pan"
            onclick={() => (mode = 'pan')}
          >
            <Hand class="size-3.5" />
          </button>
        </div>

        <!-- colour by -->
        <Select
          bind:value={colorByValue}
          options={colorOptions}
          class="h-7 w-32"
          ariaLabel="Colour points by"
        />

        <!-- display settings (point size, filtered opacity) -->
        <div class="relative">
          <button
            type="button"
            class="flex items-center rounded p-1 text-muted-foreground transition-colors hover:bg-secondary/60 hover:text-foreground"
            class:bg-secondary={showSettings}
            title="Display settings"
            aria-label="Display settings"
            onclick={() => (showSettings = !showSettings)}
          >
            <Settings2 class="size-3.5" />
          </button>
          {#if showSettings}
            <button
              type="button"
              aria-label="Close settings"
              class="fixed inset-0 z-10 cursor-default"
              onclick={() => (showSettings = false)}
            ></button>
            <div
              class="absolute left-0 top-full z-20 mt-1 w-56 space-y-3 rounded-md border border-border bg-card p-3 text-[11px] shadow-md"
            >
              <label class="block">
                <span class="mb-1 flex items-center justify-between text-muted-foreground">
                  Point size <span class="font-mono">{pointSize === 0 ? 'auto' : pointSize}</span>
                </span>
                <input
                  type="range"
                  min="0"
                  max="20"
                  step="0.5"
                  bind:value={pointSize}
                  class="w-full accent-primary"
                />
              </label>
              <label class="block">
                <span class="mb-1 flex items-center justify-between text-muted-foreground">
                  Dimmed opacity <span class="font-mono">{filterAlpha}</span>
                </span>
                <input
                  type="range"
                  min="0"
                  max="120"
                  step="1"
                  bind:value={filterAlpha}
                  class="w-full accent-primary"
                />
                <span class="mt-0.5 block text-[10px] text-muted-foreground/70">
                  How visible excluded points are — search-filtered or not in the
                  lasso/cluster selection (left = hidden).
                </span>
              </label>
            </div>
          {/if}
        </div>
      </div>

      <!-- legend / distribution (clickable → select) -->
      <AtlasLegend
        {legendMode}
        {clusterRows}
        {categoryRows}
        clusterStats={clusterStatsValue}
        categoryTitle={categoryTitleValue}
        {categoryTotal}
        {legendFiltered}
        hiddenClusters={crossFilter.hiddenClusters}
        {isDark}
        onPickCluster={pickCluster}
        onPickCategory={pickCategory}
        onToggleClusterHidden={(id) => crossFilter.toggleClusterHidden(id)}
        onShowAllClusters={() => crossFilter.showAllClusters()}
      />

      <!-- selection status + actions -->
      <div
        class="absolute bottom-3 left-3 flex items-center gap-2 rounded-md bg-card/85 px-2 py-1 text-[11px] shadow-sm backdrop-blur"
      >
        {#if tableLoading}
          <Loader2 class="size-3.5 animate-spin text-muted-foreground" />
        {/if}
        {#if selectionCount > 0}
          <span class="text-foreground">{selectionCount.toLocaleString()} selected</span>
          {#if selectionCount > 1000}
            <span class="text-muted-foreground/70">· table shows 1000</span>
          {/if}
          <Button
            variant="ghost"
            size="sm"
            onclick={seedSearch}
            title="Open this map selection as the full results list (paging, rerank, table/grid views)"
          >
            Show as results
          </Button>
          <Button variant="ghost" size="icon" onclick={clearSelection} title="Clear selection">
            <X class="size-3.5" />
          </Button>
        {:else}
          <Lasso class="size-3.5 text-muted-foreground" />
          <span class="text-muted-foreground"
            >drag to lasso · click a legend to select · click a point to play · scroll to zoom</span
          >
        {/if}
      </div>
    {/if}
  </div>
{/if}
