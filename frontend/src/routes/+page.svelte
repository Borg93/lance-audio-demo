<script lang="ts">
  import { onMount } from 'svelte';
  import { page } from '$app/state';
  import {
    search,
    listDocuments,
    ApiError,
    thumbnailUrl,
    type Hit,
    type SearchSpec,
    type Document,
  } from '$lib/api';
  import { fmtTime, hitKey, queryTerms, makeHighlighter } from '$lib/utils';
  import SearchBar from '$lib/components/search-bar.svelte';
  import ActiveFilters from '$lib/components/active-filters.svelte';
  import HitList from '$lib/components/hit-list.svelte';
  import HitCard from '$lib/components/hit-card.svelte';
  import HitTable, { TABLE_COLUMNS } from '$lib/components/hit-table.svelte';
  import DocTile from '$lib/components/doc-tile.svelte';
  import PlayerPane from '$lib/components/player-pane.svelte';
  import ResizableSplit from '$lib/components/resizable-split.svelte';
  import AtlasMap from '$lib/atlas/AtlasMap.svelte';
  import { crossFilter } from '$lib/atlas/cross-filter.svelte';
  import { getAtlasChunks } from '$lib/api';
  import { Button } from '$lib/components/ui';
  import {
    LayoutGrid,
    List as ListIcon,
    Table as TableIcon,
    Map as MapIcon,
    Loader2,
    ChevronLeft,
    ChevronRight,
    SearchX,
    Minus,
    Plus,
  } from 'lucide-svelte';

  const PAGE_STEP = 30;
  // Default result count shown per search (the Settings "Results to return").
  let spec = $state<SearchSpec>({ q: '', n: 100, mode: 'fts' });

  // ── Search results (when there's a query or image) ──
  let hits = $state<Hit[]>([]);
  let active = $state<Hit | null>(null);
  const activeKey = $derived(active ? hitKey(active) : null);
  // Compile the query → highlighter ONCE for the grid view, then pass the
  // prebuilt fn to every tile (the grid renders HitCard directly, bypassing
  // hit-list). This avoids one RegExp compilation per card across large pages.
  const gridHighlight = $derived(makeHighlighter(queryTerms(spec.q)));
  let loadingHits = $state(false);
  let loadingMore = $state(false);
  let error = $state<string | null>(null);
  /** True once a page returns fewer hits than requested → backend exhausted. */
  let allLoaded = $state(false);

  /** Distinct document count + per-document hit breakdown for the current results. */
  const docCount = $derived(new Set(hits.map((h) => h.doc_id)).size);

  // ── Document browse (when query is empty) ──
  const PER_PAGE = 24;
  let docs = $state<Document[]>([]);
  let docsTotal = $state(0);
  let docsPage = $state(1);
  let loadingDocs = $state(false);

  // ── Layout ──
  let view = $state<'list' | 'grid' | 'table' | 'map'>('list');

  // ── Atlas map cross-filter glue ──
  // The map's lasso/box/cluster selection surfaces its hits here; the bottom
  // results table + player read this set while the Map view is active.
  let mapHits = $state<Hit[]>([]);
  let mapSelectionTotal = $state(0);
  const MAP_TABLE_COLS = ['thumbnail', 'score', 'namn', 'language', 'start', 'text'];

  // The Map-view table shows the lasso/legend selection when there is one,
  // otherwise the active search hits (text or image search) — which are also
  // highlighted on the map. So searching populates the table too, not only map
  // selections; a lasso then narrows it to the picked region.
  const mapTableHits = $derived(mapSelectionTotal > 0 ? mapHits : hits);

  // DIRECTION A — search/filter → dim+highlight the map. When the map is shown,
  // project the current search hits onto point indices so matches light up and
  // the rest dims (a single GPU recolour). Cleared to whole-corpus when empty.
  // Depends on `keyToIndex` too, so a search that ran BEFORE the map loaded its
  // points re-projects once AtlasMap rebuilds the key→index map for the space.
  $effect(() => {
    if (view !== 'map') return;
    const ready = crossFilter.keyToIndex.size > 0; // touch for reactivity
    if (hits.length > 0 && ready) crossFilter.setFilteredFromHits(hits);
    else crossFilter.clearFilter();
  });

  function onMapSelectionHits(h: Hit[], total: number) {
    mapHits = h;
    mapSelectionTotal = total;
  }

  // DIRECTION B (seed) — promote a map selection to the search's result set by
  // stable _rowid (the points' addresses), not a synthesized giant WHERE.
  async function seedSearchFromSelection(rowids: number[]) {
    loadingHits = true;
    error = null;
    try {
      hits = await getAtlasChunks(rowids);
      active = null;
      allLoaded = true;
      view = 'list';
    } catch (e) {
      error = e instanceof Error ? e.message : 'failed to load selection';
    } finally {
      loadingHits = false;
    }
  }
  // Which result columns the table view shows (persisted). Defaults to a
  // readable subset; the chooser bar toggles any of TABLE_COLUMNS.
  let tableCols = $state<string[]>([
    'thumbnail',
    'score',
    'namn',
    'start',
    'end',
    'duration',
    'text',
    'caption',
  ]);

  // Columns shown in the browse-mode (pre-search) documents table. Documents
  // carry fewer fields than search hits, so this is its own small set.
  const DOC_COLUMNS: { key: string; label: string; get: (d: Document) => string }[] = [
    { key: 'thumbnail', label: 'Thumb', get: () => '' },
    { key: 'namn', label: 'Name', get: (d) => d.namn ?? '' },
    { key: 'referenskod', label: 'Ref', get: (d) => d.referenskod ?? '' },
    { key: 'bildid', label: 'Bild ID', get: (d) => d.bildid ?? '' },
    { key: 'extraid', label: 'Internal ID', get: (d) => d.extraid ?? '' },
    { key: 'duration', label: 'Dur', get: (d) => (d.duration != null ? fmtTime(d.duration) : '') },
    { key: 'audio_path', label: 'File', get: (d) => d.audio_path },
  ];
  $effect(() => {
    if (typeof localStorage === 'undefined') return;
    const v = localStorage.getItem('raudio-table-cols-v4');
    if (v) {
      try {
        tableCols = JSON.parse(v) as string[];
      } catch {
        /* ignore */
      }
    }
  });
  function toggleCol(key: string) {
    tableCols = tableCols.includes(key) ? tableCols.filter((k) => k !== key) : [...tableCols, key];
    try {
      localStorage.setItem('raudio-table-cols-v4', JSON.stringify(tableCols));
    } catch {
      /* ignore */
    }
  }
  // Grid column count, persisted in localStorage so it sticks.
  let gridCols = $state<number>(3);
  $effect(() => {
    if (typeof localStorage === 'undefined') return;
    const v = localStorage.getItem('raudio-gridcols');
    if (v) gridCols = Math.max(2, Math.min(6, Number(v) || 3));
  });
  function setGridCols(n: number) {
    gridCols = n;
    try {
      localStorage.setItem('raudio-gridcols', String(n));
    } catch {}
  }

  /** Selected document in browse mode (so the grid can highlight it). */
  const activeDocId = $derived(active?.doc_id ?? null);

  const isBrowsing = $derived(!spec.q && !spec.image && !spec.topic);
  const docsTotalPages = $derived(Math.max(1, Math.ceil(docsTotal / PER_PAGE)));

  async function runSearch(next: SearchSpec) {
    // Reset paging on a new search.
    spec = { ...next, n: next.n ?? PAGE_STEP };
    allLoaded = false;
    if (!spec.q && !spec.image && !spec.topic) {
      hits = [];
      return;
    }
    // A new search supersedes any prior map selection so the Map-view table +
    // dimming reflect the fresh hits, not a stale lasso (mapTableHits is
    // `mapSelectionTotal > 0 ? mapHits : hits`). A later lasso re-narrows.
    crossFilter.clearSelection();
    mapHits = [];
    mapSelectionTotal = 0;
    loadingHits = true;
    error = null;
    try {
      const requested = spec.n ?? PAGE_STEP;
      hits = await search(spec);
      active = null;
      if (hits.length < requested) allLoaded = true;
    } catch (e) {
      hits = [];
      error = e instanceof ApiError ? e.detail : e instanceof Error ? e.message : 'unknown error';
    } finally {
      loadingHits = false;
    }
  }

  /** Load PAGE_STEP more hits by re-running the search with a larger limit. */
  async function loadMore() {
    if (loadingMore || allLoaded) return;
    loadingMore = true;
    try {
      const nextN = (spec.n ?? PAGE_STEP) + PAGE_STEP;
      hits = await search({ ...spec, n: nextN });
      spec = { ...spec, n: nextN };
      if (hits.length < nextN) allLoaded = true;
    } catch {
      // Silent on load-more errors — keep the hits already shown.
    } finally {
      loadingMore = false;
    }
  }

  /** Load documents whenever in browse mode + page changes. */
  $effect(() => {
    if (!isBrowsing) return;
    loadingDocs = true;
    listDocuments(docsPage, PER_PAGE)
      .then((r) => {
        docs = r.docs;
        docsTotal = r.total;
      })
      .catch(() => {
        docs = [];
        docsTotal = 0;
      })
      .finally(() => (loadingDocs = false));
  });

  /**
   * Global keyboard affordances:
   *   "/"    focus the search box (unless already typing in a field)
   *   Esc    return from a playing hit back to the results list
   */
  function onWindowKeydown(e: KeyboardEvent) {
    const t = e.target as HTMLElement | null;
    const typing =
      !!t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable);
    if (e.key === '/' && !typing) {
      e.preventDefault();
      const el = document.querySelector<HTMLInputElement>('input[type="search"]');
      el?.focus();
      el?.select();
    } else if (e.key === 'Escape' && active && !typing) {
      active = null;
    }
  }

  /**
   * When user clicks a doc tile, just open it in the player with a synthetic
   * full-document "hit" pointing at start=0. Earlier I auto-set the `namn`
   * filter — it then silently stuck across the next search and produced 0
   * hits. Now nothing is implicitly filtered.
   */
  // Hand-off from the Tree page: `/?topic=<name>` opens Search already filtered
  // to that topic — a filter-only browse (no query text), backed by the
  // topic-only branch in `run_search`. Runs once on mount.
  onMount(() => {
    const topic = page.url.searchParams.get('topic');
    if (topic) runSearch({ q: '', topic, n: spec.n ?? 100, mode: spec.mode });
  });

  function openDoc(doc: Document) {
    active = {
      _score: 0,
      doc_id: doc.doc_id,
      audio_path: doc.audio_path,
      speech_id: 0,
      chunk_id: 0,
      start: 0,
      end: doc.duration ?? 0,
      duration: doc.duration ?? null,
      text: '',
      language: null,
      namn: doc.namn ?? null,
      referenskod: doc.referenskod ?? null,
      bildid: doc.bildid ?? null,
      extraid: doc.extraid ?? null,
      alignments: [],
    };
  }
</script>

<svelte:window onkeydown={onWindowKeydown} />

<div class="grid h-full grid-rows-[auto_1fr] min-h-0">
  <div class="border-b border-border bg-card/40">
    <SearchBar bind:spec onsubmit={runSearch} />
    <ActiveFilters bind:spec onchange={runSearch} />
  </div>

  <ResizableSplit minLeft={420} minRight={360} initial={0.6}>
    {#snippet left()}
      <!-- ── Left: results ── -->
      <div class="flex h-full min-h-0 flex-col">
        <!-- Results header: count + view-mode toggle -->
        <div class="flex items-center gap-3 border-b border-border bg-card/30 px-4 py-2 text-xs">
          <span class="text-muted-foreground">
            {#if isBrowsing}
              Browsing {docsTotal} document{docsTotal === 1 ? '' : 's'}
            {:else if loadingHits}
              Searching…
            {:else if error}
              <span class="text-destructive">Error: {error}</span>
            {:else if hits.length === 0}
              No hits.
            {:else}
              <strong class="text-foreground">{hits.length}</strong>
              {hits.length === 1 ? 'chunk' : 'chunks'}
              across
              <strong class="text-foreground">{docCount}</strong>
              {docCount === 1 ? 'document' : 'documents'}
              {#if allLoaded}<span class="text-muted-foreground/70">· all results shown</span>{/if}
            {/if}
          </span>

          <div class="ml-auto flex items-center gap-2">
            {#if view === 'grid'}
              <div class="flex items-center gap-1 border-r border-border pr-2 mr-1">
                <span class="text-muted-foreground/70 mr-1">cols</span>
                <Button
                  variant="ghost"
                  size="icon"
                  disabled={gridCols <= 2}
                  title="Fewer columns"
                  onclick={() => setGridCols(Math.max(2, gridCols - 1))}
                >
                  <Minus class="size-3.5" />
                </Button>
                <span class="w-4 text-center font-mono text-[11px]">{gridCols}</span>
                <Button
                  variant="ghost"
                  size="icon"
                  disabled={gridCols >= 6}
                  title="More columns"
                  onclick={() => setGridCols(Math.min(6, gridCols + 1))}
                >
                  <Plus class="size-3.5" />
                </Button>
              </div>
            {/if}
            <Button
              variant={view === 'list' ? 'secondary' : 'ghost'}
              size="icon"
              title="List view"
              onclick={() => (view = 'list')}
            >
              <ListIcon class="size-4" />
            </Button>
            <Button
              variant={view === 'grid' ? 'secondary' : 'ghost'}
              size="icon"
              title="Grid view"
              onclick={() => (view = 'grid')}
            >
              <LayoutGrid class="size-4" />
            </Button>
            <Button
              variant={view === 'table' ? 'secondary' : 'ghost'}
              size="icon"
              title="Table view — see column values per row"
              onclick={() => (view = 'table')}
            >
              <TableIcon class="size-4" />
            </Button>
            <Button
              variant={view === 'map' ? 'secondary' : 'ghost'}
              size="icon"
              title="Map view — the EVōC embedding atlas (cross-filters with search)"
              onclick={() => (view = 'map')}
            >
              <MapIcon class="size-4" />
            </Button>
          </div>
        </div>

        {#if view === 'map'}
          <!-- ── Map view: the embedding atlas + a draggable selection/results table ── -->
          <div class="min-h-0 flex-1">
            <ResizableSplit
              orientation="vertical"
              storageKey="raudio-search-map-vsplit"
              minLeft={220}
              minRight={120}
              initial={0.66}
            >
              {#snippet left()}
                <AtlasMap
                  bind:active
                  onSeedSearch={seedSearchFromSelection}
                  onSelectionHits={onMapSelectionHits}
                />
              {/snippet}
              {#snippet right()}
                <div class="flex h-full min-h-0 flex-col border-t border-border bg-card/30">
                  <div class="flex items-center gap-2 border-b border-border px-3 py-1.5 text-xs">
                    {#if mapSelectionTotal > 0}
                      <span class="font-medium text-foreground">Map selection</span>
                      <span class="text-muted-foreground">
                        {mapSelectionTotal.toLocaleString()} chunks
                        {#if mapSelectionTotal > mapHits.length}
                          <span class="text-muted-foreground/70">· showing {mapHits.length}</span>
                        {/if}
                      </span>
                    {:else if hits.length > 0}
                      <span class="font-medium text-foreground">Search results</span>
                      <span class="text-muted-foreground">
                        {hits.length.toLocaleString()} hits · highlighted on the map
                      </span>
                    {:else}
                      <span class="font-medium text-foreground">Selection</span>
                      <span class="text-muted-foreground"
                        >lasso a region, click a legend, or search to list chunks</span
                      >
                    {/if}
                  </div>
                  <div class="min-h-0 flex-1 overflow-auto">
                    {#if mapTableHits.length}
                      <HitTable
                        hits={mapTableHits}
                        {active}
                        visible={MAP_TABLE_COLS}
                        query={mapSelectionTotal > 0 ? '' : spec.q}
                        onselect={(h) => (active = h)}
                      />
                    {/if}
                  </div>
                </div>
              {/snippet}
            </ResizableSplit>
          </div>
        {:else}
          <div class="relative min-h-0 flex-1 overflow-y-auto">
            {#if isBrowsing}
              <!-- ── Browse mode: documents ── -->
              {#if loadingDocs && docs.length === 0}
                <div class="flex h-full items-center justify-center text-sm text-muted-foreground">
                  <Loader2 class="size-4 animate-spin mr-2" /> Loading documents…
                </div>
              {:else if view === 'grid'}
                <div
                  class="grid gap-4 p-4"
                  style:grid-template-columns="repeat({gridCols}, minmax(0, 1fr))"
                >
                  {#each docs as doc (doc.doc_id)}
                    <DocTile
                      {doc}
                      active={activeDocId === doc.doc_id}
                      onclick={() => openDoc(doc)}
                    />
                  {/each}
                </div>
              {:else if view === 'table'}
                {@const docCols = DOC_COLUMNS.filter((c) => tableCols.includes(c.key))}
                <div
                  class="flex flex-wrap items-center gap-1 border-b border-border bg-card/30 px-3 py-2 text-[11px]"
                >
                  <span class="mr-1 text-muted-foreground">Columns:</span>
                  {#each DOC_COLUMNS as c (c.key)}
                    <button
                      type="button"
                      onclick={() => toggleCol(c.key)}
                      class={'rounded border px-1.5 py-0.5 transition-colors ' +
                        (tableCols.includes(c.key)
                          ? 'border-primary bg-primary/10 text-foreground'
                          : 'border-border text-muted-foreground hover:text-foreground')}
                    >
                      {c.label}
                    </button>
                  {/each}
                </div>
                <div class="overflow-x-auto">
                  <table class="w-full border-collapse text-xs">
                    <thead>
                      <tr
                        class="sticky top-0 z-10 border-b border-border bg-card text-left text-muted-foreground"
                      >
                        {#each docCols as c (c.key)}
                          <th class="px-3 py-2 font-medium whitespace-nowrap">{c.label}</th>
                        {/each}
                      </tr>
                    </thead>
                    <tbody>
                      {#each docs as doc (doc.doc_id)}
                        <tr
                          class={'cursor-pointer border-b border-border/60 hover:bg-secondary/40 ' +
                            (activeDocId === doc.doc_id
                              ? 'bg-primary/15 font-medium [box-shadow:inset_3px_0_0_0_var(--color-primary)]'
                              : '')}
                          onclick={() => openDoc(doc)}
                        >
                          {#each docCols as c (c.key)}
                            {#if c.key === 'thumbnail'}
                              <td class="px-3 py-1.5 align-top">
                                <img
                                  src={thumbnailUrl(doc.doc_id)}
                                  loading="lazy"
                                  alt=""
                                  class="h-9 w-16 rounded bg-muted object-cover"
                                  onerror={(e) =>
                                    ((e.currentTarget as HTMLImageElement).style.visibility =
                                      'hidden')}
                                />
                              </td>
                            {:else}
                              <td
                                class="max-w-[28rem] truncate px-3 py-1.5 align-top whitespace-nowrap text-muted-foreground"
                                title={c.get(doc)}
                              >
                                {c.get(doc)}
                              </td>
                            {/if}
                          {/each}
                        </tr>
                      {/each}
                    </tbody>
                  </table>
                </div>
              {:else}
                <ul class="divide-y divide-border">
                  {#each docs as doc (doc.doc_id)}
                    <li>
                      <button
                        type="button"
                        onclick={() => openDoc(doc)}
                        class="flex w-full items-center gap-3 px-4 py-2 text-left hover:bg-secondary/40"
                      >
                        <span class="flex-1 truncate text-sm">{doc.namn ?? doc.audio_path}</span>
                        <span class="font-mono text-[11px] text-muted-foreground">
                          {doc.referenskod ?? ''}
                        </span>
                      </button>
                    </li>
                  {/each}
                </ul>
              {/if}

              {#if docsTotalPages > 1}
                <div
                  class="sticky bottom-0 flex items-center justify-end gap-1 border-t border-border bg-card/80 px-4 py-2 text-xs backdrop-blur"
                >
                  <span class="mr-2 text-muted-foreground">page {docsPage} / {docsTotalPages}</span>
                  <Button
                    variant="outline"
                    size="icon"
                    disabled={docsPage <= 1}
                    onclick={() => (docsPage = Math.max(1, docsPage - 1))}
                  >
                    <ChevronLeft class="size-4" />
                  </Button>
                  <Button
                    variant="outline"
                    size="icon"
                    disabled={docsPage >= docsTotalPages}
                    onclick={() => (docsPage = Math.min(docsTotalPages, docsPage + 1))}
                  >
                    <ChevronRight class="size-4" />
                  </Button>
                </div>
              {/if}
            {:else if loadingHits}
              <div class="flex h-full items-center justify-center text-sm text-muted-foreground">
                <Loader2 class="size-4 animate-spin mr-2" /> Searching…
              </div>
            {:else if hits.length === 0}
              <div
                class="flex h-full flex-col items-center justify-center gap-2 px-6 text-center text-sm text-muted-foreground"
              >
                <SearchX class="size-6 text-muted-foreground/60" />
                <div>No hits.</div>
                <div class="text-xs">
                  Try toggling <strong>Match by</strong> to <em>Semantic</em> or switching
                  <strong>Style</strong>
                  to <em>Fuzzy</em>.
                </div>
              </div>
            {:else if view === 'grid'}
              <div
                class="grid gap-3 p-3"
                style:grid-template-columns="repeat({gridCols}, minmax(0, 1fr))"
              >
                {#each hits as hit (hitKey(hit))}
                  <HitCard
                    {hit}
                    query={spec.q}
                    highlight={gridHighlight}
                    active={activeKey === hitKey(hit)}
                    layout="tile"
                    onclick={() => (active = hit)}
                  />
                {/each}
              </div>
            {:else if view === 'table'}
              <!-- Column chooser: click a column to show/hide it in the table. -->
              <div
                class="flex flex-wrap items-center gap-1 border-b border-border bg-card/30 px-3 py-2 text-[11px]"
              >
                <span class="mr-1 text-muted-foreground">Columns:</span>
                {#each TABLE_COLUMNS as c (c.key)}
                  <button
                    type="button"
                    onclick={() => toggleCol(c.key)}
                    class={'rounded border px-1.5 py-0.5 transition-colors ' +
                      (tableCols.includes(c.key)
                        ? 'border-primary bg-primary/10 text-foreground'
                        : 'border-border text-muted-foreground hover:text-foreground')}
                  >
                    {c.label}
                  </button>
                {/each}
              </div>
              <HitTable
                {hits}
                {active}
                visible={tableCols}
                query={spec.q}
                onselect={(h) => (active = h)}
              />
            {:else}
              <HitList {hits} query={spec.q} {active} onselect={(h) => (active = h)} />
            {/if}

            <!-- Pagination: only when we have hits and aren't already exhausted. -->
            {#if !isBrowsing && hits.length > 0 && !allLoaded}
              <div class="flex justify-center border-t border-border bg-card/40 px-4 py-3">
                <Button variant="outline" size="sm" disabled={loadingMore} onclick={loadMore}>
                  {loadingMore ? 'Loading…' : `Show ${PAGE_STEP} more`}
                </Button>
              </div>
            {/if}
          </div>
        {/if}
      </div>
    {/snippet}

    {#snippet right()}
      <!-- ── Right: player pane ── -->
      <div class="h-full min-h-0 bg-muted/30">
        <PlayerPane hit={active} query={spec.q} />
      </div>
    {/snippet}
  </ResizableSplit>
</div>
