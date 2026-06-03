<script lang="ts">
  /**
   * Hover/click ANALYSIS popover for the atlas — what turns the map from a
   * viewer into a cluster-discovery + data-inspection tool.
   *
   * Cluster id, language, and namn come INSTANTLY from the point arrays; the
   * text snippet is lazy-fetched via `getAtlasChunk` on a ~150ms hover-settle
   * debounce and memoised in a tiny LRU keyed by point index. Absolutely
   * positioned (bg-card/85 backdrop-blur) to match the toolbar/legend overlays.
   */
  import { getAtlasChunk, type AtlasPoints } from '$lib/api';

  let {
    pts,
    index,
    x,
    y,
  }: {
    pts: AtlasPoints;
    /** Hovered point index, or null when nothing is hovered. */
    index: number | null;
    /** Screen position (px, relative to the map container) of the point. */
    x: number;
    y: number;
  } = $props();

  // ── instant fields from the arrays (no fetch) ─────────────────────────────
  const clusterId = $derived(index != null ? (pts.cluster?.[index] ?? null) : null);
  const language = $derived.by(() => {
    if (index == null) return null;
    const code = pts.language?.[index];
    return code != null ? (pts.languages?.[code] ?? null) : null;
  });
  const namn = $derived.by(() => {
    if (index == null) return null;
    const code = pts.namn?.[index];
    return code != null ? (pts.namns?.[code] ?? null) : null;
  });
  const topic = $derived.by(() => {
    if (index == null) return null;
    const code = pts.topic?.[index];
    const label = code != null ? (pts.topics?.[code] ?? null) : null;
    return label || null; // '' (unclustered) → null
  });

  function keyAt(i: number): [string, number, number] | null {
    const dc = pts.doc[i];
    const doc = dc !== undefined ? pts.docs[dc] : undefined;
    const s = pts.speech_id[i];
    const c = pts.chunk_id[i];
    if (doc === undefined || s === undefined || c === undefined) return null;
    return [doc, s, c];
  }

  // ── lazy text snippet (debounced fetch + LRU) ─────────────────────────────
  const LRU_MAX = 200;
  const cache = new Map<number, string>();
  let snippet = $state<string | null>(null);
  let snippetFor = $state<number | null>(null);

  function remember(i: number, text: string) {
    cache.delete(i);
    cache.set(i, text);
    if (cache.size > LRU_MAX) {
      const oldest = cache.keys().next().value;
      if (oldest !== undefined) cache.delete(oldest);
    }
  }

  $effect(() => {
    const i = index;
    if (i == null) {
      snippet = null;
      snippetFor = null;
      return;
    }
    const cached = cache.get(i);
    if (cached !== undefined) {
      snippet = cached;
      snippetFor = i;
      return;
    }
    snippet = null;
    snippetFor = null;
    const key = keyAt(i);
    if (!key) return;
    const timer = setTimeout(async () => {
      try {
        const hit = await getAtlasChunk(key[0], key[1], key[2]);
        const text = (hit.text || hit.caption || '').slice(0, 240);
        remember(i, text);
        // Only apply if still hovering the same point.
        if (index === i) {
          snippet = text;
          snippetFor = i;
        }
      } catch {
        /* snippet stays null — instant fields still show */
      }
    }, 150);
    return () => clearTimeout(timer);
  });
</script>

{#if index != null}
  <div
    class="pointer-events-none absolute z-20 w-64 max-w-[80vw] -translate-x-1/2 -translate-y-full rounded-md border border-border bg-card/85 p-2.5 text-[11px] shadow-md backdrop-blur"
    style:left="{x}px"
    style:top="{Math.max(8, y - 10)}px"
  >
    <div class="mb-1 flex items-center gap-2">
      {#if clusterId != null}
        <span class="rounded bg-secondary/70 px-1.5 py-0.5 font-mono text-foreground">
          {clusterId < 0 ? 'noise' : `cluster #${clusterId}`}
        </span>
      {/if}
      {#if language}
        <span class="rounded bg-secondary/40 px-1.5 py-0.5 uppercase text-muted-foreground">
          {language}
        </span>
      {/if}
    </div>
    {#if namn}
      <div class="mb-1 truncate font-medium text-foreground" title={namn}>{namn}</div>
    {/if}
    {#if snippet && snippetFor === index}
      <div class="line-clamp-3 text-muted-foreground">{snippet}</div>
    {:else}
      <div class="text-muted-foreground/60 italic">loading text…</div>
    {/if}
    <div class="mt-1 text-[10px] text-muted-foreground/60">click to play</div>
  </div>
{/if}
