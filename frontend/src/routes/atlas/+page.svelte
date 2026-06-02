<script lang="ts">
  /**
   * Atlas tab — an Embedding Atlas map over the Lance `chunks` table.
   *
   * Client-only (SPA, no SSR): we check `/api/atlas/status` first, and only
   * dynamically import the heavy viewer (`mount-atlas.svelte` → DuckDB-WASM +
   * Mosaic) when the projection actually exists. If it doesn't, we point the
   * user at the offline build step instead of failing.
   */
  import { browser } from '$app/environment';
  import type { Component } from 'svelte';
  import { getAtlasStatus } from '$lib/api';

  type Phase = 'loading' | 'ready' | 'unavailable' | 'error';

  let phase = $state<Phase>('loading');
  let errorMsg = $state<string | null>(null);
  let Mount = $state.raw<Component | null>(null);

  $effect(() => {
    if (!browser) return;
    let cancelled = false;

    (async () => {
      try {
        const status = await getAtlasStatus();
        if (cancelled) return;
        if (!status.projected) {
          phase = 'unavailable';
          return;
        }
        const module = await import('$lib/atlas/mount-atlas.svelte');
        if (cancelled) return;
        Mount = module.default;
        phase = 'ready';
      } catch (e) {
        if (!cancelled) {
          errorMsg = e instanceof Error ? e.message : String(e);
          phase = 'error';
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  });
</script>

<div class="h-full w-full">
  {#if phase === 'unavailable'}
    <div class="grid h-full place-items-center p-6 text-center text-sm text-muted-foreground">
      <div>
        <p class="mb-1 font-medium text-foreground">No embedding map yet</p>
        <p>
          Run <code class="rounded bg-muted px-1 py-0.5">raudio feature atlas</code> to build the 2-D
          projection of the chunks table.
        </p>
      </div>
    </div>
  {:else if phase === 'error'}
    <div class="grid h-full place-items-center p-6 text-center text-sm text-destructive">
      Failed to load Atlas: {errorMsg}
    </div>
  {:else if phase === 'ready' && Mount}
    <Mount />
  {:else}
    <div class="grid h-full place-items-center text-sm text-muted-foreground">Loading…</div>
  {/if}
</div>
