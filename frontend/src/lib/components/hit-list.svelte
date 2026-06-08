<script lang="ts">
  import HitCard from './hit-card.svelte';
  import type { Hit } from '$lib/api';
  import { hitKey } from '$lib/utils';

  type Props = {
    hits: Hit[];
    query?: string;
    active?: Hit | null;
    onselect?: (hit: Hit) => void;
    /** Status text shown when `hits` is empty (e.g. "Searching…", "No hits."). */
    emptyMessage?: string;
  };
  let {
    hits,
    query = '',
    active = null,
    onselect,
    emptyMessage = 'Enter a query above.',
  }: Props = $props();

  const activeKey = $derived(active ? hitKey(active) : null);
</script>

<div>
  {#if hits.length === 0}
    <div class="px-4 py-6 text-sm text-muted-foreground">{emptyMessage}</div>
  {:else}
    {#each hits as hit (hitKey(hit))}
      <HitCard {hit} {query} active={activeKey === hitKey(hit)} onclick={() => onselect?.(hit)} />
    {/each}
  {/if}
</div>
