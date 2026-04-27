<script lang="ts">
  import HitCard from './hit-card.svelte';
  import type { Hit } from '$lib/api';

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

  const isActive = (h: Hit) =>
    !!active && active.doc_id === h.doc_id && active.chunk_id === h.chunk_id;
</script>

<div>
  {#if hits.length === 0}
    <div class="px-4 py-6 text-sm text-muted-foreground">{emptyMessage}</div>
  {:else}
    {#each hits as hit (hit.doc_id + ':' + hit.speech_id + ':' + hit.chunk_id)}
      <HitCard
        {hit}
        {query}
        active={isActive(hit)}
        onclick={() => onselect?.(hit)}
      />
    {/each}
  {/if}
</div>
