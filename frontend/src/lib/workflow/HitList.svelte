<script lang="ts">
  /** A clickable list of result hits (thumbnail + name + snippet). Clicking a
   *  row plays it in the Inspector. Shared by the Results node and the
   *  Inspector so both render results identically. */
  import { Play } from 'lucide-svelte';
  import { chunkFrameUrl, type Hit } from '$lib/api';
  import { graph } from '$lib/workflow/graph.svelte';
  import { hitKey } from '$lib/utils';

  let { hits, maxHeight = 'max-h-72' }: { hits: Hit[]; maxHeight?: string } = $props();

  const selectedKey = $derived(graph.selectedHit ? hitKey(graph.selectedHit) : null);
</script>

<div class="nodrag nowheel flex {maxHeight} flex-col gap-1.5 overflow-y-auto pr-1">
  {#each hits as h (hitKey(h))}
    {@const isSel = selectedKey === hitKey(h)}
    <button
      type="button"
      onclick={(e) => {
        // Don't let the click bubble to the node wrapper (would fire
        // onnodeclick → inspectNode and clear the hit we just selected).
        e.stopPropagation();
        graph.selectHit(h);
      }}
      class="group flex w-full gap-2 rounded border bg-background p-1.5 text-left transition-colors hover:bg-muted"
      class:border-primary={isSel}
      class:border-border={!isSel}
    >
      <div class="relative shrink-0">
        <img
          src={chunkFrameUrl(h.doc_id, h.speech_id, h.chunk_id)}
          alt=""
          loading="lazy"
          class="h-10 w-14 rounded bg-muted object-cover"
          onerror={(e) => {
            // Many transcript chunks have no extracted frame → 404. Swap to a
            // transparent pixel once so it shows a clean muted box, not a
            // broken-image glyph (and never re-requests).
            const img = e.currentTarget as HTMLImageElement;
            if (img.dataset.fallback) return;
            img.dataset.fallback = '1';
            img.src =
              'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7';
          }}
        />
        <span
          class="pointer-events-none absolute inset-0 grid place-items-center rounded bg-black/45 opacity-0 transition-opacity group-hover:opacity-100"
        >
          <Play class="size-4 text-white" />
        </span>
      </div>
      <div class="min-w-0">
        {#if h.namn}
          <div class="truncate text-[10px] font-medium text-foreground" title={h.namn}>
            {h.namn}
          </div>
        {/if}
        <div class="line-clamp-2 text-[10px] text-muted-foreground">{h.text}</div>
      </div>
    </button>
  {/each}
</div>
