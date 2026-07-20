<script lang="ts">
  // Searchable annotation list — the review queue. Predictions first, highest
  // uncertainty first (the active-learning order), then click to select on canvas.
  import { Search } from 'lucide-svelte';
  import { Input } from '$lib/components/ui';
  import { cn } from '$lib/utils';
  import { statusDot } from './statusStyle';
  import type { AnnotatorController } from '../annotator.svelte';

  let { controller }: { controller: AnnotatorController } = $props();

  let filter = $state('');

  // The review order lives on the controller (shared with accept-and-advance); the
  // list only adds its text filter on top.
  const queue = $derived(
    controller.reviewQueue.filter((r) => {
      const q = filter.trim().toLowerCase();
      if (!q) return true;
      return (r.text + ' ' + r.label + ' ' + r.group).toLowerCase().includes(q);
    }),
  );
</script>

<div class="flex min-h-0 flex-1 flex-col" data-testid="annotation-list">
  <div class="relative px-3 py-2">
    <Search class="pointer-events-none absolute left-5 top-1/2 size-3.5 -translate-y-1/2 text-muted-foreground" />
    <Input bind:value={filter} placeholder="Filter annotations…" class="pl-7" />
  </div>

  <ul class="min-h-0 flex-1 overflow-y-auto px-1.5 pb-2">
    {#each queue as r (r.index)}
      <li>
        <button
          class={cn(
            'flex w-full items-start gap-2 rounded px-2 py-1.5 text-left hover:bg-muted/60',
            controller.selectedIndex === r.index && 'bg-primary/10 ring-1 ring-primary/40',
          )}
          onclick={() => controller.select(r.index)}
        >
          <span class={cn('mt-1 size-2 shrink-0 rounded-full', statusDot(r.status))}></span>
          <span class="min-w-0 flex-1">
            <span class="flex items-center justify-between gap-2">
              <span class="truncate text-xs font-medium">{r.label || `#${r.index}`}</span>
              {#if r.uncertainty != null}
                <span class="shrink-0 text-[10px] tabular-nums text-muted-foreground" title="uncertainty">
                  {r.uncertainty.toFixed(2)}
                </span>
              {/if}
            </span>
            {#if r.text}
              <span class="block truncate text-[11px] text-muted-foreground">{r.text}</span>
            {/if}
          </span>
        </button>
      </li>
    {:else}
      <li class="px-3 py-6 text-center text-xs text-muted-foreground">No annotations</li>
    {/each}
  </ul>
</div>
