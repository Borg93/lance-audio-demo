<script lang="ts">
  // Right inspector pane. Composes the review list / single-annotation detail
  // + the layer panel. Controlled by the facade. (Ported from ra-anno, split into
  // focused sub-components per our no-god-files rule.)
  import { ChevronLeft } from 'lucide-svelte';
  import { Button } from '$lib/components/ui';
  import { cn } from '$lib/utils';
  import { statusDot } from './statusStyle';
  import AnnotationDetail from './AnnotationDetail.svelte';
  import AnnotationList from './AnnotationList.svelte';
  import LayerPanel from './LayerPanel.svelte';
  import type { AnnotatorController } from '../annotator.svelte';

  let { controller }: { controller: AnnotatorController } = $props();

  const summary = $derived.by(() => {
    const counts = new Map<string, number>();
    for (const r of controller.rows) counts.set(r.status, (counts.get(r.status) ?? 0) + 1);
    return [...counts.entries()].sort((a, b) => a[0].localeCompare(b[0]));
  });
</script>

<aside
  class="flex h-full w-full min-w-0 flex-col border-l border-border bg-card"
  data-testid="annotation-sidebar"
>
  <header class="flex items-center justify-between border-b border-border px-3 py-2">
    <h2 class="text-sm font-semibold">Review queue</h2>
    <span class="text-xs tabular-nums text-muted-foreground">{controller.count}</span>
  </header>

  <!-- status summary -->
  <div class="flex flex-wrap gap-1.5 border-b border-border px-3 py-2">
    {#each summary as [status, n] (status)}
      <span class="flex items-center gap-1 text-[11px] text-muted-foreground">
        <span class={cn('size-2 rounded-full', statusDot(status))}></span>
        {status || '—'} <span class="tabular-nums">{n}</span>
      </span>
    {/each}
  </div>

  <div class="flex min-h-0 flex-1 flex-col">
    {#if controller.selected}
      <div class="border-b border-border px-2 py-1.5">
        <Button variant="ghost" size="sm" onclick={() => controller.select(null)}>
          <ChevronLeft class="size-3.5" /> Back to list
        </Button>
      </div>
      <div class="min-h-0 flex-1 overflow-y-auto">
        <AnnotationDetail {controller} />
      </div>
    {:else}
      <AnnotationList {controller} />
    {/if}
  </div>

  <LayerPanel {controller} />
</aside>
