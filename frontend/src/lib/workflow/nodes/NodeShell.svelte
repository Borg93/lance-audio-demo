<script lang="ts">
  /** Shared card chrome for every workflow node: a titled surface with a
   *  run-status dot and a hover-to-reveal delete (✕) button. */
  import type { Snippet } from 'svelte';
  import { X } from 'lucide-svelte';
  import { graph, STATUS_DOT, type RunStatus } from '$lib/workflow/graph.svelte';

  let {
    id,
    title,
    status = 'idle',
    selected = false,
    width = 'w-64',
    children,
  }: {
    id: string;
    title: string;
    status?: RunStatus;
    selected?: boolean;
    width?: string;
    children: Snippet;
  } = $props();

</script>

<div
  class="group {width} rounded-lg border border-border bg-card shadow-sm transition-shadow"
  class:ring-2={selected}
  class:ring-primary={selected}
>
  <div class="flex items-center gap-2 border-b border-border px-3 py-1.5">
    <span class="size-2 shrink-0 rounded-full {STATUS_DOT[status]}"></span>
    <span class="min-w-0 flex-1 truncate text-xs font-semibold text-foreground">{title}</span>
    <button
      type="button"
      onclick={(e) => {
        e.stopPropagation();
        graph.removeNode(id);
      }}
      title="Delete this node"
      aria-label="Delete node"
      class="nodrag shrink-0 rounded p-0.5 text-muted-foreground/40 opacity-0 transition-opacity group-hover:opacity-100 hover:bg-destructive/15 hover:text-destructive"
    >
      <X class="size-3.5" />
    </button>
  </div>
  <div class="px-3 py-2 text-xs text-foreground">
    {@render children()}
  </div>
</div>
