<script lang="ts">
  /** Atlas selection source. Reads the crossFilter singleton (the map selection
   *  made on /atlas), converts selectedIds (point indices) back to pseudo-hits via
   *  the keyToIndex reverse map, and emits them as a result set. The node UI shows
   *  the selection count + a hint (select a region on the Atlas, then Run). */
  import { Handle, Position, type NodeProps } from '@xyflow/svelte';
  import { graph } from '$lib/workflow/graph.svelte';
  import { crossFilter } from '$lib/atlas/cross-filter.svelte';
  import NodeShell from './NodeShell.svelte';

  let { id, selected }: NodeProps = $props();
  const cfg = $derived(graph.config[id]);
  const rt = $derived(graph.runtime[id]);
  const selectionCount = $derived(crossFilter.selectedIds.size);
</script>

{#if cfg && rt}
  <NodeShell {id} title="Atlas selection" status={rt.status} {selected}>
    <div class="flex flex-col gap-2">
      <p class="text-sm font-medium text-foreground">{selectionCount} points selected</p>
      {#if selectionCount === 0}
        <p class="text-[10px] text-muted-foreground">
          Select a region on the <span class="text-amber-500">Atlas</span> page, then
          <span class="text-foreground">Run</span> to emit the selection into the workflow.
        </p>
      {:else}
        <p class="text-[10px] text-emerald-500">
          Selection ready. Wire into Search to refine within it.
        </p>
      {/if}
    </div>
    <Handle type="source" position={Position.Right} />
  </NodeShell>
{/if}
