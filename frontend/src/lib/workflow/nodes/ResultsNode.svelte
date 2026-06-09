<script lang="ts">
  /** Sink node: renders the Hit[] handed down its incoming edge. Click a row to
   *  play it in the Inspector. */
  import { Handle, Position, NodeResizer, type NodeProps } from '@xyflow/svelte';
  import { graph } from '$lib/workflow/graph.svelte';
  import HitList from '$lib/workflow/HitList.svelte';
  import NodeShell from './NodeShell.svelte';

  let { id, selected }: NodeProps = $props();
  const rt = $derived(graph.runtime[id]);
  const hits = $derived(rt?.hits ?? []);

  // Resize floor (px) — shared by the wrapper min-size and the NodeResizer so
  // they can't drift out of sync.
  const MIN_W = 320;
  const MIN_H = 144;
</script>

{#if rt}
  <div class="relative h-full w-full" style="min-width: {MIN_W}px; min-height: {MIN_H}px;">
    <NodeResizer minWidth={MIN_W} minHeight={MIN_H} isVisible={selected} />
    <NodeShell {id} title="Results" status={rt.status} {selected} width="h-full w-full">
      <Handle type="target" position={Position.Left} />
      {#if rt.status === 'idle'}
        <p class="text-[11px] text-muted-foreground">Run the graph to see hits here.</p>
      {:else if hits.length === 0}
        <p class="text-[11px] text-muted-foreground">No results.</p>
      {:else}
        <div class="mb-1.5 text-[10px] text-muted-foreground">
          <span class="text-foreground">{hits.length}</span> results · click to play
        </div>
        <HitList {hits} />
      {/if}
    </NodeShell>
  </div>
{/if}
