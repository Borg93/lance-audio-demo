<script lang="ts">
  /** Sink node: collects the incoming Hit[] and downloads them as JSON/CSV with
   *  the columns chosen in the Inspector. Format + columns are configured there;
   *  the node itself shows a summary and a one-click Download. */
  import { Handle, Position, type NodeProps } from '@xyflow/svelte';
  import { Download } from 'lucide-svelte';
  import { graph } from '$lib/workflow/graph.svelte';
  import { exportHits } from '$lib/workflow/export';
  import NodeShell from './NodeShell.svelte';

  let { id, selected }: NodeProps = $props();
  const rt = $derived(graph.runtime[id]);
  const cfg = $derived(graph.config[id]);
  const hits = $derived(rt?.hits ?? []);
  const canDownload = $derived(hits.length > 0 && (cfg?.exportColumns.length ?? 0) > 0);
</script>

{#if rt && cfg}
  <NodeShell {id} title="Export" status={rt.status} {selected}>
    <Handle type="target" position={Position.Left} />
    <div class="flex flex-col gap-2">
      <div class="text-[11px] text-muted-foreground">
        {#if hits.length}
          <span class="text-foreground">{hits.length}</span> hits ·
          {cfg.exportColumns.length} col{cfg.exportColumns.length === 1 ? '' : 's'} ·
          {cfg.exportFormat.toUpperCase()}
          {#if cfg.exportColumns.includes('tags')}<span class="text-primary">+ tags</span>{/if}
        {:else if rt.status === 'idle'}
          Run the graph to feed this export.
        {:else}
          No results — check upstream nodes.
        {/if}
      </div>
      <button
        type="button"
        class="nodrag inline-flex items-center justify-center gap-1.5 rounded border border-border bg-background px-2 py-1 text-[11px] font-medium text-foreground transition-colors hover:bg-muted disabled:opacity-50"
        disabled={!canDownload}
        onclick={(e) => {
          e.stopPropagation();
          exportHits(graph.tags.withTags(hits), cfg.exportFormat, new Date(), cfg.exportColumns);
        }}
      >
        <Download class="size-3" />
        Download {cfg.exportFormat.toUpperCase()}
      </button>
    </div>
  </NodeShell>
{/if}
