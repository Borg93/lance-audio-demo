<script lang="ts">
  /** Metadata filter. Emits `{ where, language, namn }` — merged into the Search
   *  request so the search runs over a narrowed slice of the corpus. */
  import { Handle, Position, type NodeProps } from '@xyflow/svelte';
  import { graph } from '$lib/workflow/graph.svelte';
  import { FIELD_CLASS } from './field';
  import NodeShell from './NodeShell.svelte';

  let { id, selected }: NodeProps = $props();
  const cfg = $derived(graph.config[id]);
  const rt = $derived(graph.runtime[id]);
</script>

{#if cfg && rt}
  <NodeShell {id} title="Filter" status={rt.status} {selected}>
    <div class="flex flex-col gap-2">
      <div>
        <label class="mb-1 block text-[10px] text-muted-foreground" for="lang-{id}">Language</label>
        <input id="lang-{id}" class="{FIELD_CLASS} w-full" placeholder="e.g. sv" bind:value={cfg.language} />
      </div>
      <div>
        <label class="mb-1 block text-[10px] text-muted-foreground" for="namn-{id}">Name (namn)</label>
        <input
          id="namn-{id}"
          class="{FIELD_CLASS} w-full"
          placeholder="archival name contains…"
          bind:value={cfg.namn}
        />
      </div>
      <div>
        <label class="mb-1 block text-[10px] text-muted-foreground" for="where-{id}">SQL where</label>
        <input
          id="where-{id}"
          class="{FIELD_CLASS} w-full font-mono"
          placeholder="duration > 30"
          bind:value={cfg.where}
        />
      </div>
    </div>
    <Handle type="source" position={Position.Right} />
  </NodeShell>
{/if}
