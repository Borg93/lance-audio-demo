<script lang="ts">
  /** The execution node. Searches for its inline query (or a connected Query
   *  node / Image), in its chosen mode. If another Search feeds into it, this
   *  one is scoped to the videos that upstream result came from — so chaining
   *  Search → Search refines progressively. */
  import { Handle, Position, type NodeProps } from '@xyflow/svelte';
  import type { SearchMode } from '$lib/api';
  import { graph, SEARCH_MODES, modeLabel } from '$lib/workflow/graph.svelte';
  import { FIELD_CLASS } from './field';
  import NodeShell from './NodeShell.svelte';

  let { id, selected }: NodeProps = $props();
  const cfg = $derived(graph.config[id]);
  const rt = $derived(graph.runtime[id]);
  const isVisual = $derived(cfg?.mode === 'visual');
</script>

{#if cfg && rt}
  <NodeShell {id} title="Search · {modeLabel(cfg.mode)}" status={rt.status} {selected}>
    <Handle type="target" position={Position.Left} />

    <label class="mb-1 block text-[10px] text-muted-foreground" for="q-{id}">
      {isVisual ? 'Query (optional — image drives it)' : 'Query'}
    </label>
    <input
      id="q-{id}"
      class="{FIELD_CLASS} w-full"
      placeholder={isVisual ? 'uses the connected image' : 'e.g. skatt'}
      bind:value={cfg.q}
    />

    <label class="mt-2 mb-1 block text-[10px] text-muted-foreground" for="mode-{id}">Mode</label>
    <select
      id="mode-{id}"
      class="{FIELD_CLASS} w-full"
      value={cfg.mode}
      onchange={(e) => graph.setConfig(id, { mode: e.currentTarget.value as SearchMode })}
    >
      {#each SEARCH_MODES as m (m.value)}
        <option value={m.value}>{m.label}</option>
      {/each}
    </select>

    <div class="mt-2 flex items-center gap-2">
      <label class="text-[10px] text-muted-foreground" for="n-{id}">Results</label>
      <input
        id="n-{id}"
        type="number"
        min="1"
        max="100"
        class="{FIELD_CLASS} w-16"
        value={cfg.n}
        oninput={(e) =>
          graph.setConfig(id, { n: Math.max(1, Math.min(100, Number(e.currentTarget.value) || 24)) })}
      />
      <label class="nodrag ml-auto flex items-center gap-1.5 text-[10px] text-muted-foreground">
        <input type="checkbox" bind:checked={cfg.rerank} />
        Rerank
      </label>
    </div>

    <div class="mt-2 border-t border-border pt-1.5 text-[10px]">
      {#if rt.status === 'running'}
        <span class="text-primary">Searching…</span>
      {:else if rt.status === 'error'}
        <span class="text-destructive">{rt.error}</span>
      {:else if rt.status === 'done'}
        <span class="text-muted-foreground">
          <span class="text-foreground">{rt.count}</span> hits · {rt.ms} ms{#if rt.scopedDocs}
            · within <span class="text-foreground">{rt.scopedDocs}</span> videos{#if rt.scopeCapped}
              <span class="text-amber-500"> (capped)</span>{/if}{/if}
        </span>
        {#if rt.count === 0 && rt.scopedDocs}
          <div class="mt-1 text-amber-500">
            Nothing matched inside those {rt.scopedDocs} videos — delete the incoming refine edge to search
            all.
          </div>
        {/if}
      {:else}
        <span class="text-muted-foreground/70">idle — add a query or image, then Run</span>
      {/if}
      {#if rt.droppedInputs > 0}
        <div class="mt-1 text-amber-500">
          {rt.droppedInputs} extra input{rt.droppedInputs > 1 ? 's' : ''} ignored — a Search uses one
          query + one image. Use a Combine node to merge result sets.
        </div>
      {/if}
    </div>

    <Handle type="source" position={Position.Right} />
  </NodeShell>
{/if}
