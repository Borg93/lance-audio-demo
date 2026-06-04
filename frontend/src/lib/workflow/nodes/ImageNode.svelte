<script lang="ts">
  /** Image input. Emits `{ image }` (a File) to the Search node, which POSTs it
   *  multipart so the backend embeds it for the visual leg. Set the Search mode
   *  to Image (or All) to actually rank by frame similarity. */
  import { Handle, Position, type NodeProps } from '@xyflow/svelte';
  import { graph } from '$lib/workflow/graph.svelte';
  import NodeShell from './NodeShell.svelte';

  let { id, selected }: NodeProps = $props();
  const cfg = $derived(graph.config[id]);
  const rt = $derived(graph.runtime[id]);

  let preview = $state<string | null>(null);

  function onFile(e: Event): void {
    const input = e.currentTarget as HTMLInputElement;
    const file = input.files?.[0] ?? null;
    if (preview) {
      URL.revokeObjectURL(preview);
      preview = null;
    }
    if (file) preview = URL.createObjectURL(file);
    graph.setConfig(id, { image: file, imageName: file?.name ?? '' });
  }
</script>

{#if cfg && rt}
  <NodeShell {id} title="Image" status={rt.status} {selected}>
    <input
      type="file"
      accept="image/*"
      class="nodrag w-full text-[10px] text-muted-foreground file:mr-2 file:rounded file:border-0 file:bg-secondary file:px-2 file:py-1 file:text-[10px] file:text-secondary-foreground"
      onchange={onFile}
    />
    {#if preview}
      <img
        src={preview}
        alt={cfg.imageName || 'query image'}
        class="mt-2 h-24 w-full rounded border border-border bg-muted object-cover"
      />
    {/if}
    <p class="mt-1 text-[10px] text-muted-foreground">
      Wire into Search, then set mode = <span class="text-foreground">Image</span> or
      <span class="text-foreground">All</span>.
    </p>
    <Handle type="source" position={Position.Right} />
  </NodeShell>
{/if}
