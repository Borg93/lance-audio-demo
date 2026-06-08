<script lang="ts">
  /** Canvas toolbar (Run / Clear / Reset / Delete / Add) — rendered INSIDE
   *  <SvelteFlow> so `useSvelteFlow()` is in scope and new nodes can spawn at
   *  the centre of the visible canvas instead of off-screen. */
  import { useSvelteFlow } from '@xyflow/svelte';
  import { Play, RotateCcw, Plus, LoaderCircle, Trash2 } from 'lucide-svelte';
  import { Button } from '$lib/components/ui';
  import { graph, nodeLabel, type NodeKind } from '$lib/workflow/graph.svelte';

  const ADDABLE: NodeKind[] = ['query', 'image', 'filter', 'search', 'combine', 'results'];
  const { screenToFlowPosition } = useSvelteFlow();

  // Stagger repeated adds so they don't stack exactly. Not rendered → plain let.
  let spawn = 0;

  function add(kind: NodeKind): void {
    spawn += 1;
    const offset = (spawn % 5) * 30;
    // Centre of the actual flow pane (handles the split layout), in client
    // coords, mapped to flow coords so the node lands where the user is looking.
    const pane = document.querySelector('.svelte-flow')?.getBoundingClientRect();
    const cx = (pane ? pane.left + pane.width / 2 : window.innerWidth / 2) + offset;
    const cy = (pane ? pane.top + pane.height / 2 : window.innerHeight / 2) + offset;
    graph.addNode(kind, screenToFlowPosition({ x: cx, y: cy }));
  }
</script>

<div
  class="flex flex-col gap-2 rounded-lg border border-border bg-card/95 p-2 shadow-md backdrop-blur"
>
  <div class="flex items-center gap-1.5">
    <Button size="sm" onclick={() => graph.run()} disabled={graph.running}>
      {#if graph.running}
        <LoaderCircle class="size-3.5 animate-spin" />
      {:else}
        <Play class="size-3.5" />
      {/if}
      Run
    </Button>
    <Button size="sm" variant="outline" onclick={() => graph.resetRun()} disabled={graph.running}>
      Clear run
    </Button>
    <Button size="sm" variant="ghost" onclick={() => graph.reset()} disabled={graph.running}>
      <RotateCcw class="size-3.5" />
      Reset
    </Button>
    <Button
      size="sm"
      variant="outline"
      onclick={() => graph.deleteSelected()}
      disabled={!graph.hasSelection || graph.running}
      title="Delete the selected node(s) / edge(s)"
    >
      <Trash2 class="size-3.5" />
      Delete
    </Button>
  </div>
  <div class="flex flex-wrap items-center gap-1">
    <span class="mr-0.5 text-[10px] font-medium tracking-wide text-muted-foreground uppercase">
      Add
    </span>
    {#each ADDABLE as kind (kind)}
      <Button size="xs" variant="secondary" onclick={() => add(kind)}>
        <Plus class="size-3" />
        {nodeLabel(kind)}
      </Button>
    {/each}
  </div>
  {#if graph.lastError}
    <div class="max-w-[18rem] text-[10px] text-destructive">{graph.lastError}</div>
  {/if}
  <div class="max-w-[19rem] space-y-0.5 text-[10px] text-muted-foreground/80">
    <div>
      <span class="text-foreground">Connect</span> — drag a node's right ● onto another's left ●. A Search
      accepts several inputs at once (a query/image + a refine).
    </div>
    <div>
      <span class="text-foreground">Delete</span> — hover a node and click ✕, or select a node/edge and
      press ⌫ (or the Delete button).
    </div>
    <div>
      <span class="text-foreground">Refine</span> — Search → Search scopes the second to the first's videos.
      Click any node to inspect it on the right.
    </div>
  </div>
</div>
