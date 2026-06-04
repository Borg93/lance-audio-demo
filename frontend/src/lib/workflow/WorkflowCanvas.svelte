<script lang="ts">
  /** The node-pipeline editor canvas: a Svelte Flow bound to the `graph`
   *  singleton, with the toolbar (in a Panel) and selection/inspection wiring. */
  import { SvelteFlow, Background, Controls, MiniMap, Panel, type ColorMode } from '@xyflow/svelte';
  import '@xyflow/svelte/dist/style.css';
  import { browser } from '$app/environment';
  import { graph } from '$lib/workflow/graph.svelte';
  import { nodeTypes } from '$lib/workflow/node-types';
  import WorkflowToolbar from '$lib/workflow/WorkflowToolbar.svelte';

  // Match the app's theme (boots dark; .dark class on <html>). Read once at
  // mount — good enough for a POC, the canvas just needs the right base palette.
  const colorMode: ColorMode =
    browser && document.documentElement.classList.contains('dark') ? 'dark' : 'light';

  // Autosave: snapshot() reads nodes/edges/config deeply, so this effect re-runs
  // on any change; debounce the localStorage write so typing doesn't thrash it.
  $effect(() => {
    const json = graph.snapshot();
    const timer = setTimeout(() => graph.persist(json), 400);
    return () => clearTimeout(timer);
  });
</script>

<div class="h-full w-full">
  <SvelteFlow
    bind:nodes={graph.nodes}
    bind:edges={graph.edges}
    {nodeTypes}
    {colorMode}
    fitView
    onnodeclick={(e) => graph.inspectNode(e.node.id)}
    ondelete={({ nodes, edges }) =>
      graph.syncDeleted(
        nodes.map((n) => n.id),
        edges.map((e) => e.id),
      )}
    onselectionchange={(p) =>
      graph.setSelection(
        p.nodes.map((n) => n.id),
        p.edges.map((e) => e.id),
      )}
  >
    <Background />
    <Controls />
    <MiniMap />
    <Panel position="top-left">
      <WorkflowToolbar />
    </Panel>
  </SvelteFlow>
</div>
