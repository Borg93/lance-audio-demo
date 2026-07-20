<script lang="ts">
  // Image/document viewer — the ra-anno PixiJS engine over a page image. WORKING.
  import { tableFromIPC } from 'apache-arrow';
  import type { PixiContext } from '$lib/engine';
  import PixiCanvas from './PixiCanvas.svelte';
  import type { ViewerProps } from './types';

  let { unit, onload, controller }: ViewerProps = $props();

  async function onready(ctx: PixiContext): Promise<void> {
    if (unit.imageUrl) await ctx.plugins.image.load(unit.imageUrl);
    const res = await fetch(unit.annotationsUrl);
    if (!res.ok) throw new Error(`annotations HTTP ${res.status}`);
    const version = Number(res.headers.get('X-Annotations-Version') ?? '0');
    const table = tableFromIPC(new Uint8Array(await res.arrayBuffer()));
    ctx.plugins.arrow.load(table);
    ctx.plugins.arrow.sync();
    // Lift the engine + data to the route-level facade so the annotator layout
    // (toolbar/sidebar/zoom/layers) can bind to it. No-op when unwired. The
    // annotations URL doubles as the Save (POST) target; the version drives the
    // optimistic-concurrency handshake.
    controller?.attach(ctx, table, unit.annotationsUrl, version);
    onload?.(table.numRows);
  }
</script>

<PixiCanvas {onready} />
