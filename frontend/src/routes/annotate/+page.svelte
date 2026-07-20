<script lang="ts">
  // Increment-2 wire: the ra-anno PixiJS engine, mounted and fed REAL Lance
  // annotations over Arrow IPC. A frame image (chunk_frames) is the backdrop; the
  // annotations table (backend /api/annotations → Arrow IPC) draws on top.
  import PixiCanvas from '$lib/viewer/PixiCanvas.svelte';
  import { tableFromIPC } from 'apache-arrow';
  import type { PixiContext } from '$lib/engine';

  // A real doc/frame that has seeded Lance annotations (doc/speech/chunk keys).
  const KEY = 'fe00cd746463ad2c/0/19';

  let status = $state('loading…');
  let count = $state(0);

  async function onready(ctx: PixiContext): Promise<void> {
    try {
      await ctx.plugins.image.load(`/api/chunk-frame/${KEY}`);
      const res = await fetch(`/api/annotations/${KEY}`);
      if (!res.ok) throw new Error(`annotations HTTP ${res.status}`);
      const table = tableFromIPC(new Uint8Array(await res.arrayBuffer()));
      ctx.plugins.arrow.load(table);
      ctx.plugins.arrow.sync();
      count = table.numRows;
      status = `${count} annotations from Lance`;
    } catch (e) {
      status = `failed: ${e instanceof Error ? e.message : String(e)}`;
    }
  }
</script>

<div class="relative h-screen w-screen">
  <div
    class="absolute left-3 top-3 z-10 rounded bg-black/70 px-2 py-1 font-mono text-xs text-white"
    data-testid="annotate-status"
  >
    annotate · {status}
  </div>
  <PixiCanvas {onready} />
</div>
