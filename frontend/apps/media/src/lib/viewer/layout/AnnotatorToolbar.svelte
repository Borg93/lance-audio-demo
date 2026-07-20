<script lang="ts">
  // Left tool rail — the primary command surface. Fully controlled: reads/writes
  // the AnnotatorController facade, never the engine directly. (Ported from
  // ra-anno Toolbar.svelte, trimmed to functional controls for our engine.)
  import {
    MousePointer2,
    Hand,
    Square,
    Pentagon,
    Crosshair,
    Minus,
    Paintbrush,
    Lasso,
    Eye,
    Pencil,
    Trash2,
    Spline,
    Eraser,
    Undo2,
    Redo2,
  } from 'lucide-svelte';
  import { Button } from '$lib/components/ui';
  import { cn } from '$lib/utils';
  import type { Tool } from '$lib/engine';
  import type { AnnotatorController } from '../annotator.svelte';

  let { controller }: { controller: AnnotatorController } = $props();

  type ToolDef = { tool: Tool; icon: typeof MousePointer2; label: string; key: string; drawing: boolean };
  const TOOLS: ToolDef[] = [
    { tool: 'select', icon: MousePointer2, label: 'Select', key: '1', drawing: false },
    { tool: 'pan', icon: Hand, label: 'Pan', key: '2', drawing: false },
    { tool: 'rect', icon: Square, label: 'Rectangle', key: '3', drawing: true },
    { tool: 'polygon', icon: Pentagon, label: 'Polygon', key: '4', drawing: true },
    { tool: 'point', icon: Crosshair, label: 'Point', key: '5', drawing: true },
    { tool: 'line', icon: Minus, label: 'Line', key: '6', drawing: true },
    { tool: 'lasso', icon: Lasso, label: 'Lasso', key: '7', drawing: true },
    { tool: 'brush', icon: Paintbrush, label: 'Brush', key: 'B', drawing: true },
  ];

  const visible = $derived(TOOLS.filter((t) => !t.drawing || controller.canDraw));
</script>

<div
  class="flex h-full w-11 shrink-0 flex-col items-center gap-1 border-r border-border bg-card py-2"
  data-testid="annotator-toolbar"
>
  <!-- Mode toggle -->
  <Button
    variant={controller.mode === 'edit' ? 'default' : 'ghost'}
    size="icon-sm"
    title={controller.mode === 'edit' ? 'Edit mode (click to view)' : 'View mode (click to edit)'}
    aria-pressed={controller.mode === 'edit'}
    onclick={() => controller.toggleMode()}
  >
    {#if controller.mode === 'edit'}<Pencil class="size-4" />{:else}<Eye class="size-4" />{/if}
  </Button>

  <div class="my-1 h-px w-6 bg-border"></div>

  {#each visible as t (t.tool)}
    {@const Icon = t.icon}
    <Button
      variant={controller.activeTool === t.tool ? 'default' : 'ghost'}
      size="icon-sm"
      title={`${t.label} (${t.key})`}
      aria-pressed={controller.activeTool === t.tool}
      onclick={() => controller.setTool(t.tool)}
    >
      <Icon class="size-4" />
    </Button>
  {/each}

  {#if controller.activeTool === 'brush'}
    <Button
      variant={controller.brushOptions.erasing ? 'default' : 'ghost'}
      size="icon-sm"
      title="Erase (brush)"
      aria-pressed={controller.brushOptions.erasing}
      onclick={() => controller.setBrushOptions({ erasing: !controller.brushOptions.erasing })}
    >
      <Eraser class="size-4" />
    </Button>
  {/if}

  <div class="my-1 h-px w-6 bg-border"></div>

  <!-- Undo / redo (field edits: relabel / accept / reject / text) -->
  <Button
    variant="ghost"
    size="icon-sm"
    title="Undo (Ctrl+Z)"
    disabled={!controller.canUndo}
    onclick={() => controller.undo()}
  >
    <Undo2 class="size-4" />
  </Button>
  <Button
    variant="ghost"
    size="icon-sm"
    title="Redo (Ctrl+Shift+Z)"
    disabled={!controller.canRedo}
    onclick={() => controller.redo()}
  >
    <Redo2 class="size-4" />
  </Button>

  <div class="my-1 h-px w-6 bg-border"></div>

  <!-- Selection actions -->
  <Button
    variant="ghost"
    size="icon-sm"
    title="Convert to polygon (P)"
    disabled={controller.selectedIndex == null}
    onclick={() => controller.convertToPolygon()}
  >
    <Spline class="size-4" />
  </Button>
  <Button
    variant="ghost"
    size="icon-sm"
    title="Delete selected (Del)"
    disabled={controller.selectedIndex == null}
    onclick={() => controller.deleteSelected()}
  >
    <Trash2 class="size-4" />
  </Button>

  <div class="mt-auto flex flex-col items-center gap-1">
    <span
      class={cn(
        'size-2 rounded-full',
        controller.dirty ? 'bg-amber-500' : 'bg-transparent',
      )}
      title={controller.dirty ? 'Unsaved edits' : 'No pending edits'}
    ></span>
    <span class="text-[10px] tabular-nums text-muted-foreground" title="Annotation count">
      {controller.count}
    </span>
  </div>
</div>
