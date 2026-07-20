<script lang="ts">
  // The annotator shell — three-zone layout (tool rail · resizable canvas+overlays ·
  // review inspector) over ONE media unit. Owns the AnnotatorController + the keyboard
  // controller; every child is dumb + controlled. The /annotate route re-mounts this
  // per unit (via {#key}) so navigating a review selection loads each unit fresh.
  import { viewerFor } from '$lib/viewer/registry';
  import type { MediaUnit } from '$lib/viewer/types';
  import { AnnotatorController } from '$lib/viewer/annotator.svelte';
  import { reviewSelection } from '$lib/labeling/review-selection.svelte';
  import ResizableSplit from '$lib/components/resizable-split.svelte';
  import AnnotatorToolbar from './AnnotatorToolbar.svelte';
  import AnnotationSidebar from './AnnotationSidebar.svelte';
  import ZoomControls from './ZoomControls.svelte';
  import PageNav from './PageNav.svelte';
  import AiAssistBar from './AiAssistBar.svelte';
  import type { Tool } from '$lib/engine';

  let { unit }: { unit: MediaUnit } = $props();

  const Viewer = $derived(viewerFor(unit.kind));
  const controller = new AnnotatorController();
  let status = $state('loading…');

  // Page nav = the review selection (else this single unit). Navigating drives the
  // shared store, whose index change re-mounts this shell with the next unit.
  const pages = $derived(
    reviewSelection.total > 0
      ? reviewSelection.units.map((u, i) => ({ key: u.key, label: `#${i + 1}` }))
      : [{ key: unit.key, label: 'p' }],
  );
  const pageIndex = $derived(reviewSelection.total > 0 ? reviewSelection.index : 0);
  function navigate(i: number): void {
    if (reviewSelection.total > 0) reviewSelection.go(i);
  }

  const TOOL_KEYS: Record<string, Tool> = {
    '1': 'select',
    '2': 'pan',
    '3': 'rect',
    '4': 'polygon',
    '5': 'point',
    '6': 'line',
    '7': 'lasso',
    b: 'brush',
  };
  function onKeydown(e: KeyboardEvent): void {
    const el = e.target as HTMLElement | null;
    if (el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.isContentEditable)) return;
    const k = e.key.toLowerCase();
    if (e.ctrlKey || e.metaKey) {
      if (k === 'z') {
        e.preventDefault();
        if (e.shiftKey) controller.redo();
        else controller.undo();
      } else if (k === 'y') {
        e.preventDefault();
        controller.redo();
      } else if (k === 's') {
        e.preventDefault();
        void controller.save();
      }
      return;
    }
    const tool = TOOL_KEYS[k];
    if (tool) {
      if (controller.canDraw || tool === 'select' || tool === 'pan') controller.setTool(tool);
      return;
    }
    if (k === 'a' || e.key === 'Enter') {
      controller.acceptAndAdvance('accepted');
    } else if (k === 'r') {
      controller.acceptAndAdvance('rejected');
    } else if (k === 'j' || e.key === 'ArrowDown') {
      e.preventDefault();
      controller.next();
    } else if (k === 'k' || e.key === 'ArrowUp') {
      e.preventDefault();
      controller.prev();
    } else if (e.key === 'Delete' || e.key === 'Backspace') {
      controller.deleteSelected();
    } else if (k === 'p') {
      controller.convertToPolygon();
    } else if (e.key === 'Escape') {
      controller.select(null);
    }
  }
</script>

<svelte:window onkeydown={onKeydown} />

<div class="flex h-screen w-screen">
  <AnnotatorToolbar {controller} />

  <div class="min-w-0 flex-1">
    <ResizableSplit storageKey="lance-media-annotate" initial={0.72} minLeft={420} minRight={320}>
      {#snippet left()}
        <div class="relative h-full w-full">
          <div
            class="absolute left-3 top-3 z-10 rounded bg-black/70 px-2 py-1 font-mono text-xs text-white"
            data-testid="annotate-status"
          >
            annotate · {unit.kind} · {status}
          </div>
          <Viewer {unit} {controller} onload={(n) => (status = `${n} annotations from Lance`)} />
          {#if controller.canDraw}
            <AiAssistBar {controller} />
          {/if}
          <PageNav {pages} current={pageIndex} onNavigate={navigate} />
          <ZoomControls {controller} />
        </div>
      {/snippet}
      {#snippet right()}
        <AnnotationSidebar {controller} />
      {/snippet}
    </ResizableSplit>
  </div>
</div>
