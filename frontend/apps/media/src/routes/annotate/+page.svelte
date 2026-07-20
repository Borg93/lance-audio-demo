<script lang="ts">
  // The annotator shell — ported from ra-anno's three-zone layout, decoupled from
  // any specific viewer. Left tool rail · resizable (canvas + overlays) · right
  // review/inspector. The route owns the AnnotatorController facade and is the only
  // place that wires the engine; every layout child is dumb + controlled. Adding a
  // modality never touches this route. (RA_ANNO_MERGE.md §5c–5d.)
  import { mediaKindOf, viewerFor } from '$lib/viewer/registry';
  import type { MediaUnit } from '$lib/viewer/types';
  import { AnnotatorController } from '$lib/viewer/annotator.svelte';
  import ResizableSplit from '$lib/components/resizable-split.svelte';
  import AnnotatorToolbar from '$lib/viewer/layout/AnnotatorToolbar.svelte';
  import AnnotationSidebar from '$lib/viewer/layout/AnnotationSidebar.svelte';
  import ZoomControls from '$lib/viewer/layout/ZoomControls.svelte';
  import PageNav from '$lib/viewer/layout/PageNav.svelte';
  import type { Tool } from '$lib/engine';

  // Demo unit: the seeded image doc. In the app this comes from the descriptor
  // (`document.mime` → kind) + the selected doc/frame — same shape, resolved upstream.
  const KEY = 'fe00cd746463ad2c/0/19';
  const unit: MediaUnit = {
    kind: mediaKindOf('image/jpeg'),
    key: KEY,
    imageUrl: `/api/chunk-frame/${KEY}`,
    annotationsUrl: `/api/annotations/${KEY}`,
  };
  const Viewer = viewerFor(unit.kind);

  // The single reactive source of truth the whole layout binds to.
  const controller = new AnnotatorController();

  let status = $state('loading…');

  // Page list — single seeded page today; a real page-list source (backend
  // descriptor) drops in here without touching the layout.
  const pages = [{ key: KEY, label: 'p19' }];
  let pageIndex = $state(0);

  // ── keyboard controller (route owns it; ra-anno had this at route level too) ──
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
  function onKeydown(e: KeyboardEvent) {
    const el = e.target as HTMLElement | null;
    if (el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.isContentEditable)) return;
    const k = e.key.toLowerCase();
    // Undo/redo first — Ctrl/Cmd combos never fall through to tool hotkeys.
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
      // drawing tools require edit mode
      if (controller.canDraw || tool === 'select' || tool === 'pan') controller.setTool(tool);
      return;
    }
    if (e.key === 'Delete' || e.key === 'Backspace') controller.deleteSelected();
    else if (k === 'p') controller.convertToPolygon();
    else if (e.key === 'Escape') controller.select(null);
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
          <PageNav {pages} current={pageIndex} onNavigate={(i) => (pageIndex = i)} />
          <ZoomControls {controller} />
        </div>
      {/snippet}
      {#snippet right()}
        <AnnotationSidebar {controller} />
      {/snippet}
    </ResizableSplit>
  </div>
</div>
