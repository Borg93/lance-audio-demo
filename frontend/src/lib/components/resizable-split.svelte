<script lang="ts">
  import type { Snippet } from 'svelte';

  /**
   * Two-pane horizontal split with a draggable divider in the middle.
   * The split fraction (0..1) is persisted to localStorage so the user's
   * choice survives reloads.
   */

  type Props = {
    left: Snippet;
    right: Snippet;
    /** Storage key — change to keep separate splits across pages. */
    storageKey?: string;
    /** Initial fraction of the LEFT pane width, 0..1. */
    initial?: number;
    /** Hard limits. */
    minLeft?: number;
    minRight?: number;
  };
  let {
    left,
    right,
    storageKey = 'raudio-split',
    initial = 0.6,
    minLeft = 360,
    minRight = 320,
  }: Props = $props();

  import { untrack } from 'svelte';

  let container = $state<HTMLDivElement | null>(null);
  // Read `initial` once at component creation; `fraction` is the live state.
  let fraction = $state<number>(untrack(() => initial));

  // Hydrate persisted fraction once mounted (don't run on the server).
  $effect(() => {
    try {
      const v = localStorage.getItem(storageKey);
      if (v !== null) {
        const f = parseFloat(v);
        if (!Number.isNaN(f) && f > 0 && f < 1) fraction = f;
      }
    } catch {}
  });

  let dragging = $state(false);

  function onPointerDown(e: PointerEvent) {
    if (!container) return;
    dragging = true;
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    e.preventDefault();
  }

  function onPointerMove(e: PointerEvent) {
    if (!dragging || !container) return;
    const rect = container.getBoundingClientRect();
    const total = rect.width;
    const x = e.clientX - rect.left;
    const minF = minLeft / total;
    const maxF = 1 - minRight / total;
    fraction = Math.max(minF, Math.min(maxF, x / total));
  }

  function onPointerUp(e: PointerEvent) {
    if (!dragging) return;
    dragging = false;
    (e.currentTarget as HTMLElement).releasePointerCapture(e.pointerId);
    try {
      localStorage.setItem(storageKey, fraction.toFixed(3));
    } catch {}
  }

  // Double-click resets to default
  function onDoubleClick() {
    fraction = initial;
    try {
      localStorage.setItem(storageKey, initial.toFixed(3));
    } catch {}
  }
</script>

<div
  bind:this={container}
  class="grid h-full min-h-0"
  style:grid-template-columns="{(fraction * 100).toFixed(2)}% 6px 1fr"
  class:cursor-col-resize={dragging}
  class:select-none={dragging}
>
  <div class="min-h-0 overflow-hidden">{@render left()}</div>

  <button
    type="button"
    aria-label="Resize panels"
    onpointerdown={onPointerDown}
    onpointermove={onPointerMove}
    onpointerup={onPointerUp}
    ondblclick={onDoubleClick}
    onkeydown={(e) => {
      if (e.key === 'ArrowLeft') fraction = Math.max(0.2, fraction - 0.02);
      if (e.key === 'ArrowRight') fraction = Math.min(0.8, fraction + 0.02);
    }}
    title="Drag to resize · double-click to reset"
    class="group relative flex cursor-col-resize items-center justify-center border-x border-border bg-secondary/40
           transition-colors hover:bg-primary/30 active:bg-primary/40
           focus-visible:bg-primary/40 focus-visible:outline-none"
  >
    <!-- Persistent grip dots so it's obvious the bar is draggable -->
    <span aria-hidden="true" class="flex flex-col gap-0.5 text-muted-foreground/70 group-hover:text-foreground">
      <span class="size-0.5 rounded-full bg-current"></span>
      <span class="size-0.5 rounded-full bg-current"></span>
      <span class="size-0.5 rounded-full bg-current"></span>
      <span class="size-0.5 rounded-full bg-current"></span>
      <span class="size-0.5 rounded-full bg-current"></span>
    </span>
    <!-- Hover tooltip that confirms the affordance -->
    <span class="pointer-events-none absolute top-1/2 left-1/2 -translate-x-1/2 translate-y-6 whitespace-nowrap
                 rounded border border-border bg-card px-2 py-0.5 text-[10px] text-muted-foreground opacity-0
                 transition-opacity group-hover:opacity-100">
      drag to resize
    </span>
  </button>

  <div class="min-h-0 overflow-hidden">{@render right()}</div>
</div>
