<script lang="ts">
  // Video viewer — SCAFFOLD. Fill in BOTH lanes:
  //  • SPATIAL: a `<video>` + the ra-anno PixiJS overlay synced to currentTime —
  //    mask + pixels + bbox/polygon drawn ON the current frame; each shape carries a
  //    temporal facet (frame_idx / t). Reuse ImagePlugin+ArrowDataPlugin on the frame.
  //  • TEMPORAL: a peaks.js timeline over the video's AUDIO track — segments/points.
  // Both feed the SAME AnnotationStore + annotator service (one annotations table;
  // the geometry present says which lane a row belongs to). Contract-satisfying stub
  // so the registry + route stay decoupled — later wiring is drop-in.
  import type { ViewerProps } from './types';

  let { unit, onload }: ViewerProps = $props();
  $effect(() => onload?.(0));
</script>

<div
  class="flex h-full w-full items-center justify-center p-6 text-center text-sm text-muted-foreground"
>
  <div>
    <div class="font-medium">video annotation — scaffold</div>
    <div class="mt-1 text-xs">
      frame overlay: mask · pixels · bbox/polygon (@currentTime) + audio-track timeline · {unit.key}
    </div>
  </div>
</div>
