<script lang="ts">
  // Audio viewer — SCAFFOLD. Fill in a peaks.js WaveformCanvas: the audio waveform
  // with temporal SEGMENTS/points (t_start/t_end), sharing the same AnnotationStore
  // + annotator service. Precomputed waveform peaks come from a `waveform` silver
  // stage (compact Arrow), not client-side decode. Contract-satisfying today so the
  // registry + route stay decoupled — wiring it later is drop-in, not a rewire.
  import type { ViewerProps } from './types';

  let { unit, onload }: ViewerProps = $props();
  $effect(() => onload?.(0));
</script>

<div
  class="flex h-full w-full items-center justify-center p-6 text-center text-sm text-muted-foreground"
>
  <div>
    <div class="font-medium">audio annotation — scaffold</div>
    <div class="mt-1 text-xs">peaks.js waveform · temporal segments (t_start/t_end) · {unit.key}</div>
  </div>
</div>
