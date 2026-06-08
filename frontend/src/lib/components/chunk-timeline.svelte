<script lang="ts">
  import type { DocTranscriptChunk } from '$lib/api';

  type Props = {
    chunks: DocTranscriptChunk[];
    duration: number;
    currentTime: number;
    activeKey: string | null;
    onSeek: (t: number) => void;
  };
  let { chunks, duration, currentTime, activeKey, onSeek }: Props = $props();

  // Stable identity for a chunk = speech_id+chunk_id (matches the activeKey the
  // parent derives from the opened hit). Built once per chunks change.
  const key = (c: DocTranscriptChunk): string => `${c.speech_id}:${c.chunk_id}`;

  // Playhead fraction in [0,1]; clamped so a stale currentTime can't overflow.
  const playFrac = $derived(
    duration > 0 ? Math.min(1, Math.max(0, currentTime / duration)) : 0,
  );
</script>

{#if duration > 0 && chunks.length > 0}
  <div
    class="relative h-2 w-full shrink-0 overflow-hidden bg-secondary/40"
    role="group"
    aria-label="Chunk timeline"
  >
    {#each chunks as c (key(c))}
      <!-- Clamp to [0,100] so drifted data (end>duration, end<start) can't
           overflow the track or render a negative-width segment. -->
      {@const left = Math.min(100, Math.max(0, (c.start / duration) * 100))}
      {@const width = Math.min(
        100 - left,
        Math.max(0, ((c.end - c.start) / duration) * 100),
      )}
      <button
        type="button"
        title={c.text}
        aria-label={c.text}
        onclick={() => onSeek(c.start)}
        class="absolute top-0 h-full cursor-pointer border-r border-background/70 transition-colors hover:bg-primary/30 {key(
          c,
        ) === activeKey
          ? 'bg-primary/40'
          : 'bg-muted'}"
        style="left: {left}%; width: {width}%;"
      ></button>
    {/each}
    <!-- Playhead: a thin vertical line at currentTime/duration. pointer-events
         none so it never eats segment clicks. -->
    <div
      class="pointer-events-none absolute top-0 h-full w-px bg-primary"
      style="left: {playFrac * 100}%;"
    ></div>
  </div>
{/if}
