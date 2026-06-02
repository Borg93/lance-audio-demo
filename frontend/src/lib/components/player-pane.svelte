<script lang="ts">
  import { type Hit, mediaUrl } from '$lib/api';
  import { fmtTime } from '$lib/utils';
  import TranscriptHighlighter from './transcript-highlighter.svelte';

  type Props = {
    hit: Hit | null;
    query?: string;
  };
  let { hit, query = '' }: Props = $props();

  let mediaEl = $state<HTMLVideoElement | null>(null);
  let mediaError = $state<string | null>(null);

  /**
   * Whenever `hit` changes, seek the player to `hit.start` and play.
   *
   * `src` is owned reactively by the `<video src={mediaUrl(hit.doc_id)}>`
   * binding below — this effect must NOT touch it. The old code called
   * `el.removeAttribute('src'); el.load()` in cleanup; when the next hit was
   * in the *same* document the bound URL didn't change, so Svelte never
   * re-applied `src` and the element was left sourceless and wedged (the
   * "second click freezes the player until full refresh" bug).
   *
   * We read `hit.doc_id` + `hit.start` so the effect re-runs on either a new
   * document (src changes → metadata reloads → seek on `loadedmetadata`) or a
   * new chunk in the same already-loaded document (seek immediately, since
   * `loadedmetadata`/`canplay` won't fire again).
   */
  $effect(() => {
    mediaError = null;
    if (!hit || !mediaEl) return;

    const el = mediaEl;
    const start = hit.start;
    let cancelled = false;
    const seek = () => {
      if (cancelled) return;
      try {
        el.currentTime = start;
      } catch {
        // ignore — some browsers throw if metadata isn't fully ready
      }
      el.play().catch(() => {});
    };
    const onError = () => {
      const err = el.error;
      const codes: Record<number, string> = {
        1: 'MEDIA_ERR_ABORTED',
        2: 'MEDIA_ERR_NETWORK',
        3: 'MEDIA_ERR_DECODE',
        4: 'MEDIA_ERR_SRC_NOT_SUPPORTED',
      };
      mediaError = err ? (codes[err.code] ?? 'unknown') : 'unknown';
    };

    if (el.readyState >= 1 /* HAVE_METADATA */) {
      seek();
    } else {
      el.addEventListener('loadedmetadata', seek, { once: true });
      el.addEventListener('canplay', seek, { once: true });
    }
    el.addEventListener('error', onError);

    return () => {
      cancelled = true;
      el.removeEventListener('loadedmetadata', seek);
      el.removeEventListener('canplay', seek);
      el.removeEventListener('error', onError);
      el.pause();
    };
  });
</script>

<div class="flex h-full min-h-0 flex-col gap-3 p-4">
  {#if !hit}
    <div class="m-auto text-sm text-muted-foreground">Click a hit to play.</div>
  {:else}
    <div class="shrink-0 text-sm font-medium">
      {hit.audio_path} · {fmtTime(hit.start)} → {fmtTime(hit.end)}
    </div>

    {#if hit.caption}
      <div class="shrink-0 text-xs italic text-muted-foreground" title="AI scene caption">
        🎬 {hit.caption}
      </div>
    {/if}

    <!-- Grow to fill the pane height; object-contain keeps the aspect ratio
         (letterboxed) instead of the old fixed 320px cap. -->
    <video
      bind:this={mediaEl}
      controls
      preload="auto"
      src={mediaUrl(hit.doc_id)}
      class="min-h-0 w-full flex-1 rounded-lg bg-black object-contain"
    >
      <track kind="captions" />
    </video>

    {#if mediaError}
      <div class="text-sm text-destructive">
        Video failed to load: {mediaError}. Check
        <code>/api/media/{hit.doc_id}</code>.
      </div>
    {/if}

    <TranscriptHighlighter alignments={hit.alignments} media={mediaEl} {query} />
  {/if}
</div>
