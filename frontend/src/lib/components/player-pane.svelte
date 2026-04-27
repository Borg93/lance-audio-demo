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
   * Whenever `hit` changes, dispose the old media and seek the new one to
   * `hit.start`. The cleanup function ensures we never leak a playing
   * video when the user clicks a different hit (the bug that haunted the
   * old vanilla frontend).
   */
  $effect(() => {
    mediaError = null;
    if (!hit || !mediaEl) return;

    const el = mediaEl;
    let sought = false;
    const seekOnce = () => {
      if (sought) return;
      sought = true;
      try {
        el.currentTime = hit.start;
      } catch (e) {
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
    el.addEventListener('loadedmetadata', seekOnce);
    el.addEventListener('canplay', seekOnce);
    el.addEventListener('error', onError);

    return () => {
      el.removeEventListener('loadedmetadata', seekOnce);
      el.removeEventListener('canplay', seekOnce);
      el.removeEventListener('error', onError);
      try {
        el.pause();
        el.removeAttribute('src');
        el.load();
      } catch (e) {
        // ignore — element may already be detached
      }
    };
  });
</script>

<div class="flex h-full flex-col gap-4 overflow-y-auto p-6">
  {#if !hit}
    <div class="m-auto text-sm text-muted-foreground">Click a hit to play.</div>
  {:else}
    <div class="text-sm font-medium">
      {hit.audio_path} · {fmtTime(hit.start)} → {fmtTime(hit.end)}
    </div>

    <video
      bind:this={mediaEl}
      controls
      preload="auto"
      src={mediaUrl(hit.doc_id)}
      class="max-h-[320px] w-full rounded-lg bg-black"
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
