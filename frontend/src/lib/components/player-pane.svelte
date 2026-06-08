<script lang="ts">
  import { type Hit, type Alignment, getDocTranscript, mediaUrl } from '$lib/api';
  import { fmtTime } from '$lib/utils';
  import { Captions, Maximize2, Minimize2 } from 'lucide-svelte';
  import TranscriptHighlighter from './transcript-highlighter.svelte';

  type Props = {
    hit: Hit | null;
    query?: string;
  };
  let { hit, query = '' }: Props = $props();

  let mediaEl = $state<HTMLVideoElement | null>(null);
  let mediaError = $state<string | null>(null);

  // Search hits ship `alignments: []` (the per-word timing blob is ~80% of a
  // search payload and only the player needs it). On opening a hit we lazy-fetch
  // the WHOLE document transcript (keyed on doc_id) so karaoke continues past the
  // selected chunk; atlas/selection hits carry only the selected chunk so we
  // can't rely on `hit.alignments`. The seek $effect below sets currentTime to
  // hit.start, and TranscriptHighlighter's time loop scrolls to the sentence in
  // the opened chunk once these absolute-time alignments populate.
  let alignments = $state<Alignment[]>([]);
  $effect(() => {
    const h = hit;
    if (!h) {
      alignments = [];
      return;
    }
    alignments = [];
    let cancelled = false; // supersede guard + leak guard
    getDocTranscript(h.doc_id)
      .then((doc) => {
        if (!cancelled) alignments = doc.chunks.flatMap((c) => c.alignments);
      })
      .catch(() => {
        /* leave empty — the video still plays, just no karaoke */
      });
    return () => {
      cancelled = true;
    };
  });

  // The unified video+transcript card. We fullscreen THIS wrapper (not the bare
  // <video>) so the live transcript overlay survives — native video fullscreen
  // can't render HTML on top.
  let fsWrap = $state<HTMLDivElement | null>(null);
  let isFullscreen = $state(false);

  const toggleFullscreen = () => {
    if (document.fullscreenElement) {
      void document.exitFullscreen();
    } else {
      void fsWrap?.requestFullscreen();
    }
  };

  // Mirror the browser's fullscreen state into a rune so the layout can react.
  $effect(() => {
    const onChange = () => {
      isFullscreen = document.fullscreenElement === fsWrap;
    };
    document.addEventListener('fullscreenchange', onChange);
    return () => {
      document.removeEventListener('fullscreenchange', onChange);
    };
  });

  // Archival metadata shown under the player — only fields that are present.
  const metaRows = $derived.by((): [string, string][] => {
    const h = hit;
    if (!h) return [];
    const rows: [string, string][] = [];
    const add = (label: string, v: string | number | null | undefined) => {
      if (v !== null && v !== undefined && v !== '') rows.push([label, String(v)]);
    };
    add('Caption', h.caption);
    add('File', h.audio_path);
    add('Time', `${fmtTime(h.start)} → ${fmtTime(h.end)}`);
    if (h.duration != null) add('Length', fmtTime(h.duration));
    add('Name', h.namn);
    add('Reference', h.referenskod);
    add('Image ID', h.bildid);
    add('Extra ID', h.extraid);
    add('Language', h.language);
    add('Segment', `speech ${h.speech_id} · chunk ${h.chunk_id}`);
    return rows;
  });

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
    <!-- ONE media component: the video and its synced transcript live in a
         single card, bridged by a slim "Transcript" bar that also carries the
         fullscreen toggle (off the video, so nothing covers it). In fullscreen
         we fullscreen THIS wrapper so the transcript can overlay the video as a
         centred caption band. -->
    {#snippet fsToggle(cls: string)}
      <button
        type="button"
        onclick={toggleFullscreen}
        title={isFullscreen ? 'Exit fullscreen' : 'Fullscreen with transcript'}
        aria-label={isFullscreen ? 'Exit fullscreen' : 'Fullscreen with transcript'}
        class={cls}
      >
        {#if isFullscreen}
          <Minimize2 class="size-4" />
        {:else}
          <Maximize2 class="size-4" />
        {/if}
      </button>
    {/snippet}

    <div
      bind:this={fsWrap}
      class={isFullscreen
        ? 'relative flex min-h-0 flex-col bg-black'
        : 'relative flex min-h-0 flex-1 flex-col overflow-hidden rounded-xl border border-border bg-card shadow-sm'}
    >
      {#if isFullscreen}
        {@render fsToggle(
          'absolute right-2 top-2 z-10 rounded-md bg-black/50 p-1.5 text-white/90 backdrop-blur transition-colors hover:bg-black/70 hover:text-white',
        )}
      {/if}

      <!-- 16:9 box sized from the pane WIDTH (not height) so the video stays
           visible even when the pane is dragged short. In fullscreen it fills
           the screen height instead. -->
      <video
        bind:this={mediaEl}
        controls
        controlslist="nofullscreen"
        preload="auto"
        src={mediaUrl(hit.doc_id)}
        class={isFullscreen
          ? 'min-h-0 w-full flex-1 bg-black object-contain'
          : 'aspect-video max-h-[55vh] w-full shrink-0 bg-black object-contain'}
      >
        <track kind="captions" />
      </video>

      {#if !isFullscreen}
        <!-- Bridge bar: ties the video to its transcript and holds the
             fullscreen toggle (kept off the video so nothing covers it). -->
        <div
          class="flex items-center gap-1.5 border-t border-border/70 px-3 py-1.5 text-xs font-medium text-muted-foreground"
        >
          <Captions class="size-3.5" />
          <span>Transcript</span>
          {@render fsToggle(
            'ml-auto rounded p-1 transition-colors hover:bg-secondary hover:text-foreground',
          )}
        </div>
      {/if}

      <!-- Transcript body. Normal: scrolls below the bar inside the card.
           Fullscreen: a centred caption band lifted ABOVE the native control bar
           (bottom-20) so it never covers the play button. -->
      <div
        class={isFullscreen
          ? 'absolute inset-x-0 bottom-20 mx-auto max-h-[32%] w-[min(92%,60rem)] overflow-y-auto rounded-xl bg-black/55 px-6 py-4 text-center text-lg leading-8 text-white backdrop-blur-sm'
          : 'min-h-0 flex-1 overflow-y-auto text-sm leading-7'}
      >
        <TranscriptHighlighter {alignments} media={mediaEl} {query} chrome={false} />
      </div>
    </div>

    {#if mediaError}
      <div class="shrink-0 text-sm text-destructive">
        Video failed to load: {mediaError}. Check
        <code>/api/media/{hit.doc_id}</code>.
      </div>
    {/if}

    {#if metaRows.length}
      <dl
        class="grid shrink-0 grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 rounded-md border border-border bg-card/40 px-3 py-2 text-xs"
      >
        {#each metaRows as [label, value] (label)}
          <dt class="text-muted-foreground">{label}</dt>
          <dd class="truncate font-medium text-foreground" title={value}>{value}</dd>
        {/each}
      </dl>
    {/if}
  {/if}
</div>
