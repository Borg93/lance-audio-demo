<script lang="ts">
  import { type Hit, type DocTranscriptChunk, getDocTranscript, mediaUrl } from '$lib/api';
  import { fmtTime } from '$lib/utils';
  import { Captions, ChevronRight, Maximize2, Minimize2 } from 'lucide-svelte';
  import TranscriptWindow from './transcript-window.svelte';
  import ChunkTimeline from './chunk-timeline.svelte';

  type Props = {
    hit: Hit | null;
    query?: string;
  };
  let { hit, query = '' }: Props = $props();

  // Sliding transcript window: how many chunks of context to show on each side
  // of the currently-playing chunk. Tunable; the window stays ≤ BEFORE+AFTER+1.
  const WINDOW_BEFORE = 1;
  const WINDOW_AFTER = 1;

  let mediaEl = $state<HTMLVideoElement | null>(null);
  let mediaError = $state<string | null>(null);

  // Whole-doc time + chunk envelope drive the timeline under the video. Svelte
  // polls currentTime via its own rAF (only while playing) through the
  // `bind:currentTime` on <video> — smooth, auto-cleaned, zero manual loop.
  let currentTime = $state(0);
  let duration = $state(0);
  let docChunks = $state<DocTranscriptChunk[]>([]);

  // Search hits ship `alignments: []` (the per-word timing blob is ~80% of a
  // search payload and only the player needs it). On opening a hit we lazy-fetch
  // the WHOLE document transcript (keyed on doc_id) so a sliding window of chunks
  // around the playhead can render past the selected chunk; atlas/selection hits
  // carry only the selected chunk so we can't rely on `hit.alignments`. The seek
  // $effect below sets currentTime to hit.start, and the windowed transcript
  // tracks it once these absolute-time chunks populate. `docChunks` is the single
  // source — we no longer flatten into one giant alignments array.
  $effect(() => {
    const h = hit;
    if (!h) {
      docChunks = [];
      return;
    }
    docChunks = [];
    let cancelled = false; // supersede guard + leak guard
    getDocTranscript(h.doc_id)
      .then((doc) => {
        if (!cancelled) docChunks = doc.chunks;
      })
      .catch(() => {
        /* leave empty — the video still plays, just no karaoke */
      });
    return () => {
      cancelled = true;
    };
  });

  // Which chunk the playhead is in. Reactive on currentTime + docChunks + hit.
  // 1) time-based: the chunk whose [start,end) contains currentTime.
  // 2) once playing but in an inter-chunk gap (silence between speech segments):
  //    clamp to the nearest PRECEDING chunk so the window stays at the playhead.
  //    This must beat the hit fallback — otherwise the window snaps back to the
  //    opened hit's chunk on every silence gap after the user has played past it.
  // 3) before playback (t === 0, pre-seek): fall back to the opened hit's segment
  //    so the right chunk is shown the instant a hit opens; else 0.
  const currentChunkIdx = $derived.by((): number => {
    const cs = docChunks;
    if (cs.length === 0) return 0;
    const t = currentTime;
    const i = cs.findIndex((c) => t >= c.start && t < c.end);
    if (i !== -1) return i;
    if (t > 0) {
      let last = 0;
      for (let k = 0; k < cs.length; k++) if (cs[k]!.start <= t) last = k;
      return last;
    }
    const h = hit;
    if (h) {
      const j = cs.findIndex((c) => c.speech_id === h.speech_id && c.chunk_id === h.chunk_id);
      if (j !== -1) return j;
    }
    return 0;
  });

  // The slice `lo` — the absolute index of the first windowed chunk. The window
  // child uses it to flag the playing block without re-deriving the window.
  const windowStartIdx = $derived(Math.max(0, currentChunkIdx - WINDOW_BEFORE));

  // Window of chunks around the playhead. As currentTime crosses into the next
  // chunk, currentChunkIdx increments → this slice shifts by one (prev drops,
  // next appears): the recycle behaviour, with zero manual rotation.
  const windowChunks = $derived.by((): DocTranscriptChunk[] => {
    const cs = docChunks;
    if (cs.length === 0) return [];
    const lo = windowStartIdx;
    const hi = Math.min(cs.length, currentChunkIdx + WINDOW_AFTER + 1);
    return cs.slice(lo, hi);
  });

  // "<speech_id>:<chunk_id>" of the opened hit → highlights its segment.
  const activeKey = $derived(hit ? `${hit.speech_id}:${hit.chunk_id}` : null);

  // Timeline click → jump there and play. Mirrors the hit-seek $effect; here
  // the doc is already loaded so we seek immediately.
  const seekTo = (t: number) => {
    if (!mediaEl) return;
    try {
      mediaEl.currentTime = t;
    } catch {
      // ignore — metadata may not be ready; the bound currentTime corrects it
    }
    mediaEl.play().catch(() => {});
  };

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

  // Archival metadata, collapsed by default into an accordion so it doesn't eat
  // the height the transcript needs — expand on demand.
  let showMeta = $state(false);

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
        bind:currentTime
        bind:duration
        controls
        controlslist="nofullscreen"
        preload="auto"
        src={mediaUrl(hit.doc_id)}
        class={isFullscreen
          ? 'min-h-0 w-full flex-1 bg-black object-contain'
          : 'aspect-video max-h-[45vh] w-full shrink-0 bg-black object-contain'}
      >
        <track kind="captions" />
      </video>

      {#if !isFullscreen}
        <ChunkTimeline
          chunks={docChunks}
          {duration}
          {currentTime}
          {activeKey}
          {currentChunkIdx}
          onSeek={seekTo}
        />
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

      <!-- Transcript body. Normal: scrolls below the bar inside the card with a
           guaranteed min-height floor so it never collapses to 0 in a short pane
           (the map view's right column is wide-but-short, which used to squeeze
           the flex-1 transcript to nothing). Fullscreen: a centred caption band
           lifted ABOVE the native control bar (bottom-20) so it never covers the
           play button — only the ≤3 windowed chunks render, so it stays small. -->
      <div
        class={isFullscreen
          ? 'absolute inset-x-0 bottom-20 mx-auto max-h-[32%] w-[min(92%,60rem)] overflow-y-auto rounded-xl bg-black/55 px-6 py-4 text-lg leading-8 text-white backdrop-blur-sm'
          : 'min-h-[8rem] flex-1 overflow-y-auto text-sm leading-7'}
      >
        <TranscriptWindow
          chunks={windowChunks}
          {currentChunkIdx}
          {windowStartIdx}
          media={mediaEl}
          {query}
          variant={isFullscreen ? 'overlay' : 'panel'}
        />
      </div>
    </div>

    {#if mediaError}
      <div class="shrink-0 text-sm text-destructive">
        Video failed to load: {mediaError}. Check
        <code>/api/media/{hit.doc_id}</code>.
      </div>
    {/if}

    {#if metaRows.length}
      <div class="shrink-0 overflow-hidden rounded-md border border-border bg-card/40 text-xs">
        <button
          type="button"
          onclick={() => (showMeta = !showMeta)}
          aria-expanded={showMeta}
          class="flex w-full items-center gap-1.5 px-3 py-1.5 text-muted-foreground transition-colors hover:bg-secondary/40"
        >
          <ChevronRight class="size-3.5 transition-transform {showMeta ? 'rotate-90' : ''}" />
          <span class="font-medium">Metadata</span>
          <span class="ml-auto text-[10px] text-muted-foreground/70">{metaRows.length} fields</span>
        </button>
        {#if showMeta}
          <dl class="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 border-t border-border px-3 py-2">
            {#each metaRows as [label, value] (label)}
              <dt class="text-muted-foreground">{label}</dt>
              <dd class="truncate font-medium text-foreground" title={value}>{value}</dd>
            {/each}
          </dl>
        {/if}
      </div>
    {/if}
  {/if}
</div>
