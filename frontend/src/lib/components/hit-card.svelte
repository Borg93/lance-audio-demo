<script lang="ts">
  import { goto } from '$app/navigation';
  import { type Hit, thumbnailUrl, chunkFrameUrl, isVoiceHit, voiceBandOf } from '$lib/api';
  import { features } from '$lib/feature-flags.svelte';
  import { voiceSearch } from '$lib/voice-search.svelte';
  import { fmtTime, queryTerms, makeHighlighter, cn } from '$lib/utils';
  import { AudioLines } from 'lucide-svelte';

  type Props = {
    hit: Hit;
    query?: string;
    active?: boolean;
    /** "row" = side-by-side (list); "tile" = thumbnail on top (grid). */
    layout?: 'row' | 'tile';
    /**
     * Prebuilt highlighter (HTML-escapes + wraps query matches). A list renders
     * one card per hit, so the parent (hit-list / the grid) compiles the match
     * RegExp ONCE and passes it down — at ~1000 cards this avoids ~1000 regex
     * compilations. When omitted (standalone use), we fall back to deriving it
     * from `query` here so the card stays self-contained.
     */
    highlight?: (text: string) => string;
    onclick?: () => void;
  };
  let {
    hit,
    query = '',
    active = false,
    layout = 'row',
    highlight: highlightProp,
    onclick,
  }: Props = $props();

  const title = $derived(hit.namn ?? hit.audio_path ?? hit.doc_id);
  const highlight = $derived(highlightProp ?? makeHighlighter(queryTerms(query)));

  // ── Voice search ──
  // Voice-mode hits carry the matched diarized turn; render its speaker chip +
  // confidence badge. Ordinary text hits leave `voice` null (no extra DOM).
  const voice = $derived(isVoiceHit(hit) ? hit : null);
  const band = $derived(voice ? voiceBandOf(voice.turn_score) : null);
  const bandTitle = $derived(
    voice
      ? `Voice similarity ${voice.turn_score.toFixed(3)} (1 − cosine distance). ` +
          'Confidence bands are still calibrating.'
      : '',
  );

  /** Query-by-example: rank everywhere this hit's voice speaks. A voice hit
   *  anchors on its exact matched turn; a text hit anchors on whoever speaks
   *  at the chunk's midpoint — with the chunk start as a one-shot fallback,
   *  since the midpoint can land in a mid-speech diarization gap (ASR chunks
   *  and diarized turns disagree at boundaries). Auto-applies: the request
   *  queues on the shared store, and the Search page consumes it. */
  function findVoice() {
    const stem = hit.audio_path.replace(/\.[^.]+$/, '');
    if (voice) voiceSearch.request({ docId: hit.doc_id, turnId: voice.turn_id }, stem);
    else
      voiceSearch.request({ docId: hit.doc_id, t: (hit.start + hit.end) / 2 }, stem, {
        docId: hit.doc_id,
        t: hit.start,
      });
    // Only +page.svelte consumes `pending`, but HitCard also renders on /tree
    // (topic results panel) — navigate like player-pane does. goto('/') is a
    // no-op-safe same-route navigation when the Search page is already up.
    void goto('/');
  }
</script>

{#snippet voiceMeta()}
  {#if voice}
    <div class="flex flex-wrap items-center gap-1 pt-0.5">
      <span
        class="rounded-full border border-border bg-secondary px-1.5 py-px font-mono text-[10px] text-foreground"
      >
        {voice.speaker_label} · {fmtTime(voice.turn_start)}–{fmtTime(voice.turn_end)}
      </span>
      {#if band === 'strong'}
        <span
          title={bandTitle}
          class="rounded-full bg-emerald-500/15 px-1.5 py-px text-[10px] font-medium text-emerald-600 dark:text-emerald-400"
        >
          Strong match
        </span>
      {:else if band === 'possible'}
        <span
          title={bandTitle}
          class="rounded-full bg-amber-500/15 px-1.5 py-px text-[10px] font-medium text-amber-600 dark:text-amber-400"
        >
          Possible
        </span>
      {:else}
        <span
          title={bandTitle}
          class="rounded-full bg-muted px-1.5 py-px font-mono text-[10px] text-muted-foreground"
        >
          {voice.turn_score.toFixed(2)}
        </span>
      {/if}
    </div>
  {/if}
{/snippet}

{#if layout === 'tile'}
  <!-- Relative wrapper: the card root is a <button>, so the voice action must
       be a SIBLING overlay (nested buttons are invalid HTML). -->
  <div class="group relative h-full">
    <button
      type="button"
      {onclick}
      aria-pressed={active}
      class={cn(
        'flex h-full w-full flex-col overflow-hidden rounded-lg border bg-card text-left transition-all',
        'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
        active
          ? 'border-primary ring-2 ring-primary shadow-md -translate-y-0.5'
          : 'border-border hover:border-primary hover:shadow-md hover:-translate-y-0.5',
      )}
    >
      <div class="relative aspect-video w-full overflow-hidden bg-muted">
        <img
          src={thumbnailUrl(hit.doc_id)}
          loading="lazy"
          alt=""
          class="h-full w-full object-cover transition-transform group-hover:scale-105"
          onerror={(e) => ((e.currentTarget as HTMLImageElement).style.visibility = 'hidden')}
        />
        {#if !features.framesUnavailable}
          <img
            src={chunkFrameUrl(hit.doc_id, hit.speech_id, hit.chunk_id)}
            loading="lazy"
            alt=""
            class="absolute right-1.5 bottom-1.5 h-10 w-16 rounded border border-background bg-black object-cover shadow"
            onerror={(e) => {
              features.framesUnavailable = true;
              (e.currentTarget as HTMLImageElement).style.display = 'none';
            }}
          />
        {/if}
        <span
          class="absolute bottom-1.5 left-1.5 rounded bg-black/70 px-1.5 py-0.5 font-mono text-[10px] text-white"
        >
          {fmtTime(hit.start)}
        </span>
      </div>
      <div class="flex flex-1 flex-col gap-1 p-2.5">
        <div class="line-clamp-1 text-xs font-semibold leading-snug">{title}</div>
        <div class="font-mono text-[10px] text-muted-foreground">
          {fmtTime(hit.start)} → {fmtTime(hit.end)}
          {#if hit.referenskod}· {hit.referenskod}{/if}
        </div>
        {@render voiceMeta()}
        <div class="line-clamp-3 text-xs leading-snug [overflow-wrap:anywhere]">
          <!-- highlight() HTML-escapes then wraps matches — safe to inject -->
          {@html highlight(hit.text ?? '')}
        </div>
        {#if hit.caption}
          <div class="line-clamp-2 text-[10px] italic text-muted-foreground" title={hit.caption}>
            🎬 {hit.caption}
          </div>
        {/if}
      </div>
    </button>
    {#if voiceSearch.built}
      <button
        type="button"
        title="Find this voice — everywhere this speaker talks, across videos"
        aria-label="Find this voice"
        onclick={findVoice}
        class="absolute top-1.5 right-1.5 z-[2] inline-flex size-6 items-center justify-center rounded-md bg-black/60 text-white opacity-0 transition-opacity hover:bg-primary focus-visible:opacity-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring group-hover:opacity-100"
      >
        <AudioLines class="size-3.5" />
      </button>
    {/if}
  </div>
{:else}
  <div class="group relative">
    <button
      type="button"
      {onclick}
      aria-pressed={active}
      class={cn(
        'flex w-full items-start gap-3 border-b border-border px-3 py-2.5 text-left transition-colors',
        'hover:bg-secondary/40',
        // ring-inset keeps the highlight inside the row's box so it survives
        // the parent's overflow-clip; bg + thick left bar make selection
        // unmistakable in both light and dark themes.
        active &&
          'bg-primary/15 ring-2 ring-inset ring-primary z-[1] relative shadow-[inset_4px_0_0_0_var(--color-primary)]',
      )}
    >
      <div class="relative flex-none">
        <img
          src={thumbnailUrl(hit.doc_id)}
          loading="lazy"
          alt=""
          class="h-[54px] w-[96px] rounded bg-black object-cover"
          onerror={(e) => ((e.currentTarget as HTMLImageElement).style.visibility = 'hidden')}
        />
        {#if !features.framesUnavailable}
          <img
            src={chunkFrameUrl(hit.doc_id, hit.speech_id, hit.chunk_id)}
            loading="lazy"
            alt=""
            class="absolute -right-0.5 -bottom-0.5 h-5 w-9 rounded-sm border border-background bg-black object-cover"
            onerror={(e) => {
              features.framesUnavailable = true;
              (e.currentTarget as HTMLImageElement).style.display = 'none';
            }}
          />
        {/if}
      </div>

      <div class="min-w-0 flex-1 space-y-0.5">
        <div class="line-clamp-2 text-sm font-semibold leading-snug [overflow-wrap:anywhere]">
          {title}
        </div>
        <div class="font-mono text-[11px] text-muted-foreground">
          {fmtTime(hit.start)} → {fmtTime(hit.end)}
          {#if hit.language}· {hit.language}{/if}
          {#if hit.referenskod}· {hit.referenskod}{/if}
        </div>
        {@render voiceMeta()}
        <div class="line-clamp-3 text-sm leading-snug [overflow-wrap:anywhere]">
          <!-- highlight() HTML-escapes then wraps matches — safe to inject -->
          {@html highlight(hit.text ?? '')}
        </div>
        {#if hit.caption}
          <div class="line-clamp-1 text-[11px] italic text-muted-foreground" title={hit.caption}>
            🎬 {hit.caption}
          </div>
        {/if}
      </div>
    </button>
    {#if voiceSearch.built}
      <button
        type="button"
        title="Find this voice — everywhere this speaker talks, across videos"
        aria-label="Find this voice"
        onclick={findVoice}
        class="absolute top-2 right-2 z-[2] inline-flex size-6 items-center justify-center rounded-md border border-border bg-card text-muted-foreground opacity-0 shadow-sm transition-opacity hover:text-primary focus-visible:opacity-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring group-hover:opacity-100"
      >
        <AudioLines class="size-3.5" />
      </button>
    {/if}
  </div>
{/if}
