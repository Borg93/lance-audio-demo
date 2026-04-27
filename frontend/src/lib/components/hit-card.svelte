<script lang="ts">
  import { type Hit, thumbnailUrl, chunkFrameUrl } from '$lib/api';
  import { features } from '$lib/feature-flags.svelte';
  import { fmtTime, queryTerms, escapeHtml } from '$lib/utils';
  import { cn } from '$lib/utils';

  type Props = {
    hit: Hit;
    query?: string;
    active?: boolean;
    /** "row" = side-by-side (list); "tile" = thumbnail on top (grid). */
    layout?: 'row' | 'tile';
    onclick?: () => void;
  };
  let { hit, query = '', active = false, layout = 'row', onclick }: Props = $props();

  const title = $derived(hit.namn ?? hit.audio_path ?? hit.doc_id);
  const terms = $derived(queryTerms(query));

  /** Wrap matching terms in a highlight span. Output is HTML-safe. */
  const highlighted = $derived.by(() => {
    const text = hit.text ?? '';
    if (!terms.length) return escapeHtml(text);
    const escaped = terms.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
    const re = new RegExp(`(${escaped.join('|')})`, 'giu');
    return escapeHtml(text).replace(
      re,
      '<mark class="bg-highlight/30 text-foreground rounded-sm">$1</mark>',
    );
  });
</script>

{#if layout === 'tile'}
  <button
    type="button"
    {onclick}
    aria-pressed={active}
    class={cn(
      'group flex h-full flex-col overflow-hidden rounded-lg border bg-card text-left transition-all',
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
      <span class="absolute bottom-1.5 left-1.5 rounded bg-black/70 px-1.5 py-0.5 font-mono text-[10px] text-white">
        {fmtTime(hit.start)}
      </span>
    </div>
    <div class="flex flex-1 flex-col gap-1 p-2.5">
      <div class="line-clamp-1 text-xs font-semibold leading-snug">{title}</div>
      <div class="font-mono text-[10px] text-muted-foreground">
        {fmtTime(hit.start)} → {fmtTime(hit.end)}
        {#if hit.referenskod}· {hit.referenskod}{/if}
      </div>
      <div class="line-clamp-3 text-xs leading-snug [overflow-wrap:anywhere]">
        <!-- highlighted is HTML-escaped + safe to inject -->
        {@html highlighted}
      </div>
    </div>
  </button>
{:else}
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
      <div class="line-clamp-2 text-sm font-semibold leading-snug [overflow-wrap:anywhere]">{title}</div>
      <div class="font-mono text-[11px] text-muted-foreground">
        {fmtTime(hit.start)} → {fmtTime(hit.end)}
        {#if hit.language}· {hit.language}{/if}
        {#if hit.referenskod}· {hit.referenskod}{/if}
      </div>
      <div class="line-clamp-3 text-sm leading-snug [overflow-wrap:anywhere]">
        <!-- highlighted is HTML-escaped + safe to inject -->
        {@html highlighted}
      </div>
    </div>
  </button>
{/if}
