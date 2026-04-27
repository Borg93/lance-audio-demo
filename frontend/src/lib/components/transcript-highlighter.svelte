<script lang="ts">
  import type { Alignment } from '$lib/api';
  import { queryTerms } from '$lib/utils';

  type Props = {
    alignments: Alignment[];
    /** Live media element. We read `currentTime` and listen for `seeked`. */
    media: HTMLMediaElement | null;
    query?: string;
  };
  let { alignments, media, query = '' }: Props = $props();

  const terms = $derived(new Set(queryTerms(query)));

  /** Strip leading/trailing punctuation, lowercase. */
  const normalize = (w: string) => w.replace(/^\W+|\W+$/gu, '').toLowerCase();

  let scrollContainer = $state<HTMLDivElement | null>(null);

  /**
   * Karaoke cursor.
   *
   * The DOM (`scrollContainer`) and the data (`alignments`) are independent
   * reactive sources, so we drive the RAF loop from `$effect`. Whenever
   * either changes, the previous RAF + listeners are torn down via the
   * cleanup return, then the wordMap/sentMap are rebuilt against the
   * current DOM. This was the bug in my first port — `{@attach}` only
   * runs once per parent <div> mount, but the inner spans re-render on
   * every hit change, so the captured maps went stale.
   */
  $effect(() => {
    if (!scrollContainer || !media) return;

    type Ref = { el: HTMLElement; start: number; end: number };
    const wordMap: Ref[] = [];
    const sentMap: Ref[] = [];

    // Touch `alignments` so the effect re-runs when a new hit is selected
    // and the spans below have been replaced by Svelte's reconciler.
    void alignments;

    for (const el of scrollContainer.querySelectorAll<HTMLElement>('[data-word]')) {
      wordMap.push({
        el,
        start: parseFloat(el.dataset.start ?? '0'),
        end: parseFloat(el.dataset.end ?? '0'),
      });
    }
    for (const el of scrollContainer.querySelectorAll<HTMLElement>('[data-sentence]')) {
      sentMap.push({
        el,
        start: parseFloat(el.dataset.start ?? '0'),
        end: parseFloat(el.dataset.end ?? '0'),
      });
    }

    function find(segs: Ref[], t: number): Ref | null {
      let lo = 0,
        hi = segs.length - 1;
      while (lo <= hi) {
        const mid = (lo + hi) >> 1;
        if (t < segs[mid].start) hi = mid - 1;
        else if (t >= segs[mid].end) lo = mid + 1;
        else return segs[mid];
      }
      return null;
    }

    let rafId = 0;
    let prevWord: HTMLElement | null = null;
    let prevSent: HTMLElement | null = null;

    function refresh() {
      if (!media) return;
      const t = media.currentTime;
      const w = find(wordMap, t);
      if (w !== null && w.el !== prevWord) {
        prevWord?.classList.remove('cursor-word');
        w.el.classList.add('cursor-word');
        prevWord = w.el;
      }
      const s = find(sentMap, t);
      if (s !== null && s.el !== prevSent) {
        prevSent?.classList.remove('cursor-sentence');
        s.el.classList.add('cursor-sentence');
        s.el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        prevSent = s.el;
      }
    }

    // Always tick — even when paused, so seek-while-paused updates the cursor.
    // Skipping refresh when paused (the old behavior) meant the highlight
    // didn't track when the user dragged the playhead on a paused video.
    function tick() {
      refresh();
      rafId = requestAnimationFrame(tick);
    }

    media.addEventListener('seeked', refresh);
    media.addEventListener('timeupdate', refresh);
    rafId = requestAnimationFrame(tick);
    refresh();

    return () => {
      cancelAnimationFrame(rafId);
      media.removeEventListener('seeked', refresh);
      media.removeEventListener('timeupdate', refresh);
      prevWord?.classList.remove('cursor-word');
      prevSent?.classList.remove('cursor-sentence');
    };
  });
</script>

<div
  bind:this={scrollContainer}
  class="max-h-[340px] overflow-y-auto rounded-md border border-border bg-input p-3 text-sm leading-7"
>
  {#each alignments as a (a.start)}
    {@const sentEndsWithSpace = (a.text ?? '').endsWith(' ')}
    <span
      data-sentence
      data-start={a.start}
      data-end={a.end}
      class="rounded-sm transition-colors hover:bg-secondary/40 cursor-pointer"
      onclick={() => {
        if (media) {
          media.currentTime = a.start;
          media.play().catch(() => {});
        }
      }}
      onkeydown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          if (media) {
            media.currentTime = a.start;
            media.play().catch(() => {});
          }
        }
      }}
      role="button"
      tabindex="0"
    >
      {#each a.words ?? [] as w (w.start + ':' + w.text)}
        {@const stripped = normalize(w.text ?? '')}
        <span
          data-word
          data-start={w.start}
          data-end={w.end}
          class="rounded-sm"
          class:underline={terms.has(stripped)}
          class:decoration-highlight={terms.has(stripped)}
          class:decoration-2={terms.has(stripped)}
          class:underline-offset-2={terms.has(stripped)}
        >{w.text}</span>
      {/each}
    </span>
    {#if !sentEndsWithSpace}<br />{/if}
  {/each}
</div>
