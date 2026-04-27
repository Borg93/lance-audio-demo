<script lang="ts">
  import { Button, Input } from '$lib/components/ui';
  import FilterPopover from './filter-popover.svelte';
  import HelpPopover from './help-popover.svelte';
  import { Popover } from 'bits-ui';
  import type { SearchSpec, SearchMode } from '$lib/api';
  import { Paperclip, Search, X, ImagePlus, Settings2 } from 'lucide-svelte';

  type Props = {
    spec: SearchSpec;
    onsubmit?: (spec: SearchSpec) => void;
  };
  let { spec = $bindable(), onsubmit }: Props = $props();

  /**
   * Two independent dials:
   *
   *  1) `kind`: keyword | meaning | both    — what kind of match to run
   *  2) `style`: loose | phrase | fuzzy     — only relevant when keyword is involved
   *
   * Default = `both` + `loose` — the best general-purpose setting. Most users
   * never need to touch the advanced popover.
   */
  type Kind = 'keyword' | 'meaning' | 'both';
  type Style = 'loose' | 'phrase' | 'fuzzy';

  let kind = $state<Kind>(
    spec.mode === 'semantic' ? 'meaning'
      : spec.mode === 'fts' ? 'keyword'
      : 'both',
  );
  let style = $state<Style>(
    spec.phrase ? 'phrase' : spec.fuzziness === 2 ? 'fuzzy' : 'loose',
  );
  let rerank = $state(spec.rerank ?? false);
  // Hybrid blend: 0 = pure FTS, 1 = pure vector. Slider in [0, 100] for UI.
  // `null` = leave undefined → backend uses RRF (parameter-free fusion).
  let weightPct = $state<number | null>(
    spec.weight === undefined || spec.weight === null
      ? null
      : Math.round(spec.weight * 100),
  );
  let q = $state(spec.q);
  let imageFile = $state<File | null>(spec.image ?? null);

  let imagePreview: string | null = $state(null);
  $effect(() => {
    if (imageFile) {
      const url = URL.createObjectURL(imageFile);
      imagePreview = url;
      return () => URL.revokeObjectURL(url);
    }
    imagePreview = null;
  });

  /** Derive backend SearchMode from the two dials + image-attach state. */
  function buildSpec(): SearchSpec {
    let mode: SearchMode;
    const keywordInvolved = kind === 'keyword' || kind === 'both';
    if (imageFile && (kind === 'meaning' || kind === 'both')) mode = 'all';
    else if (imageFile) mode = 'visual';
    else if (kind === 'meaning') mode = 'semantic';
    else if (kind === 'both') mode = 'hybrid';
    else mode = 'fts';
    return {
      q: q.trim(),
      mode,
      phrase: keywordInvolved && style === 'phrase',
      fuzziness: keywordInvolved && style === 'fuzzy' ? 2 : 0,
      rerank,
      weight: kind === 'both' && weightPct !== null ? weightPct / 100 : undefined,
      image: imageFile,
      language: spec.language,
      namn: spec.namn,
      referenskod: spec.referenskod,
      extraid: spec.extraid,
      n: spec.n,
    };
  }

  function submit() {
    const next = buildSpec();
    spec = next;
    onsubmit?.(next);
  }

  let fileInput = $state<HTMLInputElement | null>(null);

  function dropZone(node: HTMLElement) {
    const onDragOver = (e: DragEvent) => e.preventDefault();
    const onDrop = (e: DragEvent) => {
      e.preventDefault();
      const f = e.dataTransfer?.files?.[0];
      if (f && f.type.startsWith('image/')) imageFile = f;
    };
    node.addEventListener('dragover', onDragOver);
    node.addEventListener('drop', onDrop);
    return () => {
      node.removeEventListener('dragover', onDragOver);
      node.removeEventListener('drop', onDrop);
    };
  }

  // Help popover examples — also reused as plain-language summary copy.
  const examples = {
    keyword: { label: 'Keyword',  example: 'betänkandet',  explain: 'Match transcripts that CONTAIN your words (in any order). Swedish stemmer also accepts inflections — "betänkandet" finds "betänkande" / "betänkanden" / "betänkandet".' },
    phrase:  { label: 'Phrase',   example: 'alkoholmonopolets framtid', explain: 'Words must appear in this EXACT order, side by side.' },
    fuzzy:   { label: 'Fuzzy',    example: 'betänkadet',   explain: 'Like Keyword but allows up to 2 letter typos per word — useful when unsure of spelling.' },
    meaning: { label: 'Meaning',  example: 'klimatkris',   explain: 'Vector search — finds chunks that DISCUSS the topic, even if those exact words aren\'t there. "klimat" can find "miljö" / "ekosystem".' },
    both:    { label: 'Both',     example: 'regeringens beslut', explain: 'Run Keyword AND Meaning together, fuse the rankings. Recommended default.' },
  };

  /** Single-line summary that always reflects the current setting. */
  const summary = $derived.by(() => {
    if (imageFile) {
      if (kind === 'meaning' || kind === 'both')
        return 'Image + your text → fused over transcript meaning AND visually similar frames.';
      return 'Image only → finds visually similar video frames. (Switch kind to Meaning or Both to also use your text.)';
    }
    if (kind === 'meaning') return examples.meaning.explain;
    if (kind === 'both') {
      return `Both kinds: ${examples.keyword.explain} PLUS ${examples.meaning.explain}`;
    }
    if (style === 'phrase') return examples.phrase.explain;
    if (style === 'fuzzy') return examples.fuzzy.explain;
    return examples.keyword.explain;
  });

  let advancedOpen = $state(false);
</script>

<div class="px-6 py-3" {@attach dropZone}>
  <form
    class="flex flex-wrap items-center gap-2"
    onsubmit={(e) => {
      e.preventDefault();
      submit();
    }}
  >
    <Button
      type="button"
      variant="outline"
      size="icon"
      title="Attach an image (drag-drop also works) — search by visual similarity"
      onclick={() => fileInput?.click()}
    >
      <Paperclip class="size-4" />
    </Button>
    <input
      bind:this={fileInput}
      type="file"
      accept="image/*"
      class="hidden"
      onchange={(e) => {
        const f = (e.currentTarget as HTMLInputElement).files?.[0];
        if (f && f.type.startsWith('image/')) imageFile = f;
        (e.currentTarget as HTMLInputElement).value = '';
      }}
    />

    {#if imagePreview && imageFile}
      <div class="flex items-center gap-1 rounded-md border border-primary bg-primary/10 px-2 py-1">
        <img src={imagePreview} alt="" class="h-7 w-auto rounded-sm" />
        <button
          type="button"
          title="Remove image"
          class="text-muted-foreground hover:text-foreground"
          onclick={() => (imageFile = null)}
        >
          <X class="size-3" />
        </button>
      </div>
    {/if}

    <Input
      bind:value={q}
      type="search"
      class="min-w-[260px] flex-1"
      placeholder="Search transcripts…"
    />

    <!-- ── Tune popover (the only "config" surface in the bar) ── -->
    <Popover.Root bind:open={advancedOpen}>
      <Popover.Trigger
        class="inline-flex h-9 items-center gap-1.5 rounded-md border border-border bg-input px-3 text-xs text-muted-foreground hover:text-foreground"
        title="Tune the search — pick keyword / meaning / both, match style, and reranking"
      >
        <Settings2 class="size-3.5" />
        <span>Tune</span>
        <span class="ml-1 rounded bg-secondary px-1.5 py-0.5 text-[10px] font-medium text-foreground">
          {kind === 'both' ? 'Both' : kind === 'meaning' ? 'Meaning' : `Keyword · ${style}`}
        </span>
      </Popover.Trigger>

      <Popover.Portal>
        <Popover.Content
          sideOffset={6}
          align="end"
          class="z-50 w-[420px] rounded-md border border-border bg-card p-4 text-xs shadow-md"
        >
          <!-- Question 1: KIND of search -->
          <div class="mb-3">
            <div class="mb-1.5 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
              1. Kind of search
            </div>
            <div class="grid gap-1">
              {#each (['keyword', 'meaning', 'both'] as const) as k (k)}
                <label class="flex items-start gap-2 rounded p-2 cursor-pointer hover:bg-secondary/40 {kind === k ? 'bg-secondary/60' : ''}">
                  <input type="radio" bind:group={kind} value={k} class="mt-0.5 accent-primary" />
                  <div class="flex-1">
                    <div class="font-medium">
                      {examples[k].label}
                      {#if k === 'both'}<span class="ml-1 text-[10px] text-primary">(recommended)</span>{/if}
                    </div>
                    <div class="text-muted-foreground">{examples[k].explain}</div>
                  </div>
                </label>
              {/each}
            </div>
          </div>

          <!-- Hybrid balance slider — only when "Both" -->
          {#if kind === 'both'}
            <div class="mb-3">
              <div class="mb-1.5 flex items-baseline justify-between">
                <span class="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
                  Balance
                </span>
                <button
                  type="button"
                  onclick={() => (weightPct = null)}
                  class="text-[10px] text-muted-foreground hover:text-foreground"
                  title="Use Lance's parameter-free RRF fusion (default)"
                >
                  reset → RRF
                </button>
              </div>
              <input
                type="range"
                min="0"
                max="100"
                step="5"
                value={weightPct ?? 50}
                oninput={(e) => (weightPct = Number((e.currentTarget as HTMLInputElement).value))}
                class="w-full accent-primary"
              />
              <div class="flex justify-between text-[10px] text-muted-foreground">
                <span>← keyword</span>
                <span>
                  {weightPct === null
                    ? 'RRF (auto-fuse, no weight)'
                    : weightPct === 50
                      ? 'balanced'
                      : weightPct < 50
                        ? `${100 - weightPct}% keyword`
                        : `${weightPct}% meaning`}
                </span>
                <span>meaning →</span>
              </div>
            </div>
          {/if}

          <!-- Question 2: keyword match style — only relevant when keyword involved -->
          {#if kind !== 'meaning'}
            <div class="mb-3">
              <div class="mb-1.5 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
                2. Keyword match style
              </div>
              <div class="grid gap-1">
                {#each (['loose', 'phrase', 'fuzzy'] as const) as s (s)}
                  <label class="flex items-start gap-2 rounded p-2 cursor-pointer hover:bg-secondary/40 {style === s ? 'bg-secondary/60' : ''}">
                    <input type="radio" bind:group={style} value={s} class="mt-0.5 accent-primary" />
                    <div class="flex-1">
                      <div class="font-medium">
                        {s === 'loose' ? 'Loose (default)' : examples[s].label}
                      </div>
                      <div class="text-muted-foreground">
                        {s === 'loose' ? 'Words anywhere in chunk; stem-aware.' : examples[s].explain}
                      </div>
                    </div>
                  </label>
                {/each}
              </div>
            </div>
          {/if}

          <!-- Quality knob -->
          <div>
            <label class="flex items-start gap-2 rounded p-2 cursor-pointer hover:bg-secondary/40">
              <input type="checkbox" bind:checked={rerank} class="mt-0.5 accent-primary" />
              <div class="flex-1">
                <div class="font-medium">Rerank top results <span class="ml-1 text-[10px] text-muted-foreground">(slower, more accurate)</span></div>
                <div class="text-muted-foreground">
                  After the initial search, run Qwen3-VL-Reranker on the top-100 candidates to re-sort by relevance.
                  Adds ~1–3 s.
                </div>
              </div>
            </label>
          </div>
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>

    <FilterPopover bind:spec onchange={submit} />

    <Button type="submit" size="default">
      <Search class="size-4" />
      Search
    </Button>

    <HelpPopover
      examples={examples as Record<string, { label: string; example: string; explain: string }>}
      onpick={(key, ex) => {
        if (key === 'phrase' || key === 'fuzzy') { kind = 'keyword'; style = key; }
        else if (key === 'meaning') kind = 'meaning';
        else if (key === 'both') { kind = 'both'; style = 'loose'; }
        else { kind = 'keyword'; style = 'loose'; }
        q = ex;
      }}
    />
  </form>

  <!-- Single-line plain-English summary, always visible. -->
  <div class="mt-2 flex items-baseline gap-2 text-[11px] text-muted-foreground">
    {#if imageFile}<ImagePlus class="size-3.5 text-primary self-center" />{/if}
    <span>{summary}</span>
  </div>
</div>
