<script lang="ts">
  import { Button, Input, Select, type SelectOption } from '$lib/components/ui';
  import FilterPopover from './filter-popover.svelte';
  import HelpPopover from './help-popover.svelte';
  import SearchSettings from './search-settings.svelte';
  import type { SearchSpec, SearchMode } from '$lib/api';
  import { Paperclip, Search, X, ImagePlus } from 'lucide-svelte';

  type Props = {
    spec: SearchSpec;
    onsubmit?: (spec: SearchSpec) => void;
  };
  let { spec = $bindable(), onsubmit }: Props = $props();

  /**
   * `kind` (Keyword | Vector | Hybrid) maps to the backend mode; `style`
   * (loose | phrase | fuzzy) only matters when keyword is involved. Both are
   * plain strings so they bind cleanly to the shadcn Select/RadioGroup.
   * Internal values stay 'meaning'/'both' to map onto modes semantic/hybrid.
   */
  let kind = $state<string>(
    spec.mode === 'semantic'
      ? 'meaning'
      : spec.mode === 'scene' || spec.mode === 'scene_fts'
        ? 'scene'
        : spec.mode === 'fts'
          ? 'keyword'
          : 'both',
  );
  // For the Scene kind: search the caption field by meaning (vector) or keyword (BM25).
  let sceneMethod = $state<string>(spec.mode === 'scene_fts' ? 'fts' : 'vector');
  let style = $state<string>(spec.phrase ? 'phrase' : spec.fuzziness === 2 ? 'fuzzy' : 'loose');
  let rerank = $state(spec.rerank ?? false);
  let rerankN = $state(String(spec.rerankN ?? 20));
  let resultN = $state(String(spec.n ?? 100));
  // Hybrid blend: 0 = pure FTS, 100 = pure vector; null = parameter-free RRF.
  let weightPct = $state<number | null>(
    spec.weight === undefined || spec.weight === null ? null : Math.round(spec.weight * 100),
  );
  let q = $state(spec.q);
  // Optional separate text for the VECTOR leg of hybrid; falls back to `q`.
  let qVec = $state(spec.qVec ?? '');
  let imageFile = $state<File | null>(spec.image ?? null);

  const kindOptions: SelectOption[] = [
    { value: 'keyword', label: 'Keyword' },
    { value: 'meaning', label: 'Vector' },
    { value: 'both', label: 'Hybrid' },
    { value: 'scene', label: 'Scene' },
  ];

  let imagePreview: string | null = $state(null);
  $effect(() => {
    if (imageFile) {
      const url = URL.createObjectURL(imageFile);
      imagePreview = url;
      return () => URL.revokeObjectURL(url);
    }
    imagePreview = null;
  });

  /** Derive backend SearchMode from the dial + image-attach state. */
  function buildSpec(): SearchSpec {
    let mode: SearchMode;
    const keywordInvolved = kind === 'keyword' || kind === 'both';
    const hasText = q.trim() !== '' || qVec.trim() !== '';
    // Image + any text → 3-way "all" (FTS + text-vector + frame-vector). Image
    // alone → visual. Otherwise the dial decides.
    if (imageFile && hasText) mode = 'all';
    else if (imageFile) mode = 'visual';
    else if (kind === 'meaning') mode = 'semantic';
    else if (kind === 'scene') mode = sceneMethod === 'fts' ? 'scene_fts' : 'scene';
    else if (kind === 'both') mode = 'hybrid';
    else mode = 'fts';
    return {
      q: q.trim(),
      qVec: qVec.trim() || undefined,
      mode,
      phrase: keywordInvolved && style === 'phrase',
      fuzziness: keywordInvolved && style === 'fuzzy' ? 2 : 0,
      rerank,
      rerankN: rerank ? Number(rerankN) : undefined,
      weight: kind === 'both' && weightPct !== null ? weightPct / 100 : undefined,
      n: Number(resultN) || 30,
      image: imageFile,
      // Filter fields are owned by <FilterPopover> on `spec` — pass them
      // through so a new search keeps the active filters (the old buildSpec
      // dropped `where`/`prefilter`, silently discarding the SQL filter).
      language: spec.language,
      namn: spec.namn,
      referenskod: spec.referenskod,
      extraid: spec.extraid,
      where: spec.where,
      prefilter: spec.prefilter,
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
    meaning: { label: 'Vector',   example: 'klimatkris',   explain: 'Vector search — finds chunks that DISCUSS the topic, even if those exact words aren\'t there. "klimat" can find "miljö" / "ekosystem".' },
    both:    { label: 'Hybrid',   example: 'regeringens beslut', explain: 'Run Keyword (FTS) AND Vector together, fuse the rankings. Recommended default.' },
    scene:   { label: 'Scene',    example: 'demonstranter med plakat', explain: 'Searches the AI-written Swedish caption of each video FRAME — finds clips by what is visible on screen, described in words. Complements Image search (which matches raw visuals).' },
  } satisfies Record<string, { label: string; example: string; explain: string }>;

  /** Placeholder for the single query box (non-hybrid modes). */
  const singlePlaceholder = $derived(
    kind === 'scene'
      ? 'Describe what\'s on screen…'
      : kind === 'meaning'
        ? 'Search by meaning…'
        : 'Search transcripts…',
  );

  /** Single-line summary that always reflects the current setting. */
  const summary = $derived.by(() => {
    if (imageFile) {
      if (kind === 'meaning' || kind === 'both')
        return 'Image + your text → fused over transcript meaning AND visually similar frames.';
      return 'Image only → finds visually similar video frames. (Switch to Vector or Hybrid to also use your text.)';
    }
    if (kind === 'scene') return examples.scene.explain;
    if (kind === 'meaning') return examples.meaning.explain;
    if (kind === 'both') return `Hybrid — ${examples.keyword.explain} PLUS ${examples.meaning.explain}`;
    if (style === 'phrase') return examples.phrase.explain;
    if (style === 'fuzzy') return examples.fuzzy.explain;
    return examples.keyword.explain;
  });
</script>

<div class="px-6 py-3" {@attach dropZone}>
  <form class="flex flex-col gap-2.5" onsubmit={(e) => { e.preventDefault(); submit(); }}>
    <!-- ── Row A: mode + actions ── -->
    <div class="flex flex-wrap items-center gap-2">
      <Select bind:value={kind} options={kindOptions} ariaLabel="Search mode" class="w-32" />
      <span class="hidden flex-1 truncate text-[11px] text-muted-foreground lg:inline">
        {summary}
      </span>
      <div class="ml-auto flex items-center gap-1.5">
        <Button
          type="button"
          variant="outline"
          size="default"
          title="Attach an image (drag-drop also works) — search by visual similarity"
          onclick={() => fileInput?.click()}
        >
          <Paperclip class="size-4" />
          Image
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
        <SearchSettings bind:resultN bind:rerank bind:rerankN bind:weightPct bind:style bind:sceneMethod {kind} />
        <FilterPopover bind:spec onchange={submit} />
        <HelpPopover
          {examples}
          onpick={(key, ex) => {
            if (key === 'phrase' || key === 'fuzzy') { kind = 'keyword'; style = key; }
            else if (key === 'meaning') kind = 'meaning';
            else if (key === 'scene') kind = 'scene';
            else if (key === 'both') { kind = 'both'; style = 'loose'; }
            else { kind = 'keyword'; style = 'loose'; }
            q = ex;
          }}
        />
      </div>
    </div>

    <!-- ── Row B: query input(s) + Search ── -->
    <!-- Same single row in every mode (consistent height); Hybrid just adds the
         second input. Labels live in the placeholders so the row never grows. -->
    {#if kind === 'both'}
      <div class="flex items-center gap-2">
        <Input
          bind:value={q}
          type="search"
          class="h-9 sm:w-52"
          placeholder="Keyword — exact words"
          aria-label="Keyword (FTS)"
        />
        <Input
          bind:value={qVec}
          type="search"
          class="h-9 flex-1 sm:max-w-2xl"
          placeholder="Vector — search by meaning (primary)"
          aria-label="Vector — search by meaning"
        />
        <Button type="submit" size="lg">
          <Search class="size-4" />
          Search
        </Button>
      </div>
    {:else}
      <div class="flex items-center gap-2">
        <Input bind:value={q} type="search" class="h-9 flex-1" placeholder={singlePlaceholder} />
        <Button type="submit" size="lg">
          <Search class="size-4" />
          Search
        </Button>
      </div>
    {/if}

    {#if imagePreview && imageFile}
      <div class="flex w-fit items-center gap-2 rounded-md border border-primary bg-primary/10 py-1 pl-1 pr-2">
        <img src={imagePreview} alt="" class="h-12 w-auto rounded-sm object-cover" />
        <span class="max-w-[12rem] truncate text-xs text-foreground" title={imageFile.name}>
          {imageFile.name}
        </span>
        <button
          type="button"
          title="Remove image"
          class="inline-flex size-6 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
          onclick={() => (imageFile = null)}
        >
          <X class="size-4" />
        </button>
      </div>
    {/if}

    <!-- Summary on smaller screens (hidden when shown inline in Row A). -->
    <div class="flex items-baseline gap-2 text-[11px] text-muted-foreground lg:hidden">
      {#if imageFile}<ImagePlus class="size-3.5 self-center text-primary" />{/if}
      <span>{summary}</span>
    </div>
  </form>
</div>
