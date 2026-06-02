<script lang="ts">
  /**
   * Guide — the in-app explainer. Two tabs: how Search works (the full,
   * animated walk-through that used to be crammed in the `?` popover) and how
   * the Atlas map works. Self-contained: no props, no fetching. Scroll-reveal
   * via a tiny IntersectionObserver action; the pipeline + lanes use CSS
   * animations so the "several searches at once, then merge" flow reads as
   * motion. Content verified against backend/search/service.py.
   */
  import { Search, Map, Sparkles, ArrowRight } from 'lucide-svelte';

  let tab = $state<'search' | 'atlas'>('search');

  /** Add `.in-view` the first time a node scrolls into view → CSS does the rest. */
  function reveal(node: HTMLElement, opts: { delay?: number } = {}) {
    node.style.transitionDelay = `${opts.delay ?? 0}ms`;
    const io = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (e.isIntersecting) {
            node.classList.add('in-view');
            io.unobserve(node);
          }
        }
      },
      { threshold: 0.12, rootMargin: '0px 0px -6% 0px' },
    );
    io.observe(node);
    return { destroy: () => io.disconnect() };
  }

  // The four signals / "judges", in the canonical order.
  const JUDGES = [
    {
      icon: '⌨️',
      name: 'Keyword',
      mode: 'fts',
      accent: 'border-l-sky-400',
      what: 'exact words you type (FTS / BM25 on the transcript)',
      give: 'your query words',
      vs: 'transcript text (BM25). Good at exact terms, names.',
    },
    {
      icon: '💬',
      name: 'Vector',
      mode: 'semantic',
      accent: 'border-l-primary',
      what: 'what a clip is about, even in other words (text vector)',
      give: 'your text → a vector',
      vs: 'text_embedding of each transcript. Good at topics, paraphrases.',
    },
    {
      icon: '🖼️',
      name: 'Image',
      mode: 'visual',
      accent: 'border-l-amber-400',
      what: 'how much the video frame looks like your image (frame vector)',
      give: 'your image → a vector',
      vs: 'frame_embedding of each video frame. Good at visually similar scenes.',
    },
    {
      icon: '🎬',
      name: 'Scene',
      mode: 'scene',
      accent: 'border-l-emerald-400',
      what: "what's visible on screen, from each frame's Swedish caption (caption vector)",
      give: 'your text → a vector',
      vs: 'caption_embedding of each frame’s Swedish caption. Good at what’s on screen ("plakat", "snöig gata").',
    },
  ];

  // Lanes for the "parallel passes" diagram (the order text reads left→right).
  const LANES = [
    { name: 'Keyword', sub: 'transcript words', scan: 'bg-sky-400', text: 'text-sky-500 dark:text-sky-300' },
    { name: 'Vector', sub: 'transcript meaning', scan: 'bg-primary', text: 'text-primary' },
    { name: 'Image', sub: 'video frames', scan: 'bg-amber-400', text: 'text-amber-500 dark:text-amber-300' },
    { name: 'Scene', sub: 'frame captions', scan: 'bg-emerald-400', text: 'text-emerald-500 dark:text-emerald-300' },
  ];

  // Per-mode pass count (how many independent searches each UI mode fires).
  const MODES = [
    { mode: 'Keyword', passes: '1 pass', detail: 'BM25 word-match on the transcript', fuse: '—' },
    { mode: 'Vector', passes: '1 pass', detail: 'meaning-match on the transcript', fuse: '—' },
    { mode: 'Scene', passes: '1 pass', detail: 'matches the frame’s Swedish caption (meaning or keyword)', fuse: '—' },
    { mode: 'Image', passes: '1 pass', detail: 'image-similarity on the video frames (no text)', fuse: '—' },
    { mode: 'Hybrid', passes: '2 passes', detail: 'Keyword + Vector on the transcript', fuse: 'RRF, or the Balance slider' },
    {
      mode: 'Image + text',
      passes: '≤ 4 passes',
      detail: 'Keyword + Vector + Image + Scene — the dial is ignored',
      fuse: 'equal-weight RRF',
    },
  ];

  // RRF worked example — two judges on 5 clips (A–E). Rank shown 1-based for
  // readability; score = Σ 1/(60 + rank) over the lists a clip appears in.
  const K = 60;
  const keywordRanked = ['C', 'A', 'B'];
  const meaningRanked = ['A', 'C', 'D'];
  const rrfRows = (() => {
    const rankOf = (list: string[]) =>
      Object.fromEntries(list.map((c, i) => [c, i + 1])) as Record<string, number>;
    const kr = rankOf(keywordRanked);
    const mr = rankOf(meaningRanked);
    const clips = [...new Set([...keywordRanked, ...meaningRanked])];
    return clips
      .map((clip) => {
        const parts: { src: string; rank: number }[] = [];
        if (kr[clip]) parts.push({ src: '⌨️', rank: kr[clip] });
        if (mr[clip]) parts.push({ src: '💬', rank: mr[clip] });
        const score = parts.reduce((a, p) => a + 1 / (K + p.rank), 0);
        return { clip, parts, score, inBoth: parts.length === 2 };
      })
      .sort((a, b) => b.score - a.score);
  })();

  const SETTINGS = [
    { name: 'Results to return', desc: 'How many chunks to show (N, default 100).' },
    { name: 'Rerank top', desc: 'Head size K the cross-encoder re-reads (default 20).' },
    { name: 'Balance', desc: 'Keyword ↔ meaning weight. Hybrid only (ignored once an image is added).' },
    { name: 'Match style', desc: 'loose words, exact phrase, or fuzzy (typo-tolerant) keyword matching.' },
  ];

  const LIMITS = [
    'Image search = visual frame similarity, not face / identity recognition.',
    'No speaker diarization — nothing links who is on screen to who is speaking.',
    'The reranker is text-only; it never uses the image.',
    'Multi-leg fusion (up to 4 judges) is equal-weight — there is no per-leg weight yet.',
    'Scene depends on AI captions — only as accurate as the frame captioner.',
  ];
</script>

<div class="h-full overflow-y-auto">
  <div class="mx-auto max-w-3xl px-6 py-8">
    <!-- header + tabs -->
    <div class="mb-6">
      <div class="text-primary mb-1 flex items-center gap-2 text-xs font-medium tracking-wide uppercase">
        <Sparkles class="size-3.5" /> Guide
      </div>
      <h1 class="text-foreground text-2xl font-semibold tracking-tight">How the viewer works</h1>
      <div class="border-border mt-4 flex gap-1 border-b">
        <button
          type="button"
          onclick={() => (tab = 'search')}
          class={'flex items-center gap-1.5 border-b-2 px-3 py-2 text-sm font-medium transition-colors ' +
            (tab === 'search'
              ? 'border-primary text-foreground'
              : 'text-muted-foreground hover:text-foreground border-transparent')}
        >
          <Search class="size-4" /> Search
        </button>
        <button
          type="button"
          onclick={() => (tab = 'atlas')}
          class={'flex items-center gap-1.5 border-b-2 px-3 py-2 text-sm font-medium transition-colors ' +
            (tab === 'atlas'
              ? 'border-primary text-foreground'
              : 'text-muted-foreground hover:text-foreground border-transparent')}
        >
          <Map class="size-4" /> Atlas
        </button>
      </div>
    </div>

    {#if tab === 'search'}
      <!-- ░░ SEARCH ░░ -->
      <!-- 0 · frame -->
      <div use:reveal class="reveal bg-card border-border mb-6 rounded-lg border px-4 py-3 text-sm">
        <span class="text-foreground">Every result is one </span>
        <strong class="text-primary">chunk</strong>
        <span class="text-muted-foreground">
          = a short transcript span + the video frame from that moment. Search ranks chunks.
        </span>
      </div>

      <!-- 1 · the headline mental model -->
      <div
        use:reveal
        class="reveal border-primary/40 bg-primary/10 text-foreground mb-8 rounded-lg border px-4 py-3 text-sm leading-relaxed"
      >
        <strong>The one idea:</strong> each signal you turn on runs its <em>own complete search</em> over
        every chunk — at the same time. When more than one runs, the winner is whatever ranks high across
        the <em>most</em> of those searches. Nothing searches "inside" another's results.
      </div>

      <!-- 2 · MULTI-PASS vs PREFILTER — the centerpiece -->
      <section use:reveal class="reveal mb-10">
        <h2 class="text-foreground mb-1 text-lg font-semibold">
          Is "image + keyword/vector" multiple passes, or a prefilter?
        </h2>
        <p class="text-muted-foreground mb-4 text-sm leading-relaxed">
          <strong class="text-foreground">Multiple independent passes, merged by rank — never a prefilter.</strong>
          The moment you attach an image <em>and</em> have text, the Keyword/Vector/Hybrid dial stops
          mattering: the tool runs up to four searches side by side, each scanning the whole library on its
          own, then merges their rankings.
        </p>

        <!-- parallel-lanes diagram -->
        <div class="bg-card border-border rounded-xl border p-4">
          <div class="text-muted-foreground mb-2 text-center text-[11px]">
            every pass starts from the <strong class="text-foreground">same full library</strong> — all at once
          </div>
          <div class="grid grid-cols-2 gap-2 sm:grid-cols-4">
            {#each LANES as lane, i (lane.name)}
              <div
                use:reveal={{ delay: i * 80 }}
                class="reveal lane-card bg-surface2 border-border relative overflow-hidden rounded-lg border p-2.5 text-center"
              >
                <div class="{lane.text} text-xs font-semibold">{lane.name}</div>
                <div class="text-muted-foreground text-[10px]">{lane.sub}</div>
                <div class="text-muted-foreground mt-1 text-[10px]">→ own ranked list</div>
                <span class="lane-scan {lane.scan}"></span>
              </div>
            {/each}
          </div>
          <div class="text-muted-foreground my-1 text-center text-sm">▼&ensp;▼&ensp;▼&ensp;▼</div>
          <div class="border-primary/50 bg-primary/10 text-foreground rounded-lg border px-3 py-2 text-center text-sm font-medium">
            Merge by rank — RRF (equal weight). High in <em>more</em> lists → wins.
          </div>
          <div class="text-muted-foreground my-1 text-center text-sm">▼</div>
          <div class="bg-background border-border text-foreground rounded-lg border px-3 py-2 text-center text-sm font-semibold">
            Top N results
          </div>
        </div>

        <!-- DO / DON'T -->
        <div class="mt-3 grid gap-3 sm:grid-cols-2">
          <div use:reveal class="reveal rounded-lg border border-emerald-500/40 bg-emerald-500/5 p-3 text-xs leading-relaxed">
            <div class="mb-1 font-semibold text-emerald-600 dark:text-emerald-300">✓ What actually happens</div>
            <p class="text-muted-foreground">
              Four searchlights sweep the same full library at once; their rankings are pooled. A chunk the
              image never matched can still rank #1 on the strength of its transcript alone — the image is a
              <strong class="text-foreground">voice in the vote</strong>, not a gate.
            </p>
          </div>
          <div use:reveal={{ delay: 90 }} class="reveal rounded-lg border border-red-500/40 bg-red-500/5 p-3 text-xs leading-relaxed">
            <div class="mb-1 font-semibold text-red-600 line-through dark:text-red-300">✗ Not this</div>
            <p class="text-muted-foreground">
              The image runs first, grabs a shortlist of look-alike clips, and the text then searches
              <em>only inside</em> that shortlist. <strong class="text-foreground">No such funnel exists.</strong>
              (Only the <em>Filters</em> panel narrows first — see below.)
            </p>
          </div>
        </div>

        <!-- traced example -->
        <div
          use:reveal
          class="reveal bg-surface2 border-border mt-3 rounded-lg border p-3 text-xs leading-relaxed"
        >
          <div class="text-foreground mb-1 font-medium">Traced: <code class="font-mono">"arbete"</code> + a photo of a snowy street</div>
          <p class="text-muted-foreground">
            Four passes fire at once. <strong class="text-foreground">Keyword</strong> ranks a labour-market
            briefing #1; <strong class="text-foreground">Vector</strong> surfaces a clip about
            "sysselsättning" (no exact word) at #2; <strong class="text-foreground">Image</strong> ranks an
            outdoor winter stakeout by frame-similarity; <strong class="text-foreground">Scene</strong> lifts a
            frame captioned "person på snöig gata". A clip that lands #3 / #5 / #3 / #2 across all four lists
            beats the briefing that was #1 in only one — agreement wins. The photo never walled off a subset
            that "arbete" then searched within.
          </p>
        </div>
      </section>

      <!-- 2b · order of operations / priority -->
      <section use:reveal class="reveal mb-10">
        <h2 class="text-foreground mb-1 text-lg font-semibold">In what order does it happen? (and what's prioritised)</h2>
        <p class="text-muted-foreground mb-4 text-sm leading-relaxed">
          The search legs have <strong class="text-foreground">no priority over each other</strong> — they're
          equal-weight and run at the same time. So "image + Hybrid (FTS + Vector)" doesn't rank one before
          another; the only real ordering is these pipeline stages:
        </p>
        <ol class="space-y-3">
          <li use:reveal class="reveal flex gap-3">
            <span class="bg-primary text-primary-foreground grid size-6 shrink-0 place-items-center rounded-full text-xs font-semibold">1</span>
            <div>
              <div class="text-foreground text-sm font-medium">Filters first <span class="text-muted-foreground font-normal">(optional)</span></div>
              <p class="text-muted-foreground mt-0.5 text-xs leading-relaxed">
                Your metadata Filters (language, name, referenskod, extraid, raw SQL) become a
                <code class="font-mono">WHERE</code> applied to <strong class="text-foreground">every leg</strong>:
                with <em>prefilter</em> on, the corpus is narrowed <em>before</em> each leg searches; with it off,
                <em>after</em>. This is the only narrowing step. No filters set → the whole library.
              </p>
            </div>
          </li>
          <li use:reveal={{ delay: 70 }} class="reveal flex gap-3">
            <span class="bg-primary text-primary-foreground grid size-6 shrink-0 place-items-center rounded-full text-xs font-semibold">2</span>
            <div>
              <div class="text-foreground text-sm font-medium">Every leg runs in parallel</div>
              <p class="text-muted-foreground mt-0.5 text-xs leading-relaxed">
                Keyword, Vector, Image and Scene each search independently and <strong class="text-foreground">at
                the same time</strong>. No leg goes first and none outranks another. Each over-fetches ~3×N
                candidates so good chunks aren't lost before the merge.
              </p>
            </div>
          </li>
          <li use:reveal={{ delay: 140 }} class="reveal flex gap-3">
            <span class="bg-primary text-primary-foreground grid size-6 shrink-0 place-items-center rounded-full text-xs font-semibold">3</span>
            <div>
              <div class="text-foreground text-sm font-medium">Fuse by rank — equal weight (RRF)</div>
              <p class="text-muted-foreground mt-0.5 text-xs leading-relaxed">
                Each leg's ranked list contributes <code class="font-mono">1/(60 + rank)</code> per chunk; a chunk
                high in <em>more</em> lists wins. Every leg counts the same. (Hybrid <em>without</em> an image may
                instead use the Balance slider — a 2-way blend of raw scores; adding an image always reverts to
                equal-weight RRF.)
              </p>
            </div>
          </li>
          <li use:reveal={{ delay: 210 }} class="reveal flex gap-3">
            <span class="bg-primary text-primary-foreground grid size-6 shrink-0 place-items-center rounded-full text-xs font-semibold">4</span>
            <div>
              <div class="text-foreground text-sm font-medium">Rerank the head <span class="text-muted-foreground font-normal">(optional)</span></div>
              <p class="text-muted-foreground mt-0.5 text-xs leading-relaxed">
                A text-only cross-encoder re-scores just the top <em>K</em> of the fused list, reading each
                chunk's transcript (the <code class="font-mono">text</code> column) against your text — see below.
              </p>
            </div>
          </li>
          <li use:reveal={{ delay: 280 }} class="reveal flex gap-3">
            <span class="bg-primary text-primary-foreground grid size-6 shrink-0 place-items-center rounded-full text-xs font-semibold">5</span>
            <div>
              <div class="text-foreground text-sm font-medium">Trim to top N</div>
              <p class="text-muted-foreground mt-0.5 text-xs leading-relaxed">
                The fused (and optionally reranked) list is cut to N — the results you see.
              </p>
            </div>
          </li>
        </ol>
      </section>

      <!-- 3 · the 4 judges -->
      <section use:reveal class="reveal mb-10">
        <h2 class="text-foreground mb-1 text-lg font-semibold">The 4 judges</h2>
        <p class="text-muted-foreground mb-4 text-sm leading-relaxed">
          A search asks up to four independent judges to rank the chunks. Each runs as its
          <em>own pass</em> over its <em>own</em> space and hands back its <em>own</em> ranked list —
          they never compare scores with each other.
        </p>
        <div class="grid gap-3 sm:grid-cols-2">
          {#each JUDGES as j, i (j.mode)}
            <div
              use:reveal={{ delay: i * 90 }}
              class="reveal bg-surface2 border-border {j.accent} rounded-lg border border-l-[3px] p-3"
            >
              <div class="flex items-center gap-2">
                <span class="text-base leading-none">{j.icon}</span>
                <span class="text-foreground font-medium">{j.name}</span>
                <code class="bg-card text-primary ml-auto rounded px-1.5 py-0.5 font-mono text-[11px]">
                  mode: {j.mode}
                </code>
              </div>
              <p class="text-muted-foreground mt-1.5 text-xs leading-relaxed">{j.what}</p>
            </div>
          {/each}
        </div>
      </section>

      <!-- 4 · what each judge does (table) -->
      <section use:reveal class="reveal mb-10">
        <h2 class="text-foreground mb-3 text-lg font-semibold">What each judge actually does</h2>
        <div class="border-border overflow-hidden rounded-lg border text-xs">
          <div class="bg-surface2 text-foreground grid grid-cols-[84px_1fr_1.4fr] font-medium">
            <div class="border-border border-b p-2">Judge</div>
            <div class="border-border border-b border-l p-2">You give it</div>
            <div class="border-border border-b border-l p-2">Compared against</div>
          </div>
          {#each JUDGES as j, i (j.mode)}
            <div
              use:reveal={{ delay: i * 70 }}
              class="reveal text-muted-foreground grid grid-cols-[84px_1fr_1.4fr] last:[&>div]:border-b-0"
            >
              <div class="border-border text-foreground border-b p-2">{j.icon} {j.name}</div>
              <div class="border-border border-b border-l p-2">{j.give}</div>
              <div class="border-border border-b border-l p-2">{j.vs}</div>
            </div>
          {/each}
        </div>
        <div
          use:reveal
          class="reveal border-primary/40 bg-primary/5 text-muted-foreground mt-3 rounded-lg border border-dashed p-3 text-sm"
        >
          💡 Your <strong class="text-foreground">text</strong> drives <strong class="text-foreground">two</strong>
          passes at once: the <span class="text-primary font-medium">Meaning</span> leg (what was <em>said</em>)
          and the <span class="font-medium text-emerald-500 dark:text-emerald-300">Scene</span> leg (what's
          <em>shown</em>, from the frame caption).
        </div>
      </section>

      <!-- 5 · per-mode pass table -->
      <section use:reveal class="reveal mb-10">
        <h2 class="text-foreground mb-3 text-lg font-semibold">How many passes each mode runs</h2>
        <div class="border-border overflow-hidden rounded-lg border text-xs">
          <div class="bg-surface2 text-foreground grid grid-cols-[1fr_auto_1.6fr] font-medium">
            <div class="border-border border-b p-2">Mode</div>
            <div class="border-border border-b border-l p-2 text-center">Passes</div>
            <div class="border-border border-b border-l p-2">What runs · fused by</div>
          </div>
          {#each MODES as m, i (m.mode)}
            <div
              use:reveal={{ delay: i * 60 }}
              class="reveal text-muted-foreground grid grid-cols-[1fr_auto_1.6fr] last:[&>div]:border-b-0 {m.mode ===
              'Image + text'
                ? 'bg-highlight/10'
                : ''}"
            >
              <div class="border-border text-foreground border-b p-2 font-medium">{m.mode}</div>
              <div class="border-border border-b border-l p-2 text-center font-mono">{m.passes}</div>
              <div class="border-border border-b border-l p-2">
                {m.detail}{#if m.fuse !== '—'}&nbsp;· <span class="text-foreground">{m.fuse}</span>{/if}
              </div>
            </div>
          {/each}
        </div>
        <div
          use:reveal
          class="reveal border-amber-500/40 bg-amber-500/5 text-muted-foreground mt-3 rounded-lg border p-3 text-sm leading-relaxed"
        >
          ⚠️ <strong class="text-foreground">Attaching an image overrides the dial.</strong> Image + text always
          becomes the 4-way <code class="font-mono">all</code> mode regardless of whether you picked Keyword,
          Vector, or Hybrid. Image alone stays single-pass Image; text alone obeys the dial.
        </div>
      </section>

      <!-- 6 · the pipeline -->
      <section use:reveal class="reveal mb-10">
        <h2 class="text-foreground mb-4 text-lg font-semibold">The whole pipeline</h2>
        <div class="flex flex-col items-center">
          <div class="grid w-full grid-cols-3 gap-2">
            <div class="bg-surface2 border-border border-l-[3px] border-l-sky-400 rounded-md border px-2 py-2 text-center text-xs">⌨️ Keyword text</div>
            <div class="bg-surface2 border-border border-l-[3px] border-l-primary rounded-md border px-2 py-2 text-center text-xs">💬 Meaning text</div>
            <div class="bg-surface2 border-border border-l-[3px] border-l-amber-400 rounded-md border px-2 py-2 text-center text-xs">🖼️ Image</div>
          </div>
          <div class="text-muted-foreground grid w-full grid-cols-3 text-center text-[10px]">
            <span>▼</span><span>▼ embed</span><span>▼ embed</span>
          </div>
          <div class="grid w-full grid-cols-3 gap-2">
            <div class="bg-background border-border rounded-md border px-2 py-1.5 text-center text-[11px]">FTS leg<div class="text-muted-foreground">BM25 on text</div></div>
            <div class="bg-background border-border rounded-md border px-2 py-1.5 text-center text-[11px]">Text-vector leg<div class="text-muted-foreground">text_embedding</div></div>
            <div class="bg-background border-border rounded-md border px-2 py-1.5 text-center text-[11px]">Frame-vector leg<div class="text-muted-foreground">frame_embedding</div></div>
          </div>
          <p class="text-muted-foreground mt-2 text-center text-[11px]">
            + the text query also drives a <strong class="text-foreground">Scene</strong> leg
            (<code class="font-mono">caption_embedding</code>) in <code class="font-mono">all</code>.
          </p>

          <div class="flow-line"><span class="flow-dot"></span></div>

          <div class="border-primary/50 bg-primary/10 text-foreground w-full rounded-lg border px-4 py-2.5 text-center text-sm font-medium">
            Fuse into one ranking — <span class="font-mono">RRF</span> (default) or Balance slider (hybrid only)
          </div>

          <div class="flow-line"><span class="flow-dot" style="animation-delay:.3s"></span></div>

          <div class="bg-surface2 border-border w-full rounded-md border px-4 py-2 text-center text-sm">
            Rerank top <em>K</em> <span class="text-muted-foreground">(optional)</span> — re-read each transcript vs your text
          </div>

          <div class="flow-line"><span class="flow-dot" style="animation-delay:.6s"></span></div>

          <div class="bg-card border-border text-foreground w-full rounded-lg border px-4 py-2.5 text-center font-semibold">
            Top N ranked chunk results
          </div>
        </div>
      </section>

      <!-- 7 · RRF worked example -->
      <section use:reveal class="reveal mb-10">
        <h2 class="text-foreground mb-1 text-lg font-semibold">How the lists merge: RRF (worked example)</h2>
        <p class="text-muted-foreground mb-3 text-sm leading-relaxed">
          Two judges run on 5 clips (A–E) and each returns a ranked list:
        </p>
        <div class="grid gap-3 sm:grid-cols-2">
          <div use:reveal class="reveal bg-surface2 border-border rounded-lg border p-3 text-xs">
            <div class="text-foreground mb-1 font-medium">⌨️ Keyword</div>
            <div class="text-muted-foreground">
              {#each keywordRanked as c, i (c)}<span class="text-foreground font-mono">{i + 1} = {c}</span>{#if i < keywordRanked.length - 1}&ensp;·&ensp;{/if}{/each}
            </div>
          </div>
          <div use:reveal={{ delay: 90 }} class="reveal bg-surface2 border-border rounded-lg border p-3 text-xs">
            <div class="text-foreground mb-1 font-medium">💬 Meaning</div>
            <div class="text-muted-foreground">
              {#each meaningRanked as c, i (c)}<span class="text-foreground font-mono">{i + 1} = {c}</span>{#if i < meaningRanked.length - 1}&ensp;·&ensp;{/if}{/each}
            </div>
          </div>
        </div>
        <p class="text-muted-foreground mt-3 mb-2 text-sm">
          Each clip scores <code class="text-primary bg-surface2 rounded px-1 font-mono text-[11px]">Σ 1/(60 + rank)</code>
          over the lists it appears in:
        </p>
        <div class="bg-card border-border overflow-hidden rounded-lg border">
          {#each rrfRows as row, i (row.clip)}
            <div
              use:reveal={{ delay: i * 100 }}
              class="reveal border-border flex items-center gap-3 border-b px-3 py-2 text-xs last:border-b-0 {row.inBoth
                ? 'bg-highlight/10'
                : ''}"
            >
              <span class="text-foreground w-6 font-mono font-medium">{row.clip}</span>
              <span class="text-muted-foreground flex-1 font-mono">
                {row.parts.map((p) => `1/${K + p.rank}`).join(' + ')}
              </span>
              <span class="text-foreground font-mono">≈ {row.score.toFixed(4)}</span>
              {#if row.inBoth}
                <span class="bg-highlight/30 text-foreground rounded px-1.5 py-0.5 text-[10px] font-medium">in both → wins</span>
              {:else}
                <span class="text-muted-foreground text-[10px]">one list</span>
              {/if}
            </div>
          {/each}
        </div>
        <p class="text-muted-foreground mt-3 text-sm leading-relaxed">
          Clips that appear in <strong class="text-foreground">more lists</strong> rise to the top. RRF needs
          no tuning and works the same for <strong class="text-foreground">2, 3, or 4 lists</strong> — you just
          add another <code class="font-mono">1/(60 + rank)</code> term. The multi-judge
          <code class="font-mono">all</code> mode <em>always</em> uses equal-weight RRF.
        </p>
      </section>

      <!-- 8 · slider vs RRF -->
      <section use:reveal class="reveal mb-10">
        <h2 class="text-foreground mb-1 text-lg font-semibold">Balance slider vs RRF — the key difference</h2>
        <p class="text-muted-foreground mb-3 text-sm leading-relaxed">
          The Balance slider is a <strong class="text-foreground">2-way blend</strong> of the actual scores; it
          only exists for Hybrid (keyword ↔ meaning). It cannot describe 3 legs, so the moment you add an image,
          fusion falls back to equal-weight RRF and the slider is <strong class="text-foreground">ignored</strong>.
        </p>
        <div class="grid gap-3 sm:grid-cols-2">
          <div use:reveal class="reveal bg-surface2 border-border rounded-lg border p-3 text-xs">
            <div class="text-foreground font-medium">Slider (2 judges only)</div>
            <div class="text-muted-foreground mt-0.5">Hybrid keyword ↔ meaning.</div>
            <div class="text-foreground mt-2 font-mono">w·vectorScore + (1−w)·ftsScore</div>
            <div class="text-muted-foreground mt-0.5">Uses real scores. You pick the weight.</div>
          </div>
          <div use:reveal={{ delay: 90 }} class="reveal bg-surface2 border-border rounded-lg border p-3 text-xs">
            <div class="text-foreground font-medium">RRF (2, 3 or 4 judges)</div>
            <div class="text-muted-foreground mt-0.5">Default everywhere; always for multi-leg "all".</div>
            <div class="text-foreground mt-2 font-mono">Σ 1/(60 + rank)</div>
            <div class="text-muted-foreground mt-0.5">Uses ranks only. Equal weight, no tuning.</div>
          </div>
        </div>
      </section>

      <!-- 9 · rerank -->
      <section use:reveal class="reveal bg-card border-border mb-10 rounded-lg border p-4">
        <h2 class="text-foreground mb-1 text-lg font-semibold">Rerank — what it does, and which column it reads</h2>
        <p class="text-muted-foreground mb-3 text-sm leading-relaxed">
          A second, slower-but-sharper precision pass. A cross-encoder reads your query text and one chunk's
          transcript <em>together</em> and scores how well they actually answer each other — better than the
          first-stage ranking, so it only runs on the head.
        </p>
        <div class="border-border overflow-hidden rounded-lg border text-xs">
          <div class="text-muted-foreground grid grid-cols-[7.5rem_1fr]">
            <div class="border-border text-foreground border-b p-2 font-medium">Reads</div>
            <div class="border-border border-b border-l p-2">
              one column — each candidate's transcript, the <code class="bg-surface2 text-primary rounded px-1 font-mono">text</code> field on the chunk. Nothing else.
            </div>
          </div>
          <div class="text-muted-foreground grid grid-cols-[7.5rem_1fr]">
            <div class="border-border text-foreground border-b p-2 font-medium">Scores against</div>
            <div class="border-border border-b border-l p-2">
              your <code class="font-mono">Keyword + Meaning</code> text joined (<code class="font-mono">q + q_vec</code>).
            </div>
          </div>
          <div class="text-muted-foreground grid grid-cols-[7.5rem_1fr]">
            <div class="border-border text-foreground border-b p-2 font-medium">Scope</div>
            <div class="border-border border-b border-l p-2">
              the top <strong class="text-foreground">K</strong> (default 20) of the fused list; it reorders just
              that head. Everything below K keeps its first-stage order — no new search runs.
            </div>
          </div>
          <div class="text-muted-foreground grid grid-cols-[7.5rem_1fr]">
            <div class="text-foreground p-2 font-medium">Ignores</div>
            <div class="border-border border-l p-2">
              the image, the frame <code class="font-mono">caption</code>, and <strong class="text-foreground">all</strong>
              vectors. So an image-only search (no query text) → rerank is a no-op.
            </div>
          </div>
        </div>
      </section>

      <!-- 10 · the one real prefilter: Filters -->
      <section use:reveal class="reveal border-border bg-surface2 mb-10 rounded-lg border p-4">
        <h2 class="text-foreground mb-2 text-lg font-semibold">The one real "narrow-first": Filters</h2>
        <p class="text-muted-foreground text-sm leading-relaxed">
          The <strong class="text-foreground">Filters</strong> panel (language, name, referenskod, extraid, raw
          SQL) is the only thing that narrows the library before searching — and it has nothing to do with the
          image. A filter becomes a <code class="font-mono">WHERE</code> clause applied to <strong class="text-foreground">every
          pass equally</strong>: with <em>prefilter</em> on it narrows the corpus before each pass searches;
          with it off it's applied after. So <strong class="text-foreground">Filter = a fence around the whole
          library</strong>; an attached image = just one more voice in the merge.
        </p>
      </section>

      <!-- 11 · settings reference -->
      <section use:reveal class="reveal mb-10">
        <h2 class="text-foreground mb-3 text-lg font-semibold">⚙️ Settings reference</h2>
        <div class="border-border overflow-hidden rounded-lg border text-xs">
          {#each SETTINGS as s, i (s.name)}
            <div
              use:reveal={{ delay: i * 70 }}
              class="reveal text-muted-foreground grid grid-cols-[1fr_1.7fr] last:[&>div]:border-b-0"
            >
              <div class="border-border text-foreground border-b p-2">{s.name}</div>
              <div class="border-border border-b border-l p-2">{s.desc}</div>
            </div>
          {/each}
        </div>
      </section>

      <!-- 12 · limitations -->
      <section use:reveal class="reveal bg-surface2 border-border mb-8 rounded-lg border p-4">
        <h2 class="text-foreground mb-2 text-lg font-semibold">Known limitations</h2>
        <ul class="text-muted-foreground list-disc space-y-1 pl-5 text-sm leading-relaxed">
          {#each LIMITS as l (l)}<li>{l}</li>{/each}
        </ul>
      </section>

      <a
        use:reveal
        href="/"
        class="reveal border-primary/40 bg-primary/10 text-foreground hover:bg-primary/15 group flex items-center gap-2 rounded-lg border px-4 py-3 text-sm font-medium transition-colors"
      >
        Try it on the Search page
        <ArrowRight class="size-4 transition-transform group-hover:translate-x-0.5" />
      </a>
    {:else}
      <!-- ░░ ATLAS ░░ -->
      <p use:reveal class="reveal text-muted-foreground mb-6 text-sm leading-relaxed">
        The Atlas is a <strong class="text-foreground">2-D map of every chunk</strong>, laid out by the
        <em>meaning of its transcript text</em> — an EVōC projection of the same text embeddings the
        Meaning judge uses. It's a bird's-eye view of the whole corpus.
      </p>

      <div class="grid gap-3 sm:grid-cols-2">
        <div use:reveal class="reveal bg-surface2 border-border border-l-[3px] border-l-primary rounded-lg border p-3">
          <div class="text-foreground font-medium">What it shows</div>
          <p class="text-muted-foreground mt-1 text-xs leading-relaxed">
            Each dot is one chunk. Chunks with similar wording sit close together; colour encodes the
            EVōC cluster they fall into. Dense regions are recurring themes across press conferences.
          </p>
        </div>
        <div use:reveal={{ delay: 90 }} class="reveal bg-surface2 border-border border-l-[3px] border-l-emerald-400 rounded-lg border p-3">
          <div class="text-foreground font-medium">How to read it</div>
          <p class="text-muted-foreground mt-1 text-xs leading-relaxed">
            Distance ≈ semantic similarity. Two clips far apart talk about different things even if they
            share a word; two clips close together are about the same thing even in different words.
          </p>
        </div>
        <div use:reveal={{ delay: 180 }} class="reveal bg-surface2 border-border border-l-[3px] border-l-sky-400 rounded-lg border p-3">
          <div class="text-foreground font-medium">How to use it</div>
          <p class="text-muted-foreground mt-1 text-xs leading-relaxed">
            Pan and zoom, hover a point for its transcript, or lasso a region to cross-filter the linked
            charts and table. Great for discovering clusters you'd never have searched for by keyword.
          </p>
        </div>
        <div use:reveal={{ delay: 270 }} class="reveal bg-surface2 border-border border-l-[3px] border-l-amber-400 rounded-lg border p-3">
          <div class="text-foreground font-medium">Build / rebuild it</div>
          <p class="text-muted-foreground mt-1 text-xs leading-relaxed">
            The projection is computed offline. Run
            <code class="bg-card rounded px-1 py-0.5 font-mono text-[11px]">raudio feature atlas</code>
            to (re)generate it; the tab shows a prompt until it exists.
          </p>
        </div>
      </div>

      <div
        use:reveal
        class="reveal border-border bg-card text-muted-foreground mt-4 rounded-lg border p-3 text-sm leading-relaxed"
      >
        The Atlas maps <strong class="text-foreground">transcript meaning only</strong> — not the video
        frames or captions. It complements Search: Search answers "where is X?", the Atlas answers "what's
        in here, and how does it group?"
      </div>

      <a
        use:reveal
        href="/atlas"
        class="reveal border-primary/40 bg-primary/10 text-foreground hover:bg-primary/15 group mt-6 flex items-center gap-2 rounded-lg border px-4 py-3 text-sm font-medium transition-colors"
      >
        Open the Atlas
        <ArrowRight class="size-4 transition-transform group-hover:translate-x-0.5" />
      </a>
    {/if}
  </div>
</div>

<style>
  /* scroll-reveal: hidden base in scoped CSS; `.in-view` added by JS.
   * `:global(.in-view)` stops Svelte pruning the compile-time-unused class. */
  .reveal {
    opacity: 0;
    transform: translateY(14px);
    transition:
      opacity 0.55s ease,
      transform 0.55s ease;
  }
  .reveal:global(.in-view) {
    opacity: 1;
    transform: none;
  }

  /* lane "scan" bar — all lanes share one timing (no stagger) so the sweep
   * reads as "all four searches run at the same time". */
  .lane-card {
    color: transparent; /* the scan bar inherits via its own bg class, not currentColor */
  }
  .lane-scan {
    position: absolute;
    bottom: 0;
    left: 0;
    height: 2px;
    width: 100%;
    transform: scaleX(0);
    transform-origin: left;
    opacity: 0.7;
    animation: scan 2.2s ease-in-out infinite;
  }
  @keyframes scan {
    0% {
      transform: scaleX(0);
    }
    50% {
      transform: scaleX(1);
    }
    100% {
      transform: scaleX(0);
      transform-origin: right;
    }
  }

  /* vertical connector with a dot flowing down = "data moving through" */
  .flow-line {
    position: relative;
    width: 2px;
    height: 30px;
    margin: 4px auto;
    background: var(--color-border);
  }
  .flow-dot {
    position: absolute;
    left: 50%;
    width: 7px;
    height: 7px;
    border-radius: 9999px;
    background: var(--color-primary);
    box-shadow: 0 0 8px 1px var(--color-primary);
    translate: -50% 0;
    animation: flow 1.8s ease-in-out infinite;
  }
  @keyframes flow {
    0% {
      top: -4px;
      opacity: 0;
    }
    20% {
      opacity: 1;
    }
    80% {
      opacity: 1;
    }
    100% {
      top: 28px;
      opacity: 0;
    }
  }
  @media (prefers-reduced-motion: reduce) {
    .reveal {
      opacity: 1;
      transform: none;
      transition: none;
    }
    .flow-dot,
    .lane-scan {
      display: none;
    }
  }
</style>
