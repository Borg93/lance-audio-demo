<script lang="ts">
  /**
   * Zoomable topic treemap (Tree page).
   *
   * Renders the nested topic hierarchy (`getTopics().hierarchy`) one level at a
   * time: the LayerChart <Treemap> computes the rectangular layout, we draw the
   * current root's direct children as SVG cells. Clicking a branch zooms in
   * (drill-down); a breadcrumb zooms back out. "Show results" hands a topic off
   * to Search via `/?topic=<name>` (matched against every topic_l* layer).
   *
   * The `(övrigt)` bucket is HDBSCAN noise (stored as NULL, so it matches no
   * topic_l* value) — shown muted and non-interactive.
   */
  import { Chart, Svg } from 'layerchart';
  import { Treemap } from 'layerchart/hierarchy';
  import { hierarchy as d3hierarchy, type HierarchyRectangularNode } from 'd3-hierarchy';
  import { scaleOrdinal } from 'd3-scale';
  import { schemeTableau10 } from 'd3-scale-chromatic';
  import { goto } from '$app/navigation';
  import { ChevronRight, ArrowRight } from 'lucide-svelte';
  import type { TopicNode } from '$lib/api';

  // `noiseLabel` is the unclustered-bucket name, sourced from /api/topics
  // (topic_tree.py:NOISE_LABEL) — never hardcoded here, so the two can't drift.
  let { hierarchy, noiseLabel }: { hierarchy: TopicNode; noiseLabel: string } = $props();

  const fmt = (n: number) => n.toLocaleString('sv-SE');

  // Drill stack: branches descended into below the root. The breadcrumb is the
  // root prop + this stack (kept separate so referencing the prop stays
  // reactive); `current` is the deepest node shown.
  let drill = $state<TopicNode[]>([]);
  const path = $derived<TopicNode[]>([hierarchy, ...drill]);
  const current = $derived<TopicNode>(drill.at(-1) ?? hierarchy);

  // d3 hierarchy for the current node: branches contribute 0 (leaves carry the
  // chunk count), so the treemap areas sum cleanly. Largest topics first.
  const root = $derived(
    d3hierarchy<TopicNode>(current)
      .sum((d) => (d.children?.length ? 0 : (d.value ?? 0)))
      .sort((a, b) => (b.value ?? 0) - (a.value ?? 0)),
  );

  const color = scaleOrdinal<string, string>(schemeTableau10);
  const fillFor = (node: TopicNode) =>
    node.name === noiseLabel ? 'var(--color-muted)' : color(node.name);

  const isInteractive = (node: TopicNode) => node.name !== noiseLabel;

  let hovered = $state<string | null>(null);

  function onCell(node: TopicNode) {
    if (!isInteractive(node)) return;
    if (node.children?.length) drill = [...drill, node];
    else showResults(node.name);
  }

  function showResults(name: string) {
    goto(`/?topic=${encodeURIComponent(name)}`);
  }

  /** Zoom out to breadcrumb position `index` (0 = root). */
  function crumbTo(index: number) {
    drill = drill.slice(0, index);
  }

  /** Trim a label to roughly fit a cell's width (no SVG text clipping). */
  function fit(name: string, width: number): string {
    const max = Math.floor((width - 12) / 7.2);
    if (max < 2) return '';
    return name.length > max ? `${name.slice(0, max - 1)}…` : name;
  }
</script>

<div class="flex h-full w-full flex-col">
  <!-- Breadcrumb + "show results for current branch" -->
  <div class="flex items-center gap-1 border-b border-border bg-card/30 px-4 py-2 text-xs">
    {#each path as node, i (i)}
      {#if i > 0}<ChevronRight class="size-3 text-muted-foreground/50" />{/if}
      <button
        type="button"
        class="rounded px-1.5 py-0.5 font-medium transition-colors hover:bg-secondary disabled:cursor-default disabled:opacity-100"
        class:text-foreground={i === path.length - 1}
        class:text-muted-foreground={i !== path.length - 1}
        disabled={i === path.length - 1}
        onclick={() => crumbTo(i)}
      >
        {node.name}
      </button>
    {/each}
    <span class="ml-1 text-muted-foreground/70">· {fmt(root.value ?? 0)} segment</span>

    {#if path.length > 1 && isInteractive(current)}
      <button
        type="button"
        class="ml-auto inline-flex items-center gap-1 rounded-md border border-border bg-secondary px-2 py-0.5 font-medium text-foreground transition-colors hover:bg-secondary/70"
        onclick={() => showResults(current.name)}
      >
        Visa resultat <ArrowRight class="size-3" />
      </button>
    {/if}
  </div>

  <!-- Treemap canvas -->
  <div class="min-h-0 flex-1 p-2">
    <Chart data={root}>
      <Svg>
        <Treemap hierarchy={root} paddingInner={3}>
          {#snippet children({ nodes }: { nodes: HierarchyRectangularNode<TopicNode>[] })}
            {#each nodes.filter((n) => n.depth === 1) as node (node.data.name)}
              {@const w = node.x1 - node.x0}
              {@const h = node.y1 - node.y0}
              {@const interactive = isInteractive(node.data)}
              {@const label = fit(node.data.name, w)}
              <!-- svelte-ignore a11y_no_noninteractive_tabindex -->
              <g
                role={interactive ? 'button' : undefined}
                tabindex={interactive ? 0 : undefined}
                class={interactive ? 'cursor-pointer' : 'cursor-not-allowed'}
                onclick={() => onCell(node.data)}
                onkeydown={(e) => {
                  if (interactive && (e.key === 'Enter' || e.key === ' ')) {
                    e.preventDefault();
                    onCell(node.data);
                  }
                }}
                onmouseenter={() => (hovered = node.data.name)}
                onmouseleave={() => (hovered = null)}
              >
                <title>
                  {node.data.name} — {fmt(node.value ?? 0)} segment{node.data.children?.length
                    ? ' (klicka för att zooma in)'
                    : interactive
                      ? ' (klicka för att visa resultat)'
                      : ' (ej klustrade)'}
                </title>
                <rect
                  x={node.x0}
                  y={node.y0}
                  width={w}
                  height={h}
                  rx={3}
                  fill={fillFor(node.data)}
                  opacity={interactive ? (hovered === node.data.name ? 1 : 0.85) : 0.35}
                  stroke="var(--color-background)"
                  stroke-width={hovered === node.data.name ? 2 : 1}
                />
                {#if w > 46 && h > 22 && label}
                  <text
                    x={node.x0 + 7}
                    y={node.y0 + 17}
                    class="pointer-events-none fill-white text-[12px] font-semibold"
                    style="paint-order: stroke; stroke: rgba(0,0,0,0.35); stroke-width: 2.5px;"
                  >
                    {label}
                  </text>
                  {#if h > 38}
                    <text
                      x={node.x0 + 7}
                      y={node.y0 + 33}
                      class="pointer-events-none fill-white/85 text-[11px] tabular-nums"
                      style="paint-order: stroke; stroke: rgba(0,0,0,0.3); stroke-width: 2px;"
                    >
                      {fmt(node.value ?? 0)}
                    </text>
                  {/if}
                {/if}
              </g>
            {/each}
          {/snippet}
        </Treemap>
      </Svg>
    </Chart>
  </div>
</div>
