<script lang="ts">
  /**
   * Sankey ("Flow") view of the topic hierarchy — the third Tree-page view next
   * to the flat/nested treemaps. Every hierarchy level is a column; ribbon
   * width = the sub-topic's chunk count, so how a broad topic splits into
   * sub-topics is readable at a glance. Clicking any (non-noise) topic opens
   * its results, same as a treemap leaf.
   *
   * LayerChart's <Sankey> (d3-sankey) computes the layout from the same
   * d3-hierarchy the treemaps use; nodes ARE the hierarchy nodes, so depth /
   * ancestors drive colour exactly like the treemap.
   */
  import { Chart, Svg, Link } from 'layerchart';
  import { Sankey } from 'layerchart/graph';
  import { hierarchy as d3hierarchy, type HierarchyNode } from 'd3-hierarchy';
  import { scaleOrdinal } from 'd3-scale';
  import { schemeTableau10 } from 'd3-scale-chromatic';
  import type { TopicNode } from '$lib/api';

  let {
    data,
    noiseLabel,
    onSelect,
  }: {
    /** The (already noise-pruned, drill-scoped) hierarchy to lay out. */
    data: TopicNode;
    noiseLabel: string;
    /** Open a topic's results (any non-noise node). */
    onSelect: (name: string) => void;
  } = $props();

  /** A hierarchy node after d3-sankey has stamped its layout onto it. */
  type FlowNode = HierarchyNode<TopicNode> & {
    x0: number;
    x1: number;
    y0: number;
    y1: number;
  };
  type FlowLink = { source: FlowNode; target: FlowNode; value: number; width?: number };

  const fmt = (n: number) => n.toLocaleString('sv-SE');
  const sumLeaves = (d: TopicNode) => (d.children?.length ? 0 : (d.value ?? 0));

  // Fresh hierarchy per data change (d3-sankey mutates the nodes in place).
  // Same sum + largest-first sort as the treemap, so the ordinal colour scale
  // sees topics in the same order and the two views share hues.
  const graph = $derived.by(() => {
    const root = d3hierarchy<TopicNode>(data)
      .sum(sumLeaves)
      .sort((a, b) => (b.value ?? 0) - (a.value ?? 0));
    return {
      nodes: root.descendants(),
      links: root.links().map((l) => ({
        source: l.source,
        target: l.target,
        value: l.target.value ?? 0,
      })),
    };
  });

  const color = scaleOrdinal<string, string>(schemeTableau10);
  /** Hue = the broadest (depth-1) ancestor, mirroring the treemap. */
  function nodeFill(node: FlowNode): string {
    if (node.data.name === noiseLabel) return 'var(--color-muted)';
    const top = node.ancestors().find((n) => n.depth === 1);
    return color(top?.data.name ?? node.data.name);
  }

  const isInteractive = (node: FlowNode) => node.depth > 0 && node.data.name !== noiseLabel;
  let hovered = $state<string | null>(null);

  /** Vertical room per leaf row. d3-sankey does NOT shrink its padding to fit:
   *  once a column's padding alone exceeds the canvas height, the scale goes
   *  negative and every node collapses to 0px. So the canvas grows with the
   *  leaf count and the wrapper scrolls instead. */
  const ROW_PX = 9;
  const MIN_HEIGHT_PX = 560;
  const canvasHeight = $derived.by(() => {
    let leaves = 0;
    (function count(n: TopicNode): void {
      if (n.children?.length) n.children.forEach(count);
      else leaves += 1;
    })(data);
    return Math.max(MIN_HEIGHT_PX, leaves * ROW_PX);
  });
</script>

<div class="h-full overflow-y-auto">
  <div style="height: {canvasHeight}px">
    <Chart data={graph}>
      <Svg>
        <Sankey nodeWidth={10} nodePadding={6} nodeAlign="justify">
          {#snippet children({ nodes, links }: { nodes: FlowNode[]; links: FlowLink[] })}
            {#each links as link (`${link.source.data.name}›${link.target.data.name}`)}
              <Link
                sankey
                data={link}
                fill="none"
                stroke={nodeFill(link.target)}
                stroke-opacity={hovered === link.target.data.name ? 0.65 : 0.3}
                stroke-width={Math.max(1, link.width ?? 0)}
              />
            {/each}
            {#each nodes as node (node
              .ancestors()
              .map((a) => a.data.name)
              .join('›'))}
              {@const h = node.y1 - node.y0}
              {@const interactive = isInteractive(node)}
              {@const hot = hovered === node.data.name}
              <!-- svelte-ignore a11y_no_noninteractive_tabindex -->
              <g
                role={interactive ? 'button' : undefined}
                tabindex={interactive ? 0 : undefined}
                class={interactive ? 'cursor-pointer' : undefined}
                onclick={() => interactive && onSelect(node.data.name)}
                onkeydown={(e) => {
                  if (interactive && (e.key === 'Enter' || e.key === ' ')) {
                    e.preventDefault();
                    onSelect(node.data.name);
                  }
                }}
                onmouseenter={() => (hovered = node.data.name)}
                onmouseleave={() => (hovered = null)}
              >
                <title
                  >{node.data.name} — {fmt(node.value ?? 0)} chunks{interactive
                    ? ' (click to show results)'
                    : ''}</title
                >
                <rect
                  x={node.x0}
                  y={node.y0}
                  width={node.x1 - node.x0}
                  height={Math.max(1, h)}
                  rx={2}
                  fill={nodeFill(node)}
                  opacity={hot ? 1 : 0.9}
                />
                {#if h > 11}
                  <!-- Leaves sit on the right edge (justify): label to the left. -->
                  <text
                    x={node.children?.length ? node.x1 + 5 : node.x0 - 5}
                    y={(node.y0 + node.y1) / 2}
                    dy="0.32em"
                    text-anchor={node.children?.length ? 'start' : 'end'}
                    class="pointer-events-none fill-foreground text-[10px]"
                  >
                    {node.data.name}
                    <tspan class="fill-muted-foreground tabular-nums">
                      {fmt(node.value ?? 0)}</tspan
                    >
                  </text>
                {/if}
              </g>
            {/each}
          {/snippet}
        </Sankey>
      </Svg>
    </Chart>
  </div>
</div>
