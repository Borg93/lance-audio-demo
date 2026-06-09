/**
 * Auto-layout the node graph left-to-right with dagre — drives the toolbar's
 * "Tidy" button. Pure: takes nodes/edges, returns nodes with new positions.
 */
import dagre from '@dagrejs/dagre';
import type { Edge, Node } from '@xyflow/svelte';

/** Fallback box when a node hasn't been measured yet (pre-first-render). */
const DEFAULT_W = 256;
const DEFAULT_H = 150;
/** dagre spacing (px): gap between siblings, gap between ranks, outer margin. */
const NODE_SEP = 40;
const RANK_SEP = 90;
const MARGIN = 20;

/** Re-position `nodes` with dagre. `direction` follows dagre rankdir
 *  ('LR' = left→right, the natural reading order for these pipelines). */
export function autoLayout(nodes: Node[], edges: Edge[], direction: 'LR' | 'TB' = 'LR'): Node[] {
  if (!nodes.length) return nodes;

  const g = new dagre.graphlib.Graph();
  g.setDefaultEdgeLabel(() => ({}));
  g.setGraph({
    rankdir: direction,
    nodesep: NODE_SEP,
    ranksep: RANK_SEP,
    marginx: MARGIN,
    marginy: MARGIN,
  });

  const size = (n: Node): { width: number; height: number } => ({
    width: n.measured?.width ?? n.width ?? DEFAULT_W,
    height: n.measured?.height ?? n.height ?? DEFAULT_H,
  });

  for (const n of nodes) g.setNode(n.id, size(n));
  for (const e of edges) {
    if (g.hasNode(e.source) && g.hasNode(e.target)) g.setEdge(e.source, e.target);
  }

  dagre.layout(g);

  // dagre reports node centres; Svelte Flow positions are top-left.
  return nodes.map((n) => {
    const p = g.node(n.id);
    if (!p) return n;
    const { width, height } = size(n);
    return { ...n, position: { x: p.x - width / 2, y: p.y - height / 2 } };
  });
}
