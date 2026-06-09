/**
 * The graph execution engine — a parallel dataflow over the node graph, kept
 * separate from the store so the algorithm is decoupled from (and testable
 * without) WorkflowGraph's internals. It reads topology + config and writes
 * results back through the narrow `RunDeps` seam; it never touches the class.
 */
import type { Edge, Node } from '@xyflow/svelte';
import { search, type Hit, type SearchSpec } from '$lib/api';
import { hitKey } from '$lib/utils';
import { chunkScopeClause, dedupeHits, videoScopeClause } from './scope';
import {
  RERANK_TOP_N,
  type NodeConfig,
  type NodeKind,
  type NodeOutput,
  type NodeRuntime,
} from './types';

/** Everything the executor needs from the graph — a narrow interface so the
 *  store stays in control of state while the algorithm lives here. */
export interface RunDeps {
  nodes: Node[];
  edges: Edge[];
  /** Resolved config for a node (the store fills in defaults for unknowns). */
  config(id: string): NodeConfig;
  kindOf(id: string): NodeKind | null;
  patchRuntime(id: string, patch: Partial<NodeRuntime>): void;
  /** Stamp a Tagger node's tags onto the passing hits (shared tag store). */
  tagHits(hits: Hit[], tags: string[]): void;
}

/**
 * Execute the graph as a dataflow: every node runs as soon as its OWN
 * predecessors resolve, so independent branches (e.g. two Search legs feeding a
 * Combine) run concurrently instead of strictly one-after-another. Promises are
 * seeded in topological order, so when a node reads its predecessors' promises
 * they already exist. Each node is isolated — one failure marks only that node
 * and dependents still run with whatever survived. Returns a cycle-error message
 * to surface, or null on success.
 */
export async function runGraph(deps: RunDeps): Promise<string | null> {
  const ids = deps.nodes.map((n) => n.id);
  const incoming = new Map<string, string[]>(ids.map((id) => [id, []]));
  for (const e of deps.edges) {
    if (incoming.has(e.target) && incoming.has(e.source)) incoming.get(e.target)!.push(e.source);
  }

  const order = topoOrder(deps, incoming);
  if (!order) return 'The graph has a cycle — remove a connection and run again.';

  const outputs = new Map<string, Promise<NodeOutput>>();
  for (const id of order) {
    const preds = incoming.get(id) ?? [];
    // `order` is topological, so every predecessor's promise is already set.
    outputs.set(
      id,
      Promise.all(preds.map((p) => outputs.get(p)!)).then((predOutputs) =>
        runNode(deps, id, predOutputs),
      ),
    );
  }
  await Promise.all(outputs.values());
  return null;
}

/** Topologically order the node ids; returns null if the graph has a cycle. */
function topoOrder(deps: RunDeps, incoming: Map<string, string[]>): string[] | null {
  const ids = deps.nodes.map((n) => n.id);
  const deg = new Map<string, number>(ids.map((id) => [id, incoming.get(id)?.length ?? 0]));
  const queue = ids.filter((id) => (deg.get(id) ?? 0) === 0);
  const order: string[] = [];
  while (queue.length) {
    const id = queue.shift()!;
    order.push(id);
    for (const e of deps.edges) {
      if (e.source !== id) continue;
      const d = (deg.get(e.target) ?? 0) - 1;
      deg.set(e.target, d);
      if (d === 0) queue.push(e.target);
    }
  }
  return order.length === ids.length ? order : null;
}

/** Run one node: merge its predecessors' outputs into its input, then execute by
 *  kind. Returns the output that travels to successors. Isolated — any throw is
 *  recorded on the node and surfaces as an empty output, so it never rejects a
 *  dependent's `Promise.all`. */
async function runNode(deps: RunDeps, id: string, predOutputs: NodeOutput[]): Promise<NodeOutput> {
  const kind = deps.kindOf(id);
  const cfg = deps.config(id);

  // Merge upstream outputs. `inSpec` = WHAT to search; `scope` = the union of
  // upstream result sets (WHERE). Track per-source hit sets (Combine·intersect)
  // and same-field collisions (honesty badges).
  const inSpec: Partial<SearchSpec> = {};
  const scopeHits: Hit[] = [];
  const sourceHitSets: Hit[][] = [];
  let qContrib = 0;
  let imgContrib = 0;
  for (const o of predOutputs) {
    if (o.spec.q) qContrib += 1;
    if (o.spec.image) imgContrib += 1;
    Object.assign(inSpec, o.spec);
    if (o.hits && o.hits.length) {
      scopeHits.push(...o.hits);
      sourceHitSets.push(o.hits);
    }
  }
  const scope: Hit[] | null = scopeHits.length ? dedupeHits(scopeHits) : null;

  // Disabled node: bypass it — forward the scope, contribute nothing.
  if (!cfg.enabled) {
    deps.patchRuntime(id, { status: 'idle', hits: scope, count: scope?.length ?? null });
    return { spec: {}, hits: scope };
  }

  try {
    switch (kind) {
      case 'query': {
        const q = cfg.q.trim();
        deps.patchRuntime(id, { status: q ? 'done' : 'idle' });
        return { spec: q ? { q } : {}, hits: null };
      }
      case 'image': {
        deps.patchRuntime(id, { status: cfg.image ? 'done' : 'idle' });
        return { spec: cfg.image ? { image: cfg.image } : {}, hits: null };
      }
      case 'filter': {
        const spec: Partial<SearchSpec> = {};
        if (cfg.where.trim()) spec.where = cfg.where.trim();
        if (cfg.language.trim()) spec.language = cfg.language.trim();
        if (cfg.namn.trim()) spec.namn = cfg.namn.trim();
        deps.patchRuntime(id, { status: Object.keys(spec).length ? 'done' : 'idle' });
        return { spec, hits: null };
      }
      case 'combine': {
        let combined: Hit[] = [];
        if (sourceHitSets.length) {
          if (cfg.combineMode === 'intersect') {
            const keySets = sourceHitSets.map((s) => new Set(s.map(hitKey)));
            combined = dedupeHits(
              sourceHitSets[0]!.filter((h) => keySets.every((ks) => ks.has(hitKey(h)))),
            );
          } else {
            combined = scope ?? [];
          }
        }
        deps.patchRuntime(id, {
          status: sourceHitSets.length ? 'done' : 'idle',
          hits: combined,
          count: combined.length,
        });
        return { spec: {}, hits: combined.length ? combined : null };
      }
      case 'tagger': {
        // Stamp this node's tags onto every passing chunk (shared store), then
        // forward them unchanged. Inline tags on the same chunks survive too.
        if (scope) deps.tagHits(scope, cfg.tags);
        deps.patchRuntime(id, {
          status: scope ? 'done' : 'idle',
          hits: scope,
          count: scope?.length ?? null,
        });
        return { spec: {}, hits: scope };
      }
      case 'search': {
        // Query is a connected Query node if wired, else this node's inline field.
        const q = inSpec.q?.trim() || cfg.q.trim();
        const image = inSpec.image ?? null;
        // Dropped wired inputs: extra duplicate upstreams, plus the inline query
        // when an upstream query also supplied one.
        const inlineQDropped = cfg.q.trim() && qContrib > 0 ? 1 : 0;
        const droppedInputs =
          Math.max(0, qContrib - 1) + Math.max(0, imgContrib - 1) + inlineQDropped;
        if (!q && !image) {
          // Nothing to search for — pass the scope through so a half-configured
          // node never breaks the chain.
          deps.patchRuntime(id, {
            status: 'idle',
            hits: scope,
            count: scope?.length ?? null,
            droppedInputs,
          });
          return { spec: {}, hits: scope };
        }
        deps.patchRuntime(id, { status: 'running' });

        const spec: SearchSpec = { q, n: cfg.n, mode: cfg.mode };
        if (cfg.rerank) {
          spec.rerank = true;
          spec.rerankN = RERANK_TOP_N;
        }
        if (image) spec.image = image;
        if (inSpec.language) spec.language = inSpec.language;
        if (inSpec.namn) spec.namn = inSpec.namn;

        // WHERE: any upstream filter, ANDed with the refinement scope — either
        // the upstream videos (`doc_id IN`) or the exact upstream chunks.
        const wheres: string[] = [];
        if (inSpec.where) wheres.push(inSpec.where);
        let scopedDocs: number | null = null;
        let scopedChunks: number | null = null;
        let scopeCapped = false;
        if (scope?.length) {
          const sc =
            cfg.refineScope === 'chunk' ? chunkScopeClause(scope) : videoScopeClause(scope);
          if (sc) {
            wheres.push(sc.clause);
            scopeCapped = sc.capped;
            if (cfg.refineScope === 'chunk') scopedChunks = sc.count;
            else scopedDocs = sc.count;
          }
        }
        if (wheres.length) spec.where = wheres.join(' AND ');

        const t0 = performance.now();
        const hits = await search(spec);
        const ms = Math.round(performance.now() - t0);
        deps.patchRuntime(id, {
          status: 'done',
          hits,
          count: hits.length,
          ms,
          scopedDocs,
          scopedChunks,
          scopeCapped,
          droppedInputs,
        });
        return { spec: {}, hits };
      }
      // Sinks: collect the incoming hits and surface them (Results renders them;
      // Export downloads them). Neither contributes a spec.
      case 'results':
      case 'export': {
        deps.patchRuntime(id, {
          status: scope ? 'done' : 'idle',
          hits: scope,
          count: scope?.length ?? null,
        });
        return { spec: {}, hits: scope };
      }
      default:
        return { spec: {}, hits: null };
    }
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    deps.patchRuntime(id, { status: 'error', error: msg });
    return { spec: {}, hits: null };
  }
}
