/**
 * Workflow graph — the shared state for the node-based pipeline editor.
 *
 * A single runes-class singleton (same convention as
 * `$lib/atlas/cross-filter.svelte.ts`) imported by the canvas AND every custom
 * node component, so they all lock-step on one source of truth.
 *
 * Three pieces of state, kept deliberately separate:
 *   • `nodes` / `edges` — the Svelte Flow graph (`$state.raw`, replaced
 *     wholesale; Svelte Flow binds + mutates these on drag/connect/delete).
 *   • `config`  — per-node USER input (the query text, search mode, filter…),
 *     keyed by node id. Deep `$state` so node forms edit it reactively.
 *   • `runtime` — per-node RUN state (status / hits / error / timing), keyed by
 *     node id. Populated by `run()`, read by the nodes to render themselves.
 *
 * Edges carry STATE forward. Each node produces a `NodeOutput`:
 *   • a partial `SearchSpec` (a query, an image, metadata filters), and/or
 *   • a `Hit[]` result set.
 * `run()` walks the graph in topological order and merges every incoming
 * output into the next node's input. Two things flow down a wire into a Search:
 *   • a query/image/filter  → WHAT to search for, and
 *   • an upstream result set → WHERE to search (the Search scopes itself to the
 *     videos those results came from, via `doc_id IN (…)`).
 * So `Image → Search·Visual → Search·Scene → Search·Keyword → Results` reads as
 * "find clips like this image, then among those find ones whose scene matches,
 * then among those find ones where this is said" — each stage refining the last.
 */

import type { Edge, Node } from '@xyflow/svelte';
import { search, type Hit, type SearchMode, type SearchSpec } from '$lib/api';

/** The five pipeline stages. A node's `type` (used to pick its component) is
 *  its kind. */
export type NodeKind = 'query' | 'image' | 'filter' | 'search' | 'results';

export type RunStatus = 'idle' | 'running' | 'done' | 'error';

/** Cap on the distinct videos a Search is scoped to by upstream results: the
 *  `doc_id IN (…)` clause keeps at most this many (highest-ranked first), so a
 *  large upstream set can't blow up the SQL. */
const MAX_SCOPE_DOCS = 80;

/** Per-node user input. One flat shape covers every kind (each node UI only
 *  edits the fields it cares about) — simpler than a discriminated union for a
 *  POC, and the unused fields stay at their harmless defaults. */
export interface NodeConfig {
    /** query node, and the Search node's own inline query. */
    q: string;
    /** image: the uploaded file fed to the visual leg (POST multipart). */
    image: File | null;
    imageName: string;
    /** filter: a raw SQL WHERE plus the common scalar facets. */
    where: string;
    language: string;
    namn: string;
    /** search: how the merged inputs are ranked. */
    mode: SearchMode;
    n: number;
    rerank: boolean;
}

export interface NodeRuntime {
    status: RunStatus;
    error: string | null;
    /** Result hits (Search and Results nodes). */
    hits: Hit[] | null;
    /** Result count summary (cheap to render without holding the array). */
    count: number | null;
    /** Last run wall-clock in ms (Search node). */
    ms: number | null;
    /** How many videos this Search was scoped to by an upstream stage (null =
     *  searched the whole corpus). */
    scopedDocs: number | null;
}

/** What travels along an edge from a node to its successors. */
interface NodeOutput {
    /** Partial search spec contributed by this node (query text, image, filters). */
    spec: Partial<SearchSpec>;
    /** A concrete result set, once a Search node has produced one. */
    hits: Hit[] | null;
}

export const SEARCH_MODES: { value: SearchMode; label: string }[] = [
    { value: 'fts', label: 'Keyword (FTS)' },
    { value: 'semantic', label: 'Meaning (vector)' },
    { value: 'hybrid', label: 'Hybrid (FTS + vector)' },
    { value: 'visual', label: 'Image (frame vector)' },
    { value: 'scene', label: 'Scene (caption vector)' },
    { value: 'scene_fts', label: 'Scene (caption keyword)' },
    { value: 'all', label: 'All judges (RRF)' },
];

export const modeLabel = (mode: SearchMode): string =>
    SEARCH_MODES.find((m) => m.value === mode)?.label ?? mode;

const KIND_LABEL: Record<NodeKind, string> = {
    query: 'Text query',
    image: 'Image',
    filter: 'Filter',
    search: 'Search',
    results: 'Results',
};

export const nodeLabel = (kind: NodeKind): string => KIND_LABEL[kind];

/** Run-status → Tailwind dot colour. Shared by NodeShell and the Inspector. */
export const STATUS_DOT: Record<RunStatus, string> = {
    idle: 'bg-muted-foreground/40',
    running: 'bg-primary animate-pulse',
    done: 'bg-emerald-500',
    error: 'bg-destructive',
};

/** Identity key for a hit (doc/speech/chunk), used to de-dupe result sets. */
const hitKey = (h: Hit): string => `${h.doc_id}|${h.speech_id}|${h.chunk_id}`;

function defaultConfig(): NodeConfig {
    return {
        q: '',
        image: null,
        imageName: '',
        where: '',
        language: '',
        namn: '',
        mode: 'fts',
        n: 24,
        rerank: false,
    };
}

function blankRuntime(): NodeRuntime {
    return { status: 'idle', error: null, hits: null, count: null, ms: null, scopedDocs: null };
}

/** Escape a value for inlining in a SQL single-quoted string literal. */
function sqlQuote(value: string): string {
    return `'${value.replace(/'/g, "''")}'`;
}

class WorkflowGraph {
    /** The Svelte Flow nodes — bound into <SvelteFlow bind:nodes>. */
    nodes = $state.raw<Node[]>([]);
    /** The Svelte Flow edges — bound into <SvelteFlow bind:edges>. */
    edges = $state.raw<Edge[]>([]);

    /** Per-node user input, keyed by node id. */
    config = $state<Record<string, NodeConfig>>({});
    /** Per-node run state, keyed by node id. */
    runtime = $state<Record<string, NodeRuntime>>({});

    /** True while `run()` is in flight (disables the Run button, shows spinner). */
    running = $state(false);
    /** Last graph-level failure (cycle, etc.); per-node errors live in `runtime`. */
    lastError = $state<string | null>(null);

    /** A result the user clicked — plays in the Inspector. */
    selectedHit = $state<Hit | null>(null);

    /** The node whose intermediate state the Inspector shows (click a node). */
    inspectedNodeId = $state<string | null>(null);

    /** Ids currently selected on the canvas (tracked via `onselectionchange`) —
     *  drives the toolbar's Delete button. */
    selectedNodeIds = $state<string[]>([]);
    selectedEdgeIds = $state<string[]>([]);

    get hasSelection(): boolean {
        return this.selectedNodeIds.length > 0 || this.selectedEdgeIds.length > 0;
    }

    /** Monotonic id source for nodes added at runtime (seeds use bare-kind ids). */
    private seq = 0;

    constructor() {
        this.seed();
    }

    /** Kind of a node by id (its Svelte Flow `type`). */
    kindOf(id: string): NodeKind | null {
        const n = this.nodes.find((x) => x.id === id);
        return n ? (n.type as NodeKind) : null;
    }

    /**
     * The starter graph: a multi-modal refinement chain that demonstrates how
     * to wire "multiples". Upload an image and Run to see all four stages; with
     * no image the visual stage politely skips and the Scene→Keyword chain still
     * runs, so the refinement is visible either way.
     *
     *   Image ─image─▶ Search·Visual ─refine─▶ Search·Scene ─refine─▶ Search·Keyword ─▶ Results
     */
    seed(): void {
        const query = (mode: SearchMode, q: string): NodeConfig => ({ ...defaultConfig(), mode, q });
        this.config = {
            image: defaultConfig(),
            'search-visual': query('visual', ''),
            'search-scene': query('scene', 'talarstol'),
            'search-said': query('fts', 'skatt'),
            results: defaultConfig(),
        };
        this.runtime = Object.fromEntries(
            Object.keys(this.config).map((id) => [id, blankRuntime()]),
        );
        this.nodes = [
            { id: 'image', type: 'image', position: { x: -60, y: 60 }, data: {} },
            { id: 'search-visual', type: 'search', position: { x: 240, y: 40 }, data: {} },
            { id: 'search-scene', type: 'search', position: { x: 560, y: 100 }, data: {} },
            { id: 'search-said', type: 'search', position: { x: 880, y: 160 }, data: {} },
            { id: 'results', type: 'results', position: { x: 1200, y: 100 }, data: {} },
        ];
        this.edges = [
            { id: 'e-img', source: 'image', target: 'search-visual', label: 'image' },
            {
                id: 'e-v-scene',
                source: 'search-visual',
                target: 'search-scene',
                label: 'refine',
                animated: true,
            },
            {
                id: 'e-scene-said',
                source: 'search-scene',
                target: 'search-said',
                label: 'refine',
                animated: true,
            },
            { id: 'e-said-res', source: 'search-said', target: 'results' },
        ];
        this.seq = 0;
        this.running = false;
        this.lastError = null;
        this.selectedHit = null;
        this.inspectedNodeId = null;
        this.selectedNodeIds = [];
        this.selectedEdgeIds = [];
    }

    /** Add a fresh node of `kind` at `position`; returns its id. */
    addNode(kind: NodeKind, position: { x: number; y: number }): string {
        const id = `${kind}-${++this.seq}`;
        this.config = { ...this.config, [id]: defaultConfig() };
        this.runtime = { ...this.runtime, [id]: blankRuntime() };
        this.nodes = [...this.nodes, { id, type: kind, position, data: {} }];
        return id;
    }

    /** Patch one node's user input (reassigns the record so reactivity fires). */
    setConfig(id: string, patch: Partial<NodeConfig>): void {
        const prev = this.config[id] ?? defaultConfig();
        this.config = { ...this.config, [id]: { ...prev, ...patch } };
    }

    /** Patch one node's run state. */
    private patchRuntime(id: string, patch: Partial<NodeRuntime>): void {
        const prev = this.runtime[id] ?? blankRuntime();
        this.runtime = { ...this.runtime, [id]: { ...prev, ...patch } };
    }

    /** Clear all run state back to idle (keeps the graph + user input). */
    resetRun(): void {
        const next: Record<string, NodeRuntime> = {};
        for (const n of this.nodes) next[n.id] = blankRuntime();
        this.runtime = next;
        this.lastError = null;
    }

    /** Reset everything to the seeded starter graph. */
    reset(): void {
        this.seed();
    }

    /** Open the detail/player drawer for a result clicked in a Results node. */
    selectHit(hit: Hit): void {
        this.selectedHit = hit;
    }

    /** Stop playing the selected result (back to the inspected node's results). */
    closeDetail(): void {
        this.selectedHit = null;
    }

    /** Show a node's intermediate state (config + results) in the Inspector.
     *  Clearing `selectedHit` lets a node click switch the panel away from a
     *  playing result. (Result-row clicks `stopPropagation`, so they never
     *  reach here and keep playing.) */
    inspectNode(id: string): void {
        this.selectedHit = null;
        this.inspectedNodeId = id;
    }

    /** Record the canvas selection (from `<SvelteFlow onselectionchange>`). */
    setSelection(nodeIds: string[], edgeIds: string[]): void {
        this.selectedNodeIds = nodeIds;
        this.selectedEdgeIds = edgeIds;
    }

    /** Delete one node, every edge touching it, and its config/runtime. */
    removeNode(id: string): void {
        this.nodes = this.nodes.filter((n) => n.id !== id);
        this.edges = this.edges.filter((e) => e.source !== id && e.target !== id);
        const config = { ...this.config };
        const runtime = { ...this.runtime };
        delete config[id];
        delete runtime[id];
        this.config = config;
        this.runtime = runtime;
        this.selectedNodeIds = this.selectedNodeIds.filter((x) => x !== id);
        const liveEdges = new Set(this.edges.map((e) => e.id));
        this.selectedEdgeIds = this.selectedEdgeIds.filter((x) => liveEdges.has(x));
        if (this.inspectedNodeId === id) this.inspectedNodeId = null;
    }

    /** Delete the current selection: selected nodes (and their edges) + edges. */
    deleteSelected(): void {
        const nodeIds = new Set(this.selectedNodeIds);
        const edgeIds = new Set(this.selectedEdgeIds);
        if (!nodeIds.size && !edgeIds.size) return;
        this.nodes = this.nodes.filter((n) => !nodeIds.has(n.id));
        this.edges = this.edges.filter(
            (e) => !edgeIds.has(e.id) && !nodeIds.has(e.source) && !nodeIds.has(e.target),
        );
        const config = { ...this.config };
        const runtime = { ...this.runtime };
        for (const id of nodeIds) {
            delete config[id];
            delete runtime[id];
        }
        this.config = config;
        this.runtime = runtime;
        if (this.inspectedNodeId && nodeIds.has(this.inspectedNodeId)) this.inspectedNodeId = null;
        this.selectedNodeIds = [];
        this.selectedEdgeIds = [];
    }

    /** Prune state for elements Svelte Flow already removed itself — its
     *  built-in Backspace/Delete mutates the bound `nodes`/`edges` directly, so
     *  it never reaches `removeNode`/`deleteSelected`. Wired via `<SvelteFlow
     *  ondelete>` so all three delete paths converge on the same cleanup. */
    syncDeleted(nodeIds: string[], edgeIds: string[]): void {
        if (!nodeIds.length && !edgeIds.length) return;
        const gone = new Set(nodeIds);
        const config = { ...this.config };
        const runtime = { ...this.runtime };
        for (const id of gone) {
            delete config[id];
            delete runtime[id];
        }
        this.config = config;
        this.runtime = runtime;
        if (this.inspectedNodeId && gone.has(this.inspectedNodeId)) this.inspectedNodeId = null;
        this.selectedNodeIds = this.selectedNodeIds.filter((x) => !gone.has(x));
        const goneEdges = new Set(edgeIds);
        this.selectedEdgeIds = this.selectedEdgeIds.filter((x) => !goneEdges.has(x));
    }

    /** Topologically order the node ids; returns null if the graph has a cycle. */
    private topoOrder(incoming: Map<string, string[]>): string[] | null {
        const ids = this.nodes.map((n) => n.id);
        const deg = new Map<string, number>(ids.map((id) => [id, incoming.get(id)?.length ?? 0]));
        const queue = ids.filter((id) => (deg.get(id) ?? 0) === 0);
        const order: string[] = [];
        while (queue.length) {
            const id = queue.shift()!;
            order.push(id);
            for (const e of this.edges) {
                if (e.source !== id) continue;
                const d = (deg.get(e.target) ?? 0) - 1;
                deg.set(e.target, d);
                if (d === 0) queue.push(e.target);
            }
        }
        return order.length === ids.length ? order : null;
    }

    /**
     * Execute the graph: topological order, merging each node's incoming
     * outputs into its input, running the real `search()` at Search nodes.
     * Each node is isolated — one failure marks only that node and the chain
     * continues with whatever the surviving nodes produced.
     */
    async run(): Promise<void> {
        if (this.running) return;
        this.running = true;
        this.resetRun();

        const ids = this.nodes.map((n) => n.id);
        const incoming = new Map<string, string[]>(ids.map((id) => [id, []]));
        for (const e of this.edges) {
            if (incoming.has(e.target) && incoming.has(e.source)) {
                incoming.get(e.target)!.push(e.source);
            }
        }

        const order = this.topoOrder(incoming);
        if (!order) {
            this.lastError = 'The graph has a cycle — remove a connection and run again.';
            this.running = false;
            return;
        }

        const outputs = new Map<string, NodeOutput>();
        for (const id of order) {
            const kind = this.kindOf(id);
            const cfg = this.config[id] ?? defaultConfig();

            // Merge every upstream output into this node's input. `spec` says
            // WHAT to search for; `hits` is the upstream result set to refine in.
            const inSpec: Partial<SearchSpec> = {};
            // Accumulate every upstream result set (union, de-duped by chunk) so
            // fanning two Searches into one scopes to BOTH, not just the last.
            const scopeHits: Hit[] = [];
            for (const src of incoming.get(id) ?? []) {
                const o = outputs.get(src);
                if (!o) continue;
                Object.assign(inSpec, o.spec);
                if (o.hits) scopeHits.push(...o.hits);
            }
            let scope: Hit[] | null = null;
            if (scopeHits.length) {
                const seen = new Set<string>();
                scope = scopeHits.filter((h) => {
                    const k = hitKey(h);
                    if (seen.has(k)) return false;
                    seen.add(k);
                    return true;
                });
            }

            try {
                switch (kind) {
                    case 'query': {
                        const q = cfg.q.trim();
                        outputs.set(id, { spec: q ? { q } : {}, hits: null });
                        this.patchRuntime(id, { status: q ? 'done' : 'idle' });
                        break;
                    }
                    case 'image': {
                        outputs.set(id, { spec: cfg.image ? { image: cfg.image } : {}, hits: null });
                        this.patchRuntime(id, { status: cfg.image ? 'done' : 'idle' });
                        break;
                    }
                    case 'filter': {
                        const spec: Partial<SearchSpec> = {};
                        if (cfg.where.trim()) spec.where = cfg.where.trim();
                        if (cfg.language.trim()) spec.language = cfg.language.trim();
                        if (cfg.namn.trim()) spec.namn = cfg.namn.trim();
                        outputs.set(id, { spec, hits: null });
                        this.patchRuntime(id, { status: Object.keys(spec).length ? 'done' : 'idle' });
                        break;
                    }
                    case 'search': {
                        // Query is a connected Query node if wired, else this
                        // Search node's own inline field.
                        const q = inSpec.q?.trim() || cfg.q.trim();
                        const image = inSpec.image ?? null;
                        if (!q && !image) {
                            // Nothing to search for — pass the scope through so a
                            // half-configured node never breaks the chain.
                            outputs.set(id, { spec: {}, hits: scope });
                            this.patchRuntime(id, {
                                status: 'idle',
                                hits: scope,
                                count: scope?.length ?? null,
                            });
                            break;
                        }
                        this.patchRuntime(id, { status: 'running' });

                        const spec: SearchSpec = { q, n: cfg.n, mode: cfg.mode };
                        if (cfg.rerank) {
                            spec.rerank = true;
                            spec.rerankN = 20;
                        }
                        if (image) spec.image = image;
                        if (inSpec.language) spec.language = inSpec.language;
                        if (inSpec.namn) spec.namn = inSpec.namn;

                        // Build the WHERE: any upstream filter, ANDed with the
                        // refinement scope (the videos the upstream results came from).
                        const wheres: string[] = [];
                        if (inSpec.where) wheres.push(inSpec.where);
                        let scopedDocs: number | null = null;
                        if (scope?.length) {
                            const docs = [...new Set(scope.map((h) => h.doc_id))].slice(0, MAX_SCOPE_DOCS);
                            if (docs.length) {
                                wheres.push(`doc_id IN (${docs.map(sqlQuote).join(', ')})`);
                                scopedDocs = docs.length;
                            }
                        }
                        if (wheres.length) spec.where = wheres.join(' AND ');

                        const t0 = performance.now();
                        const hits = await search(spec);
                        const ms = Math.round(performance.now() - t0);
                        outputs.set(id, { spec: {}, hits });
                        this.patchRuntime(id, {
                            status: 'done',
                            hits,
                            count: hits.length,
                            ms,
                            scopedDocs,
                        });
                        break;
                    }
                    case 'results': {
                        outputs.set(id, { spec: {}, hits: scope });
                        this.patchRuntime(id, {
                            status: scope ? 'done' : 'idle',
                            hits: scope,
                            count: scope?.length ?? null,
                        });
                        break;
                    }
                    default:
                        break;
                }
            } catch (err) {
                const msg = err instanceof Error ? err.message : String(err);
                this.patchRuntime(id, { status: 'error', error: msg });
                outputs.set(id, { spec: {}, hits: null });
            }
        }

        this.running = false;
    }
}

/** The singleton, imported by the canvas and every node component. */
export const graph = new WorkflowGraph();
