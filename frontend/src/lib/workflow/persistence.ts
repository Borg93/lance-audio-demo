/**
 * The localStorage boundary for the workflow graph — Zod schemas that parse
 * (don't validate) persisted JSON back into typed values, self-healing bad/old
 * data via `.catch`. Kept separate from the runtime store so the schema set is
 * easy to find and evolve. `safeParseGraph` is the one entry point.
 */
import { z } from 'zod';
import type { SearchMode } from '$lib/api';
import { DEFAULT_N, MAX_N, MIN_N, NODE_KINDS } from './types';

// SearchMode is a plain union in the descriptor (no zod schema is exported), so
// mirror its members here to parse a persisted `mode`, healing anything unknown
// to 'fts'. `satisfies` keeps this list a subset of the SearchMode union.
const MODE_VALUES = [
  'fts',
  'semantic',
  'visual',
  'scene',
  'scene_fts',
  'hybrid',
  'all',
] as const satisfies readonly SearchMode[];
const SearchModeSchema = z.enum(MODE_VALUES);

/** Per-node config as stored (no image File). Every field self-heals to a sane
 *  default on bad/old data via `.catch`, and `n` is clamped to [1, 100]. */
const ConfigSchema = z.object({
  q: z.string().catch(''),
  imageName: z.string().catch(''),
  where: z.string().catch(''),
  filters: z.record(z.string(), z.string()).catch(() => ({})),
  mode: SearchModeSchema.catch('fts'),
  n: z
    .number()
    .catch(DEFAULT_N)
    .transform((v) => Math.max(MIN_N, Math.min(MAX_N, Math.round(v)))),
  rerank: z.boolean().catch(false),
  minScore: z.number().nullable().catch(null),
  refineScope: z.enum(['video', 'chunk']).catch('video'),
  combineMode: z.enum(['union', 'intersect']).catch('union'),
  tags: z.array(z.string()).catch(() => []),
  exportFormat: z.enum(['json', 'csv']).catch('csv'),
  // `null` = every column the active dataset offers; stale field names in a
  // persisted array self-heal at export time (orderColumns drops unknowns).
  exportColumns: z.array(z.string()).nullable().catch(null),
  // Atlas modal capture is a full Hit[] (media path, alignments, …) — too heavy
  // and stale-prone to round-trip through localStorage, so it is NOT persisted:
  // the schema always heals it to null, and a reload discards the capture (the
  // user re-opens the modal to re-select). TODO: persist a minimal identity-key
  // set + rehydrate via /api/atlas/chunks if needed.
  capturedAtlasSelection: z.null().catch(null),
  label: z.string().catch(''),
  enabled: z.boolean().catch(true),
});

// Nodes have fixed sizes (no NodeResizer) — old persisted width/height keys are
// unknown to this schema and Zod strips them, so stale data still parses.
const PersistedNodeSchema = z.object({
  id: z.string(),
  type: z.enum(NODE_KINDS),
  position: z.object({ x: z.number(), y: z.number() }),
});

const PersistedEdgeSchema = z.object({
  id: z.string(),
  source: z.string(),
  target: z.string(),
  // Which target port the edge lands on (Search has "in" + "image"); absent in
  // pre-two-port snapshots — the loader infers it from the source kind.
  targetHandle: z.string().optional(),
  label: z.string().optional(),
});

/** A structurally-bad node (unknown kind, missing position) fails the whole
 *  parse, so the caller falls back to `seed()` instead of crashing the canvas. */
const PersistedGraphSchema = z.object({
  nodes: z.array(PersistedNodeSchema).min(1),
  edges: z.array(PersistedEdgeSchema).default([]),
  config: z.record(z.string(), ConfigSchema).default({}),
  tags: z.record(z.string(), z.array(z.string())).default({}),
});

export type PersistedConfig = z.infer<typeof ConfigSchema>;
export type PersistedGraph = z.infer<typeof PersistedGraphSchema>;

/** Parse a snapshot string into a typed graph, or null if absent/malformed. */
export function safeParseGraph(raw: string | null): PersistedGraph | null {
  if (!raw) return null;
  try {
    const result = PersistedGraphSchema.safeParse(JSON.parse(raw));
    return result.success ? result.data : null;
  } catch {
    return null; // not valid JSON
  }
}
