/**
 * The localStorage boundary for the workflow graph — Zod schemas that parse
 * (don't validate) persisted JSON back into typed values, self-healing bad/old
 * data via `.catch`. Kept separate from the runtime store so the schema set is
 * easy to find and evolve. `safeParseGraph` is the one entry point.
 */
import { z } from 'zod';
import { SearchModeSchema } from '$lib/api';
import { DEFAULT_N, MAX_N, MIN_N, NODE_KINDS } from './types';
import { EXPORT_COLUMNS } from './export';

/** Per-node config as stored (no image File). Every field self-heals to a sane
 *  default on bad/old data via `.catch`, and `n` is clamped to [1, 100]. */
const ConfigSchema = z.object({
  q: z.string().catch(''),
  imageName: z.string().catch(''),
  where: z.string().catch(''),
  language: z.string().catch(''),
  namn: z.string().catch(''),
  mode: SearchModeSchema.catch('fts'),
  n: z
    .number()
    .catch(DEFAULT_N)
    .transform((v) => Math.max(MIN_N, Math.min(MAX_N, Math.round(v)))),
  rerank: z.boolean().catch(false),
  refineScope: z.enum(['video', 'chunk']).catch('video'),
  combineMode: z.enum(['union', 'intersect']).catch('union'),
  tags: z.array(z.string()).catch(() => []),
  exportFormat: z.enum(['json', 'csv']).catch('csv'),
  exportColumns: z.array(z.string()).catch(() => [...EXPORT_COLUMNS]),
  // Atlas modal capture is a full Hit[] (audio_path, alignments, …) — too heavy
  // and stale-prone to round-trip through localStorage, so it is NOT persisted:
  // the schema always heals it to null, and a reload discards the capture (the
  // user re-opens the modal to re-select). TODO: persist a minimal key set
  // (doc_id|speech_id|chunk_id) + rehydrate via /api/atlas/chunks if needed.
  capturedAtlasSelection: z.null().catch(null),
  label: z.string().catch(''),
  enabled: z.boolean().catch(true),
});

const PersistedNodeSchema = z.object({
  id: z.string(),
  type: z.enum(NODE_KINDS),
  position: z.object({ x: z.number(), y: z.number() }),
  // Optional resized dimensions (NodeResizer on Results/Export sink nodes).
  width: z.number().optional(),
  height: z.number().optional(),
});

const PersistedEdgeSchema = z.object({
  id: z.string(),
  source: z.string(),
  target: z.string(),
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
