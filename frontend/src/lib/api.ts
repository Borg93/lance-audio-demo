/**
 * Typed client for the schema-agnostic FastAPI backend.
 *
 * Response envelopes are zod-defined + runtime-validated here, so backend
 * drift surfaces as a clean error instead of silent rendering bugs. The
 * schemas describe transport envelopes (ranking signals, alignments, atlas
 * wire format, graph/voice/topic shapes) — NEVER a corpus's column names.
 * Per-row field access goes through the {@link DatasetView} (see
 * `$lib/descriptor`), which every renderer reads instead of hardcoding fields.
 */

import { tableFromIPC, type Vector } from 'apache-arrow';
import { z } from 'zod';

import {
  activeView,
  AlignmentSchema,
  type Alignment,
  type DatasetView,
  type Row,
  RowSchema,
  type SearchMode,
} from '$lib/descriptor';

export type { Alignment, Row, SearchMode } from '$lib/descriptor';
export { activeView } from '$lib/descriptor';

/** Legacy alias — a search/browse result row. Field access goes through the
 *  active {@link DatasetView}; this stays for import compatibility. */
export type Hit = Row;

// ─────────────────────────────────────────────────────────────────────
// Relevance normalization (pure — reads only the ranking signals)
// ─────────────────────────────────────────────────────────────────────

const COSINE_DISTANCE_MAX = 2;

/** A single comparable relevance number for a hit, normalized so higher is
 *  always better; `null` when the hit carries no ranking signal. */
export function relevanceOf(hit: Hit, mode?: SearchMode): number | null {
  switch (mode) {
    case 'fts':
    case 'scene_fts':
      return hit._score ?? null;
    case 'semantic':
    case 'visual':
      return hit._distance != null ? COSINE_DISTANCE_MAX - hit._distance : null;
    case 'hybrid':
      return hit._relevance_score ?? null;
    case 'scene':
      return null;
    default:
      if (hit._relevance_score != null) return hit._relevance_score;
      if (hit._score != null) return hit._score;
      if (hit._distance != null) return COSINE_DISTANCE_MAX - hit._distance;
      return null;
  }
}

// ─────────────────────────────────────────────────────────────────────
// Search request shape
// ─────────────────────────────────────────────────────────────────────

// Optional fields are written `T | undefined` (exactOptionalPropertyTypes).
export interface SearchSpec {
  q: string;
  n?: number | undefined;
  mode?: SearchMode | undefined;
  rerank?: boolean | undefined;
  rerankN?: number | undefined;
  fuzziness?: (0 | 1 | 2) | undefined;
  phrase?: boolean | undefined;
  /** Hybrid weight ∈ [0,1]: 0 = pure FTS, 1 = pure vector. Undefined = RRF. */
  weight?: number | undefined;
  qVec?: string | undefined;
  where?: string | undefined;
  prefilter?: boolean | undefined;
  /** Structured metadata filters keyed by descriptor filterable field name. */
  filters?: Record<string, string> | undefined;
  /** Topic browse token — matches the dataset's topic layers server-side. */
  topic?: string | undefined;
  image?: File | null | undefined;
  /** Non-default dataset id; omitted for the default DB. */
  dataset?: string | undefined;
}

// ─────────────────────────────────────────────────────────────────────
// Fetch wrappers
// ─────────────────────────────────────────────────────────────────────

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    public readonly detail: string,
  ) {
    super(`api ${status}: ${detail}`);
    this.name = 'ApiError';
  }
}

// RFC 9457 problem+json: DomainError → { detail, title }; FastAPI 422 →
// { title, errors:[...] } with no string `detail`. Parse both keys.
const ProblemSchema = z.object({ detail: z.string().optional(), title: z.string().optional() });

async function apiErrorFrom(r: Response): Promise<ApiError> {
  const body: unknown = await r.json().catch(() => null);
  const parsed = ProblemSchema.safeParse(body);
  const detail =
    (parsed.success ? (parsed.data.detail ?? parsed.data.title) : undefined) ||
    r.statusText ||
    `HTTP ${r.status}`;
  return new ApiError(r.status, detail);
}

async function asJson<T>(r: Response, schema: z.ZodType<T>): Promise<T> {
  if (!r.ok) throw await apiErrorFrom(r);
  return schema.parse(await r.json());
}

const HitsArraySchema = z.array(RowSchema) as z.ZodType<Row[]>;

/** Append the `dataset` selector to a params bag when a non-default dataset is
 *  active or the spec names one. */
function datasetParam(spec?: SearchSpec): string | null {
  if (spec?.dataset) return spec.dataset;
  return activeView().datasetParam();
}

function appendCommonSearchParams(
  out: { append(name: string, value: string): void },
  spec: SearchSpec,
): void {
  if (spec.rerank) out.append('rerank', 'true');
  if (spec.rerank && spec.rerankN !== undefined) out.append('rerank_n', String(spec.rerankN));
  if (spec.weight !== undefined) out.append('weight', String(spec.weight));
  if (spec.qVec) out.append('q_vec', spec.qVec);
  if (spec.where) out.append('where', spec.where);
  if (spec.prefilter === false) out.append('prefilter', 'false');
  // Descriptor-declared filterable fields, marshalled by their own names.
  for (const [field, value] of Object.entries(spec.filters ?? {})) {
    if (value) out.append(field, value);
  }
  if (spec.topic) out.append('topic', spec.topic);
  const ds = datasetParam(spec);
  if (ds) out.append('dataset', ds);
}

/** Run a search. POST + multipart when an image is attached; GET otherwise. */
export async function search(spec: SearchSpec, fetcher: typeof fetch = fetch): Promise<Row[]> {
  const n = String(spec.n ?? 30);
  const mode = spec.mode ?? 'fts';

  if (spec.image) {
    const fd = new FormData();
    fd.append('image', spec.image);
    if (spec.q) fd.append('q', spec.q);
    fd.append('n', n);
    fd.append('mode', mode);
    appendCommonSearchParams(fd, spec);
    const r = await fetcher('/api/search', { method: 'POST', body: fd });
    return asJson(r, HitsArraySchema);
  }

  const params = new URLSearchParams({ q: spec.q, n, mode });
  if (spec.fuzziness) params.append('fuzziness', String(spec.fuzziness));
  if (spec.phrase) params.append('phrase', 'true');
  appendCommonSearchParams(params, spec);
  const r = await fetcher(`/api/search?${params}`);
  return asJson(r, HitsArraySchema);
}

/** `?dataset=` suffix for a bare GET URL (empty for the default dataset). */
function datasetSuffix(): string {
  const ds = activeView().datasetParam();
  return ds ? `?dataset=${encodeURIComponent(ds)}` : '';
}

const ChunkAlignmentsSchema = z.object({ alignments: z.array(AlignmentSchema) });

/** Per-word alignments for one row, fetched on demand when opened in the player.
 *  Search results ship `alignments: []` (the timing blob dominates the payload).
 *  Path arity follows the dataset's identity key fields. */
export async function getChunkAlignments(
  keys: (string | number)[],
  fetcher: typeof fetch = fetch,
): Promise<Alignment[]> {
  const path = keys.map((k) => encodeURIComponent(String(k))).join('/');
  const r = await fetcher(`/api/chunk-alignments/${path}${datasetSuffix()}`);
  const data = await asJson(r, ChunkAlignmentsSchema);
  return data.alignments;
}

const DocTranscriptChunkSchema = RowSchema;
export type DocTranscriptChunk = Row;

const DocTranscriptSchema = z.object({
  doc_id: z.string(),
  chunks: z.array(DocTranscriptChunkSchema),
});
export type DocTranscript = z.infer<typeof DocTranscriptSchema>;

// Bounded LRU of in-flight/resolved transcripts (immutable per doc, heavy
// payload) so re-opening any row in the same doc is instant.
const MAX_DOC_TRANSCRIPTS = 50;
const docTranscriptCache = new Map<string, Promise<DocTranscript>>();

/** Whole-document transcript, chunk-segmented + ordered. Lazy-fetched when a
 *  row opens so playback past the selected chunk still has karaoke. */
export async function getDocTranscript(
  docId: string,
  fetcher: typeof fetch = fetch,
): Promise<DocTranscript> {
  const suffix = datasetSuffix();
  const fetchOnce = async (): Promise<DocTranscript> => {
    const r = await fetcher(`/api/doc-transcript/${encodeURIComponent(docId)}${suffix}`);
    return asJson(r, DocTranscriptSchema);
  };
  if (fetcher !== fetch) return fetchOnce();
  const cacheKey = `${activeView().id}/${docId}`;
  const cached = docTranscriptCache.get(cacheKey);
  if (cached) {
    docTranscriptCache.delete(cacheKey);
    docTranscriptCache.set(cacheKey, cached);
    return cached;
  }
  const p: Promise<DocTranscript> = fetchOnce().catch((e: unknown) => {
    if (docTranscriptCache.get(cacheKey) === p) docTranscriptCache.delete(cacheKey);
    throw e;
  });
  docTranscriptCache.set(cacheKey, p);
  if (docTranscriptCache.size > MAX_DOC_TRANSCRIPTS) {
    const oldest = docTranscriptCache.keys().next().value;
    if (oldest !== undefined) docTranscriptCache.delete(oldest);
  }
  return p;
}

// ── Diarization (Speakers tab, capability-gated) ────────────────────────────
const DiarTurnSchema = z.object({
  turn_id: z.number().int(),
  speaker: z.string(),
  start: z.number(),
  end: z.number(),
});
export type DiarTurn = z.infer<typeof DiarTurnSchema>;

const DiarizationResponseSchema = z.object({
  built: z.boolean(),
  doc_id: z.string(),
  turns: z.array(DiarTurnSchema),
  speakers: z.array(z.string()),
});
export type DiarizationResponse = z.infer<typeof DiarizationResponseSchema>;

export async function getDiarization(
  docId: string,
  fetcher: typeof fetch = fetch,
): Promise<DiarizationResponse> {
  const r = await fetcher(`/api/diarization/${encodeURIComponent(docId)}${datasetSuffix()}`);
  return asJson(r, DiarizationResponseSchema);
}

// ── Health ──────────────────────────────────────────────────────────────
const PingSchema = z.object({ ok: z.boolean(), url: z.string(), error: z.string().nullish() });
const HealthSchema = z.object({
  db: z.object({
    path: z.string(),
    tables: z.array(z.string()),
    chunks: z.number(),
    documents: z.number(),
  }),
  embed: PingSchema,
  rerank: PingSchema,
});
export type Health = z.infer<typeof HealthSchema>;

export async function getHealth(fetcher: typeof fetch = fetch): Promise<Health> {
  const r = await fetcher('/api/health');
  return asJson(r, HealthSchema);
}

// ── Documents gallery (row envelope — corpus fields read via the view) ──────
const DocumentSchema = RowSchema;
export type Document = Row;

const DocumentsResponseSchema = z.object({
  total: z.number().int(),
  page: z.number().int(),
  docs: z.array(DocumentSchema),
});
export type DocumentsResponse = z.infer<typeof DocumentsResponseSchema>;

export async function listDocuments(
  page = 1,
  perPage = 24,
  fetcher: typeof fetch = fetch,
): Promise<DocumentsResponse> {
  const suffix = activeView().datasetParam();
  const ds = suffix ? `&dataset=${encodeURIComponent(suffix)}` : '';
  const r = await fetcher(`/api/documents?page=${page}&per_page=${perPage}${ds}`);
  return asJson(r, DocumentsResponseSchema);
}

// ── Filterable columns ───────────────────────────────────────────────────
const ColumnSchema = z.object({ name: z.string(), type: z.string() });
export type ColumnInfo = z.infer<typeof ColumnSchema>;

export async function listColumns(fetcher: typeof fetch = fetch): Promise<ColumnInfo[]> {
  const r = await fetcher(`/api/columns${datasetSuffix()}`);
  return asJson(r, z.array(ColumnSchema));
}

// ── Descriptor + dataset discovery ──────────────────────────────────────────
import {
  DatasetDescriptorSchema,
  DatasetsResponseSchema,
  DatasetView as DatasetViewClass,
} from '$lib/descriptor';

/** List the datasets the backend serves (id + table stats + capabilities). */
export async function listDatasets(fetcher: typeof fetch = fetch) {
  const r = await fetcher('/api/datasets');
  return asJson(r, DatasetsResponseSchema).then((d) => d.datasets);
}

/** Fetch + parse one dataset's descriptor and wrap it in a DatasetView. */
export async function getDatasetView(
  datasetId: string,
  isDefault: boolean,
  fetcher: typeof fetch = fetch,
): Promise<DatasetView> {
  const r = await fetcher(`/api/datasets/${encodeURIComponent(datasetId)}/descriptor`);
  if (!r.ok) throw await apiErrorFrom(r);
  // Parse directly (not via asJson) so the value keeps the schema's OUTPUT type,
  // where `.default()`-ed fields are required — the DatasetView ctor's shape.
  const descriptor = DatasetDescriptorSchema.parse(await r.json());
  return new DatasetViewClass(descriptor).withDatasetParam(isDefault);
}

// ── Row-media URL helpers (identity arity from the active view) ──────────────
export const thumbnailUrl = (row: Row): string => activeView().thumbnailUrl(row);
export const chunkFrameUrl = (row: Row): string => activeView().frameUrl(row);
export const mediaUrl = (row: Row): string => activeView().mediaUrl(row);

// ── Embedding Atlas ───────────────────────────────────────────────────────
/** The projection spaces the atlas can render — names come from the descriptor
 *  (`declared.atlas[].name`), so this is `string`, not a fixed corpus set. */
export type AtlasSpace = string;

const AtlasStatusSchema = z.object({
  projected: z.boolean(),
  rows: z.number().int(),
  space: z.string().optional(),
  // Which named spaces are built (gates the space toggle).
  spaces: z.record(z.string(), z.boolean()).optional(),
});
export type AtlasStatus = z.infer<typeof AtlasStatusSchema>;

export async function getAtlasStatus(
  space: AtlasSpace,
  fetcher: typeof fetch = fetch,
): Promise<AtlasStatus> {
  const ds = activeView().datasetParam();
  const q = ds ? `&dataset=${encodeURIComponent(ds)}` : '';
  return asJson(await fetcher(`/api/atlas/status?space=${encodeURIComponent(space)}${q}`), AtlasStatusSchema);
}

/** A factorized (codes, labels) pair from one Arrow DICTIONARY column. */
export interface DictColumn {
  codes: Int32Array;
  labels: string[];
}

/** Decoded point arrays for a space. Identity keys beyond the doc key and the
 *  categorical colour channels are keyed by name (descriptor-driven), so no
 *  corpus column appears in the type. */
export interface AtlasPoints {
  count: number;
  space?: string;
  x: Float32Array;
  y: Float32Array;
  xBits: Uint16Array;
  yBits: Uint16Array;
  docs: string[];
  doc: Int32Array;
  docFiles?: string[];
  rowid?: number[];
  cluster?: Int32Array;
  /** Non-doc identity key columns (e.g. the 2nd/3rd key field) by field name. */
  keys: Record<string, number[]>;
  /** Categorical colour channels (language/topic/…) by channel name. */
  channels: Record<string, DictColumn>;
}

function dictColumn(vec: Vector | null): DictColumn | null {
  if (!vec) return null;
  let labels: string[] = [];
  for (const d of vec.data) {
    if (d.dictionary && labels.length === 0) labels = d.dictionary.toArray() as string[];
  }
  return { codes: concatInt32(vec), labels };
}

function concatInt32(vec: Vector): Int32Array {
  const chunks = vec.data;
  if (chunks.length === 1) return chunks[0]!.values as Int32Array;
  let total = 0;
  for (const d of chunks) total += d.length;
  const out = new Int32Array(total);
  let off = 0;
  for (const d of chunks) {
    out.set((d.values as Int32Array).subarray(0, d.length), off);
    off += d.length;
  }
  return out;
}

function int32Column(vec: Vector | null): Int32Array {
  return (vec?.toArray() ?? new Int32Array()) as Int32Array;
}

function u16Column(vec: Vector | null): Uint16Array {
  return (vec?.toArray() ?? new Uint16Array()) as Uint16Array;
}

/** Decode raw float16 bits → an owned Float32Array (CPU math). */
function f16ToF32(bits: Uint16Array): Float32Array {
  const out = new Float32Array(bits.length);
  for (let i = 0; i < bits.length; i++) {
    const h = bits[i]!;
    const expo = (h & 0x7c00) >> 10;
    const sigf = (h & 0x03ff) / 1024;
    const sign = (h & 0x8000) === 0 ? 1 : -1;
    if (expo === 0x1f) out[i] = sign * (sigf ? Number.NaN : Infinity);
    else if (expo === 0x00) out[i] = sign * (sigf ? 6.103515625e-5 * sigf : 0);
    else out[i] = sign * 2 ** (expo - 15) * (1 + sigf);
  }
  return out;
}

function numberColumn(vec: Vector | null): number[] {
  if (!vec) return [];
  const out: number[] = [];
  for (const v of vec) out.push(Number(v));
  return out;
}

/** Fetch the point arrays for a space as one Apache Arrow IPC stream. Identity
 *  keys (minus the doc key) and the declared channels are read by name from the
 *  active view, so the reader names no corpus column. */
export async function getAtlasPoints(
  space: AtlasSpace,
  fetcher: typeof fetch = fetch,
): Promise<AtlasPoints> {
  const view = activeView();
  const ds = view.datasetParam();
  const dsq = ds ? `&dataset=${encodeURIComponent(ds)}` : '';
  const r = await fetcher(`/api/atlas/points?space=${encodeURIComponent(space)}&v=6${dsq}`);
  if (!r.ok) throw await apiErrorFrom(r);

  const buf = await r.arrayBuffer();
  const table = tableFromIPC(new Uint8Array(buf));
  const md = table.schema.metadata;

  const count = Number(md.get('count') ?? table.numRows);
  const spaceMeta = md.get('space');
  const docFilesMeta = md.get('docFiles');

  const doc = dictColumn(table.getChild('doc'));
  if (!doc) throw new ApiError(500, 'malformed /api/atlas/points payload (no doc column)');

  const xBits = u16Column(table.getChild('x'));
  const yBits = u16Column(table.getChild('y'));

  // Non-doc identity key columns, by field name.
  const keys: Record<string, number[]> = {};
  for (const field of view.keyFields) {
    if (field === view.docKeyField) continue;
    const col = table.getChild(field);
    if (col) keys[field] = numberColumn(col);
  }

  // Declared categorical channels, by channel name.
  const channels: Record<string, DictColumn> = {};
  for (const name of view.atlasChannels(space)) {
    const col = dictColumn(table.getChild(name));
    if (col) channels[name] = col;
  }

  const data: AtlasPoints = {
    count,
    docs: doc.labels,
    doc: doc.codes,
    xBits,
    yBits,
    x: f16ToF32(xBits),
    y: f16ToF32(yBits),
    keys,
    channels,
  };
  if (spaceMeta) data.space = spaceMeta;
  if (docFilesMeta) data.docFiles = z.array(z.string()).parse(JSON.parse(docFilesMeta));
  const rowid = table.getChild('rowid');
  if (rowid) data.rowid = numberColumn(rowid);
  const cluster = table.getChild('cluster');
  if (cluster) data.cluster = int32Column(cluster);

  if (!Number.isFinite(data.count) || data.x.length !== data.count) {
    throw new ApiError(500, 'malformed /api/atlas/points payload');
  }
  return data;
}

/** Full row for one point (detail pane + playback), looked up by identity keys. */
export async function getAtlasChunk(
  keys: (string | number)[],
  fetcher: typeof fetch = fetch,
): Promise<Row> {
  const path = keys.map((k) => encodeURIComponent(String(k))).join('/');
  const r = await fetcher(`/api/atlas/chunk/${path}${datasetSuffix()}`);
  return asJson(r, RowSchema);
}

/** Full rows for a selection, addressed by stable Lance `_rowid`. */
export async function getAtlasChunks(rowids: number[], fetcher: typeof fetch = fetch): Promise<Row[]> {
  const ds = activeView().datasetParam();
  const url = ds ? `/api/atlas/chunks?dataset=${encodeURIComponent(ds)}` : '/api/atlas/chunks';
  const r = await fetcher(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ rowids }),
  });
  return asJson(r, HitsArraySchema);
}

// ── Topics (Tree page, capability-gated) ────────────────────────────────────
export interface TopicNode {
  name: string;
  value?: number | undefined;
  children?: TopicNode[] | undefined;
}

const TopicNodeSchema: z.ZodType<TopicNode> = z.lazy(() =>
  z.object({
    name: z.string(),
    value: z.number().optional(),
    children: z.array(TopicNodeSchema).optional(),
  }),
);

const TopicsResponseSchema = z.object({
  built: z.boolean(),
  layers: z.number().int(),
  n_chunks: z.number().int(),
  hierarchy: TopicNodeSchema.nullable(),
  noise_label: z.string().optional(),
});
export type TopicsResponse = z.infer<typeof TopicsResponseSchema>;

export async function getTopics(fetcher: typeof fetch = fetch): Promise<TopicsResponse> {
  return asJson(await fetcher(`/api/topics${datasetSuffix()}`), TopicsResponseSchema);
}

// ── Knowledge graph (Graph page, capability-gated) ──────────────────────────
const ENTITY_ID_RE = /^[0-9a-f]{16}$/;

export function isEntityId(id: string): boolean {
  return ENTITY_ID_RE.test(id);
}

export const GraphStatusSchema = z.object({
  built: z.boolean(),
  entities: z.number().int(),
  relations: z.number().int(),
  mentions: z.number().int(),
  videos: z.number().int(),
});
export type GraphStatus = z.infer<typeof GraphStatusSchema>;

export async function getGraphStatus(fetcher: typeof fetch = fetch): Promise<GraphStatus> {
  return asJson(await fetcher(`/api/graph/status${datasetSuffix()}`), GraphStatusSchema);
}

const CypherValueSchema = z.union([z.string(), z.number(), z.null()]);
export type CypherValue = z.infer<typeof CypherValueSchema>;

export const GraphCypherResponseSchema = z.object({
  built: z.boolean(),
  columns: z.array(z.string()),
  rows: z.array(z.array(CypherValueSchema)),
  error: z.string().nullable(),
});
export type GraphCypherResponse = z.infer<typeof GraphCypherResponseSchema>;

export async function runGraphCypher(
  query: string,
  limit = 200,
  fetcher: typeof fetch = fetch,
): Promise<GraphCypherResponse> {
  const ds = activeView().datasetParam();
  const url = ds ? `/api/graph/cypher?dataset=${encodeURIComponent(ds)}` : '/api/graph/cypher';
  const r = await fetcher(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, limit }),
  });
  return asJson(r, GraphCypherResponseSchema);
}

export const GraphMatchSchema = z.object({
  entity_id: z.string(),
  name: z.string(),
  entity_type: z.string(),
  mention_count: z.number().int(),
  videos: z.number().int(),
});
export type GraphMatch = z.infer<typeof GraphMatchSchema>;

const GraphSearchResponseSchema = z.object({
  built: z.boolean(),
  matches: z.array(GraphMatchSchema),
});
export type GraphSearchResponse = z.infer<typeof GraphSearchResponseSchema>;

export async function searchGraphEntities(
  q: string,
  fetcher: typeof fetch = fetch,
): Promise<GraphSearchResponse> {
  const ds = activeView().datasetParam();
  const dsq = ds ? `&dataset=${encodeURIComponent(ds)}` : '';
  return asJson(
    await fetcher(`/api/graph/search?q=${encodeURIComponent(q)}${dsq}`),
    GraphSearchResponseSchema,
  );
}

const GraphEntitySchema = z.object({
  entity_id: z.string(),
  name: z.string(),
  entity_type: z.string(),
  mention_count: z.number().int(),
});
export type GraphEntity = z.infer<typeof GraphEntitySchema>;

// A clip carries its doc id + time span + body text plus a display title; the
// title's source column is the dataset's (graph_presets.clip_title_column).
const GraphClipSchema = z.object({
  chunk_id: z.string(),
  doc_id: z.string(),
  title: z.string(),
  start: z.number(),
  end: z.number(),
  text: z.string(),
});
export type GraphClip = z.infer<typeof GraphClipSchema>;

const GraphNeighborSchema = z.object({
  entity_id: z.string(),
  name: z.string(),
  entity_type: z.string(),
  direction: z.enum(['out', 'in']),
  description: z.string(),
});
export type GraphNeighbor = z.infer<typeof GraphNeighborSchema>;

const GraphCooccurSchema = z.object({
  entity_id: z.string(),
  name: z.string(),
  shared: z.number().int(),
});
export type GraphCooccur = z.infer<typeof GraphCooccurSchema>;

export const GraphEntityResponseSchema = z.object({
  built: z.boolean(),
  entity: GraphEntitySchema.nullable(),
  clips: z.array(GraphClipSchema),
  neighbors: z.array(GraphNeighborSchema),
  cooccur: z.array(GraphCooccurSchema),
});
export type GraphEntityResponse = z.infer<typeof GraphEntityResponseSchema>;

export async function getGraphEntity(
  entityId: string,
  fetcher: typeof fetch = fetch,
): Promise<GraphEntityResponse> {
  if (!isEntityId(entityId)) throw new ApiError(400, `invalid entity id: ${entityId}`);
  return asJson(
    await fetcher(`/api/graph/entity/${encodeURIComponent(entityId)}${datasetSuffix()}`),
    GraphEntityResponseSchema,
  );
}

const GraphNodeSchema = z.object({
  id: z.string(),
  name: z.string(),
  type: z.string(),
  mentions: z.number().int(),
  videos: z.number().int(),
});
export type GraphNode = z.infer<typeof GraphNodeSchema>;

const GraphEdgeSchema = z.object({
  source: z.string(),
  target: z.string(),
  description: z.string(),
});
export type GraphEdge = z.infer<typeof GraphEdgeSchema>;

export const GraphSubgraphResponseSchema = z.object({
  built: z.boolean(),
  nodes: z.array(GraphNodeSchema),
  edges: z.array(GraphEdgeSchema),
});
export type GraphSubgraphResponse = z.infer<typeof GraphSubgraphResponseSchema>;

export async function getGraphSubgraph(
  entityId?: string,
  limit = 150,
  fetcher: typeof fetch = fetch,
): Promise<GraphSubgraphResponse> {
  const params = new URLSearchParams({ limit: String(limit) });
  if (entityId !== undefined) {
    if (!isEntityId(entityId)) throw new ApiError(400, `invalid entity id: ${entityId}`);
    params.set('entity_id', entityId);
  }
  const ds = activeView().datasetParam();
  if (ds) params.set('dataset', ds);
  return asJson(await fetcher(`/api/graph/subgraph?${params}`), GraphSubgraphResponseSchema);
}

// ── Voice search ("Find this voice", capability-gated) ──────────────────────
const VoiceStatusSchema = z.object({
  built: z.boolean(),
  turns: z.number().int(),
  speakers: z.number().int(),
});
export type VoiceStatus = z.infer<typeof VoiceStatusSchema>;

export async function getVoiceStatus(fetcher: typeof fetch = fetch): Promise<VoiceStatus> {
  return asJson(await fetcher(`/api/voice/status${datasetSuffix()}`), VoiceStatusSchema);
}

/** A voice-ranked hit: the matched row (renders as a normal result card) plus
 *  the matched speaker turn. The row envelope is a {@link Row}; the turn fields
 *  are the voice capability's own contract. */
export const VoiceHitSchema = RowSchema.and(
  z.object({
    speaker_label: z.string(),
    turn_id: z.number().int(),
    turn_start: z.number(),
    turn_end: z.number(),
    _distance: z.number(),
    turn_score: z.number(),
  }),
) as z.ZodType<VoiceHit>;
export type VoiceHit = Row & {
  speaker_label: string;
  turn_id: number;
  turn_start: number;
  turn_end: number;
  _distance: number;
  turn_score: number;
};

const VoiceQueryInfoSchema = z.object({
  doc_id: z.string().nullable(),
  speaker_label: z.string().nullable(),
  turn_id: z.number().int().nullable(),
  turn_start: z.number().nullable(),
  turn_end: z.number().nullable(),
});
export type VoiceQueryInfo = z.infer<typeof VoiceQueryInfoSchema>;

export const VoiceSimilarResponseSchema = z.object({
  query: VoiceQueryInfoSchema,
  hits: z.array(VoiceHitSchema),
});
export type VoiceSimilarResponse = z.infer<typeof VoiceSimilarResponseSchema>;

/** Query-by-example anchor: a document plus exactly one locator. */
export type VoiceAnchor = { docId: string } & (
  | { turnId: number }
  | { speaker: string }
  | { t: number }
);

export async function voiceSimilar(
  anchor: VoiceAnchor,
  opts: { n?: number | undefined; excludeSameDoc?: boolean | undefined } = {},
  fetcher: typeof fetch = fetch,
): Promise<VoiceSimilarResponse> {
  const params = new URLSearchParams({ doc_id: anchor.docId });
  if ('turnId' in anchor) params.set('turn_id', String(anchor.turnId));
  else if ('speaker' in anchor) params.set('speaker', anchor.speaker);
  else params.set('t', String(anchor.t));
  if (opts.n !== undefined) params.set('n', String(opts.n));
  if (opts.excludeSameDoc !== undefined) params.set('exclude_same_doc', String(opts.excludeSameDoc));
  const ds = activeView().datasetParam();
  if (ds) params.set('dataset', ds);
  return asJson(await fetcher(`/api/voice/similar?${params}`), VoiceSimilarResponseSchema);
}

export async function voiceSimilarUpload(
  file: File,
  opts: { n?: number | undefined } = {},
  fetcher: typeof fetch = fetch,
): Promise<VoiceSimilarResponse> {
  const params = new URLSearchParams();
  if (opts.n !== undefined) params.set('n', String(opts.n));
  const ds = activeView().datasetParam();
  if (ds) params.set('dataset', ds);
  const fd = new FormData();
  fd.append('file', file);
  const url = params.size > 0 ? `/api/voice/similar?${params}` : '/api/voice/similar';
  const r = await fetcher(url, { method: 'POST', body: fd });
  return asJson(r, VoiceSimilarResponseSchema);
}

export type VoiceBand = 'strong' | 'possible';

export function voiceBandOf(turnScore: number): VoiceBand | null {
  if (turnScore >= 0.7) return 'strong';
  if (turnScore >= 0.6) return 'possible';
  return null;
}

/** Narrow a Row to a VoiceHit so shared result components render the speaker
 *  chip only for voice-mode hits. Structural check (validated at fetch). */
export function isVoiceHit(hit: Row): hit is VoiceHit {
  return 'turn_score' in hit && 'speaker_label' in hit;
}
