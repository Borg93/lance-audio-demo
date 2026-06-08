/**
 * Typed client for the FastAPI backend (`backend/app.py`).
 *
 * Schemas are zod-defined here and runtime-validated, so backend schema
 * drift surfaces as a clean error in the UI instead of silent rendering
 * bugs (the old plain-HTML frontend had several of those).
 */

import { tableFromIPC, type Vector } from 'apache-arrow';
import { z } from 'zod';

// ─────────────────────────────────────────────────────────────────────
// Schemas (mirror src/raudio/schema.py:CHUNK_SCHEMA + DOC_SCHEMA)
// ─────────────────────────────────────────────────────────────────────

export const SearchModeSchema = z.enum([
  'fts',
  'semantic',
  'visual',
  'scene',
  'scene_fts',
  'hybrid',
  'all',
]);
export type SearchMode = z.infer<typeof SearchModeSchema>;

const WordSchema = z.object({
  text: z.string(),
  start: z.number(),
  end: z.number(),
  score: z.number().optional(),
});
export type Word = z.infer<typeof WordSchema>;

const AlignmentSchema = z.object({
  start: z.number(),
  end: z.number(),
  text: z.string(),
  duration: z.number().optional(),
  score: z.number().optional(),
  words: z.array(WordSchema).optional(),
});
export type Alignment = z.infer<typeof AlignmentSchema>;

export const HitSchema = z.object({
  _score: z.number().optional(),
  doc_id: z.string(),
  audio_path: z.string(),
  speech_id: z.number().int(),
  chunk_id: z.number().int(),
  start: z.number(),
  end: z.number(),
  duration: z.number().nullable().optional(),
  text: z.string(),
  language: z.string().nullable().optional(),
  namn: z.string().nullable().optional(),
  referenskod: z.string().nullable().optional(),
  bildid: z.string().nullable().optional(),
  extraid: z.string().nullable().optional(),
  // AI-written Swedish caption of the chunk's representative frame. Present
  // only once captions are built (`raudio feature caption`); null otherwise.
  caption: z.string().nullable().optional(),
  // Backend (`_postprocess_hits`) always emits this field — empty array
  // when the chunk has no alignments — so we keep it required here.
  alignments: z.array(AlignmentSchema),
});
export type Hit = z.infer<typeof HitSchema>;

export const DocumentSchema = z.object({
  doc_id: z.string(),
  audio_path: z.string(),
  duration: z.number().nullable().optional(),
  referenskod: z.string().nullable().optional(),
  namn: z.string().nullable().optional(),
  bildid: z.string().nullable().optional(),
  extraid: z.string().nullable().optional(),
});
export type Document = z.infer<typeof DocumentSchema>;

export const DocumentsResponseSchema = z.object({
  total: z.number().int(),
  page: z.number().int(),
  docs: z.array(DocumentSchema),
});
export type DocumentsResponse = z.infer<typeof DocumentsResponseSchema>;

// ─────────────────────────────────────────────────────────────────────
// Search request shape
// ─────────────────────────────────────────────────────────────────────

// Optional fields are written `T | undefined` (not just `?: T`) because the
// callers build specs with explicit `undefined` values and the project runs
// with `exactOptionalPropertyTypes`, which distinguishes "absent" from
// "present and undefined".
export interface SearchSpec {
  q: string;
  n?: number | undefined;
  mode?: SearchMode | undefined;
  rerank?: boolean | undefined;
  /** How many candidates the cross-encoder reranker scores (when rerank=true). */
  rerankN?: number | undefined;
  fuzziness?: (0 | 1 | 2) | undefined;
  phrase?: boolean | undefined;
  /** Hybrid weight ∈ [0,1]: 0 = pure FTS, 1 = pure vector. Undefined = RRF. */
  weight?: number | undefined;
  /** Separate text for the vector leg of hybrid/semantic/all; falls back to `q` when empty. */
  qVec?: string | undefined;
  /** Raw SQL WHERE expression ANDed with the structured metadata filters. */
  where?: string | undefined;
  /** Apply filter before vector/FTS search (prefilter) vs after (postfilter). Defaults true server-side. */
  prefilter?: boolean | undefined;
  language?: string | undefined;
  namn?: string | undefined;
  referenskod?: string | undefined;
  extraid?: string | undefined;
  /** Topic name (Tree page) — matches any topic_l* layer; browses that topic's chunks. */
  topic?: string | undefined;
  image?: File | null | undefined;
}

// ─────────────────────────────────────────────────────────────────────
// Fetch wrappers
// ─────────────────────────────────────────────────────────────────────

export class ApiError extends Error {
  constructor(
    public status: number,
    public detail: string,
  ) {
    super(`api ${status}: ${detail}`);
  }
}

async function asJson<T>(r: Response, schema: z.ZodType<T>): Promise<T> {
  if (!r.ok) {
    const body = await r.json().catch(() => ({}));
    throw new ApiError(r.status, body?.detail ?? r.statusText);
  }
  return schema.parse(await r.json());
}

const HitsArraySchema = z.array(HitSchema);

/** Run a search. Uses POST + multipart when an image is attached; GET otherwise. */
export async function search(spec: SearchSpec, fetcher: typeof fetch = fetch): Promise<Hit[]> {
  const n = String(spec.n ?? 30);
  const mode = spec.mode ?? 'fts';

  if (spec.image) {
    const fd = new FormData();
    fd.append('image', spec.image);
    if (spec.q) fd.append('q', spec.q);
    fd.append('n', n);
    fd.append('mode', mode);
    if (spec.rerank) fd.append('rerank', 'true');
    if (spec.rerank && spec.rerankN !== undefined) fd.append('rerank_n', String(spec.rerankN));
    if (spec.weight !== undefined) fd.append('weight', String(spec.weight));
    if (spec.qVec) fd.append('q_vec', spec.qVec);
    if (spec.where) fd.append('where', spec.where);
    if (spec.prefilter === false) fd.append('prefilter', 'false');
    if (spec.language) fd.append('language', spec.language);
    if (spec.namn) fd.append('namn', spec.namn);
    if (spec.referenskod) fd.append('referenskod', spec.referenskod);
    if (spec.extraid) fd.append('extraid', spec.extraid);
    if (spec.topic) fd.append('topic', spec.topic);
    const r = await fetcher('/api/search', { method: 'POST', body: fd });
    return asJson(r, HitsArraySchema);
  }

  const params = new URLSearchParams({ q: spec.q, n, mode });
  if (spec.fuzziness) params.set('fuzziness', String(spec.fuzziness));
  if (spec.phrase) params.set('phrase', 'true');
  if (spec.rerank) params.set('rerank', 'true');
  if (spec.rerank && spec.rerankN !== undefined) params.set('rerank_n', String(spec.rerankN));
  if (spec.weight !== undefined) params.set('weight', String(spec.weight));
  if (spec.qVec) params.set('q_vec', spec.qVec);
  if (spec.where) params.set('where', spec.where);
  if (spec.prefilter === false) params.set('prefilter', 'false');
  if (spec.language) params.set('language', spec.language);
  if (spec.namn) params.set('namn', spec.namn);
  if (spec.referenskod) params.set('referenskod', spec.referenskod);
  if (spec.extraid) params.set('extraid', spec.extraid);
  if (spec.topic) params.set('topic', spec.topic);
  const r = await fetcher(`/api/search?${params}`);
  return asJson(r, HitsArraySchema);
}

const ChunkAlignmentsSchema = z.object({ alignments: z.array(AlignmentSchema) });

/** Per-word alignments for one chunk, fetched on demand when a hit is opened in
 *  the player. Search results ship `alignments: []` (the timing blob is ~80% of a
 *  search payload and only the selected hit renders it); the player calls this for
 *  the open hit. */
export async function getChunkAlignments(
  doc_id: string,
  speech_id: number,
  chunk_id: number,
  fetcher: typeof fetch = fetch,
): Promise<Alignment[]> {
  const r = await fetcher(
    `/api/chunk-alignments/${encodeURIComponent(doc_id)}/${speech_id}/${chunk_id}`,
  );
  const data = await asJson(r, ChunkAlignmentsSchema);
  return data.alignments;
}

const DocTranscriptChunkSchema = z.object({
  speech_id: z.number().int(),
  chunk_id: z.number().int(),
  start: z.number(),
  end: z.number(),
  text: z.string(),
  alignments: z.array(AlignmentSchema),
});
export type DocTranscriptChunk = z.infer<typeof DocTranscriptChunkSchema>;

const DocTranscriptSchema = z.object({
  doc_id: z.string(),
  chunks: z.array(DocTranscriptChunkSchema),
});
export type DocTranscript = z.infer<typeof DocTranscriptSchema>;

/** Whole-document transcript, chunk-segmented + ordered. Lazy-fetched when a
 *  hit opens so playback past the selected chunk still has karaoke. The player
 *  flattens chunks[].alignments; a future timeline can use the chunk envelope. */
export async function getDocTranscript(
  doc_id: string,
  fetcher: typeof fetch = fetch,
): Promise<DocTranscript> {
  const r = await fetcher(`/api/doc-transcript/${encodeURIComponent(doc_id)}`);
  return asJson(r, DocTranscriptSchema);
}

// ── Health ──────────────────────────────────────────────────────────────
const PingSchema = z.object({ ok: z.boolean(), url: z.string(), error: z.string().optional() });
export const HealthSchema = z.object({
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

/** Paginated documents list for the gallery. */
export async function listDocuments(
  page = 1,
  perPage = 24,
  fetcher: typeof fetch = fetch,
): Promise<DocumentsResponse> {
  const r = await fetcher(`/api/documents?page=${page}&per_page=${perPage}`);
  return asJson(r, DocumentsResponseSchema);
}

// ── Filterable columns ───────────────────────────────────────────────────
export const ColumnSchema = z.object({ name: z.string(), type: z.string() });
export type ColumnInfo = z.infer<typeof ColumnSchema>;

/** The chunks table's filterable scalar columns (name + friendly type). */
export async function listColumns(fetcher: typeof fetch = fetch): Promise<ColumnInfo[]> {
  const r = await fetcher('/api/columns');
  return asJson(r, z.array(ColumnSchema));
}

/** URL helpers — used directly as `<img src=...>`, no fetch. */
export const thumbnailUrl = (doc_id: string) => `/api/thumbnail/${encodeURIComponent(doc_id)}`;
export const chunkFrameUrl = (doc_id: string, speech_id: number, chunk_id: number) =>
  `/api/chunk-frame/${encodeURIComponent(doc_id)}/${speech_id}/${chunk_id}`;
export const mediaUrl = (doc_id: string) => `/api/media/${encodeURIComponent(doc_id)}`;

// ── Embedding Atlas ───────────────────────────────────────────────────────
// The Atlas tab renders a precomputed 2-D EVōC projection of the chunks table
// (built offline by `raudio feature atlas`). `status` gates the view, `points`
// streams compact coord/colour/key arrays for the scatter renderer, and
// `chunk` lazily fetches one chunk's full detail when a point is selected.

/** The three projection spaces the atlas can be built on. `text` = transcript
 *  semantics (`text_embedding` → atlas_*); `visual` = the per-chunk frame image
 *  vector (`frame_embedding` → atlas_img_*); `caption` = the frame's Swedish
 *  caption embedding (`caption_embedding` → atlas_cap_*). */
export type AtlasSpace = 'text' | 'visual' | 'caption';

const AtlasStatusSchema = z.object({
  projected: z.boolean(),
  rows: z.number().int(),
  space: z.string().optional(),
  // Which spaces are built — gates the Text/Visual/Caption toggle.
  spaces: z.object({ text: z.boolean(), visual: z.boolean(), caption: z.boolean() }).optional(),
});
export type AtlasStatus = z.infer<typeof AtlasStatusSchema>;

export async function getAtlasStatus(
  space: AtlasSpace = 'text',
  fetcher: typeof fetch = fetch,
): Promise<AtlasStatus> {
  return asJson(await fetcher(`/api/atlas/status?space=${space}`), AtlasStatusSchema);
}

/** Compact arrays for the scatter map. `doc[i]` indexes into `docs` (the distinct
 *  doc ids); `(docs[doc[i]], speech_id[i], chunk_id[i])` is point i's chunk key.
 *  `cluster`/`language`/`namn` are per-space colour/label codes; `namn[i]`
 *  indexes into `namns` (the distinct archival names) for the hover popup.
 *
 *  The payload is an Apache Arrow IPC stream (see `getAtlasPoints`): `x`/`y` and
 *  the keys arrive as typed arrays (kept typed — `ArrayLike<number>` — to avoid
 *  a million-element copy), while the factorized colour *codes* are decoded to
 *  plain `number[]` because they feed APIs typed `readonly number[]`. */
export interface AtlasPoints {
  count: number;
  space?: AtlasSpace;
  // Float32Array from Arrow — only ever consumed via `Float32Array.from(...)`.
  x: ArrayLike<number>;
  y: ArrayLike<number>;
  docs: string[];
  doc: number[];
  /** Readable filename stem per distinct doc (aligned with `docs`) — video labels. */
  docFiles?: string[];
  speech_id: number[];
  chunk_id: number[];
  /** Stable Lance row address per point — sent back to /chunks for an
   *  O(selection) take when listing a lasso/legend selection. */
  rowid?: ArrayLike<number>;
  cluster?: number[];
  language?: number[];
  languages?: string[];
  namn?: number[];
  namns?: string[];
  /** Chunk broad topic (`topic_l2`) factorized: `topic[i]` indexes `topics`.
   *  Empty label ('') = unclustered/noise. */
  topic?: number[];
  topics?: string[];
  /** Per-video topic (`doc_topic`) factorized: `doc_topic[i]` indexes `doc_topics`. */
  doc_topic?: number[];
  doc_topics?: string[];
}

/** A factorized (codes, labels) pair pulled from one Arrow DICTIONARY column.
 *  `codes` are the per-point dictionary indices; `labels` the distinct values. */
interface DictColumn {
  codes: number[];
  labels: string[];
}

/** Extract the integer indices (codes) + dictionary values (labels) of an Arrow
 *  DICTIONARY vector.
 *
 *  WHY THIS API: a Dictionary `Vector.toArray()` returns the *decoded* values
 *  (e.g. `['sv','en','sv']`), not the indices — useless for the per-point colour
 *  codes. The indices live on the underlying `Data` as `.values` (an Int32Array,
 *  since the backend ships `dictionary<int32, utf8>`), and the label list is the
 *  attached `.dictionary` vector. A Vector may be chunked, so we concat the
 *  per-chunk index arrays; the dictionary is shared across chunks, so the first
 *  one wins. Verified against apache-arrow 21.x (`d.values` / `d.dictionary`). */
function dictColumn(vec: Vector | null): DictColumn | null {
  if (!vec) return null;
  const codes: number[] = [];
  let labels: string[] = [];
  for (const d of vec.data) {
    for (const code of d.values) codes.push(code);
    if (d.dictionary && labels.length === 0) labels = d.dictionary.toArray() as string[];
  }
  return { codes, labels };
}

/** A typed numeric column, kept as the underlying typed array (no copy). */
function typedColumn(vec: Vector | null): ArrayLike<number> {
  return (vec?.toArray() ?? []) as ArrayLike<number>;
}

/** A numeric key column decoded to a plain `number[]` (Arrow int64 columns come
 *  back as BigInt64Array, and several consumers want `readonly number[]`). */
function numberColumn(vec: Vector | null): number[] {
  if (!vec) return [];
  const out: number[] = [];
  for (const v of vec) out.push(Number(v));
  return out;
}

/** Fetch the point arrays for a space as a single Apache Arrow IPC stream
 *  (binary, parse-free — replaces a ~10 MB JSON body). A structural guard is
 *  enough here (no per-element zod) and keeps the map snappy. */
export async function getAtlasPoints(
  space: AtlasSpace = 'text',
  fetcher: typeof fetch = fetch,
): Promise<AtlasPoints> {
  // `v` busts any HTTP cache (the response sets max-age=300) left by a build
  // whose payload shape differed — notably entries from before `rowid` was
  // added, which would silently break the selection table. Bump when the
  // points payload shape changes; the backend ignores the extra param.
  // v=5: switched the wire format from JSON to an Arrow IPC stream.
  const r = await fetcher(`/api/atlas/points?space=${space}&v=5`);
  if (!r.ok) {
    const body = await r.json().catch(() => ({}));
    throw new ApiError(r.status, body?.detail ?? r.statusText);
  }

  const buf = await r.arrayBuffer();
  const table = tableFromIPC(new Uint8Array(buf));
  const md = table.schema.metadata;

  const count = Number(md.get('count') ?? table.numRows);
  const spaceMeta = md.get('space');
  const docFilesMeta = md.get('docFiles');

  const doc = dictColumn(table.getChild('doc'));
  if (!doc) throw new ApiError(500, 'malformed /api/atlas/points payload (no doc column)');

  const data: AtlasPoints = {
    count,
    docs: doc.labels,
    doc: doc.codes,
    speech_id: numberColumn(table.getChild('speech_id')),
    chunk_id: numberColumn(table.getChild('chunk_id')),
    x: typedColumn(table.getChild('x')),
    y: typedColumn(table.getChild('y')),
  };

  if (spaceMeta === 'text' || spaceMeta === 'visual' || spaceMeta === 'caption')
    data.space = spaceMeta;
  // `docFiles` rides in metadata (one entry per distinct doc, not per point), so
  // it can't be a table column; it's JSON, aligned with the `doc` dictionary.
  if (docFilesMeta) data.docFiles = JSON.parse(docFilesMeta) as string[];

  const rowid = table.getChild('rowid');
  if (rowid) data.rowid = numberColumn(rowid); // int64 → number[] (BigInt otherwise)
  const cluster = table.getChild('cluster');
  if (cluster) data.cluster = numberColumn(cluster); // iterated + fed to readonly number[]

  const language = dictColumn(table.getChild('language'));
  if (language) {
    data.language = language.codes;
    data.languages = language.labels;
  }
  const namn = dictColumn(table.getChild('namn'));
  if (namn) {
    data.namn = namn.codes;
    data.namns = namn.labels;
  }
  const topic = dictColumn(table.getChild('topic'));
  if (topic) {
    data.topic = topic.codes;
    data.topics = topic.labels;
  }
  const docTopic = dictColumn(table.getChild('doc_topic'));
  if (docTopic) {
    data.doc_topic = docTopic.codes;
    data.doc_topics = docTopic.labels;
  }

  if (typeof data.count !== 'number' || data.x.length !== data.count) {
    throw new ApiError(500, 'malformed /api/atlas/points payload');
  }
  return data;
}

/** Full hit for one chunk (detail pane + playback), looked up by its key. */
export async function getAtlasChunk(
  doc_id: string,
  speech_id: number,
  chunk_id: number,
  fetcher: typeof fetch = fetch,
): Promise<Hit> {
  const r = await fetcher(
    `/api/atlas/chunk/${encodeURIComponent(doc_id)}/${speech_id}/${chunk_id}`,
  );
  return asJson(r, HitSchema);
}

/** Full hits for a selection, addressed by stable Lance `_rowid` (from
 *  `AtlasPoints.rowid`). The backend resolves these with a single `_rowid IN`
 *  take — no per-key full-table scan. Drives the lasso/box selection table. */
export async function getAtlasChunks(
  rowids: number[],
  fetcher: typeof fetch = fetch,
): Promise<Hit[]> {
  const r = await fetcher('/api/atlas/chunks', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ rowids }),
  });
  return asJson(r, z.array(HitSchema));
}

// ── Topics (Tree page) ────────────────────────────────────────────────────
// `raudio feature topics` clusters chunks (Toponymy) into a nested topic
// hierarchy and stores it as Lance JSONB in `topics.lance`. `getTopics` reads
// that one tree; the LayerChart <Treemap> renders it, and clicking a node sends
// `topic=` to /api/search (matched against every topic_l* layer column).

/** One node of the topic tree: a leaf carries `value` (chunk count), a branch
 *  carries `children`. Mirrors the shape `build_topic_tree` writes. */
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

export const TopicsResponseSchema = z.object({
  built: z.boolean(),
  layers: z.number().int(),
  n_chunks: z.number().int(),
  hierarchy: TopicNodeSchema.nullable(),
  // The bucket name the backend uses for unclustered chunks (source of truth in
  // topic_tree.py:NOISE_LABEL) — the treemap reads it instead of hardcoding it.
  // Optional so an older backend (pre-`noise_label`) degrades (noise shown as a
  // normal topic) rather than hard-failing the whole page on a zod mismatch.
  noise_label: z.string().optional(),
});
export type TopicsResponse = z.infer<typeof TopicsResponseSchema>;

/** The topic hierarchy for the Tree treemap (`built: false` if not generated). */
export async function getTopics(fetcher: typeof fetch = fetch): Promise<TopicsResponse> {
  return asJson(await fetcher('/api/topics'), TopicsResponseSchema);
}
