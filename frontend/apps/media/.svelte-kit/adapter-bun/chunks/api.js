import * as v from "valibot";
const int = () => v.pipe(v.number(), v.integer());
const ColumnInfoSchema = v.object({
  name: v.string(),
  arrow_type: v.string(),
  nullable: v.boolean(),
  vector_dim: v.optional(v.nullable(int()), null),
  is_blob: v.optional(v.boolean(), false)
});
const TableInfoSchema = v.object({
  name: v.string(),
  row_count: int(),
  version: int(),
  columns: v.array(ColumnInfoSchema),
  indexes: v.array(
    v.object({ name: v.string(), index_type: v.string(), columns: v.array(v.string()) })
  )
});
const IdentitySchema = v.object({
  key_fields: v.pipe(v.array(v.string()), v.minLength(1)),
  doc_key: v.optional(v.string(), "doc_id"),
  doc_key_pattern: v.optional(v.string(), ".*")
});
const DocumentBindingSchema = v.object({
  table: v.string(),
  media_blob: v.string(),
  mime: v.optional(v.nullable(v.string()), null),
  thumbnail: v.optional(v.nullable(v.string()), null),
  thumbnail_mime: v.optional(v.nullable(v.string()), null)
});
const TimeBindingSchema = v.object({ start: v.string(), end: v.string() });
const MetadataFieldSchema = v.object({ field: v.string(), label: v.string() });
const DisplaySchema = v.object({
  title: v.optional(v.array(v.string()), []),
  body: v.optional(v.nullable(v.string()), null),
  caption: v.optional(v.nullable(v.string()), null),
  metadata: v.optional(v.array(MetadataFieldSchema), [])
});
const VectorBindingSchema = v.object({
  table: v.string(),
  column: v.string(),
  dim: int(),
  query_encoder: v.string(),
  caption_source: v.optional(v.nullable(v.string()), null)
});
const SearchSchema = v.object({
  row_table: v.string(),
  fts: v.optional(
    v.nullable(
      v.object({
        table: v.string(),
        column: v.string(),
        language: v.optional(v.string(), "English")
      })
    ),
    null
  ),
  vectors: v.optional(v.record(v.string(), VectorBindingSchema), {}),
  filterable: v.optional(v.array(v.string()), []),
  rerank: v.optional(v.boolean(), false)
});
const AtlasChannelSchema = v.object({
  name: v.string(),
  column: v.optional(v.nullable(v.string()), null),
  broadest_prefix: v.optional(v.nullable(v.string()), null)
});
const AtlasSpaceSchema = v.object({
  name: v.string(),
  x: v.string(),
  y: v.string(),
  cluster: v.string(),
  source_column: v.string(),
  table: v.string(),
  channels: v.optional(v.array(AtlasChannelSchema), [])
});
const DeclaredSchema = v.looseObject({
  identity: IdentitySchema,
  document: v.optional(v.nullable(DocumentBindingSchema), null),
  time: v.optional(v.nullable(TimeBindingSchema), null),
  display: v.optional(DisplaySchema, {}),
  search: v.optional(v.nullable(SearchSchema), null),
  atlas: v.optional(v.array(AtlasSpaceSchema), []),
  capabilities: v.optional(v.record(v.string(), v.string()), {})
});
const DatasetDescriptorSchema = v.object({
  id: v.string(),
  tables: v.record(v.string(), TableInfoSchema),
  declared: DeclaredSchema
});
const DatasetSummarySchema = v.object({
  id: v.string(),
  tables: v.optional(v.record(v.string(), v.object({ row_count: int(), version: int() })), {}),
  capabilities: v.optional(v.array(v.string()), [])
});
v.object({ datasets: v.array(DatasetSummarySchema) });
const WordSchema = v.object({
  text: v.string(),
  start: v.number(),
  end: v.number(),
  score: v.optional(v.number())
});
const AlignmentSchema = v.object({
  start: v.number(),
  end: v.number(),
  text: v.string(),
  duration: v.optional(v.number()),
  score: v.optional(v.number()),
  words: v.optional(v.array(WordSchema))
});
const RowSchema = v.looseObject({
  _score: v.optional(v.number()),
  _distance: v.optional(v.number()),
  _relevance_score: v.optional(v.number()),
  alignments: v.optional(v.array(AlignmentSchema)),
  tags: v.optional(v.array(v.string()))
});
const _NUMERIC_RE = /\b(u?int\d*|float\d*|double|decimal\d*|half_?float)\b/;
const _TEMPORAL_RE = /\b(timestamp|date\d*|time\d*|duration|interval)\b/;
function categoryOf(col, ftsColumns) {
  if (col.vector_dim != null) return "embedding";
  if (col.is_blob) return "blob";
  const t = col.arrow_type.toLowerCase();
  if (ftsColumns?.has(col.name) && (t.includes("string") || t.includes("utf8"))) return "text";
  if (t.includes("dictionary")) return "categorical";
  if (_NUMERIC_RE.test(t)) return "numerical";
  if (_TEMPORAL_RE.test(t)) return "temporal";
  if (t.includes("string") || t.includes("utf8") || t.includes("bool")) return "categorical";
  return "other";
}
const COSINE_DISTANCE_MAX$1 = 2;
function str(value) {
  return value === null || value === void 0 ? null : String(value);
}
function num(value) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}
class DatasetView {
  constructor(descriptor) {
    this.descriptor = descriptor;
  }
  descriptor;
  get id() {
    return this.descriptor.id;
  }
  get declared() {
    return this.descriptor.declared;
  }
  // ── identity ──────────────────────────────────────────────────────
  get keyFields() {
    return this.declared.identity.key_fields;
  }
  get docKeyField() {
    return this.declared.identity.doc_key;
  }
  /** Stable per-row identity key (the descriptor's key fields joined). */
  rowKey(row) {
    return this.keyFields.map((k) => String(row[k] ?? "")).join("|");
  }
  docId(row) {
    return String(row[this.docKeyField] ?? "");
  }
  /** The path segments a media route takes (doc key first, then the rest). */
  keyPath(row) {
    return this.keyFields.map((k) => encodeURIComponent(String(row[k] ?? "")));
  }
  // ── display ───────────────────────────────────────────────────────
  /** First non-empty declared title field, else the doc id. */
  title(row) {
    for (const field of this.declared.display.title) {
      const value = str(row[field]);
      if (value) return value;
    }
    return this.docId(row);
  }
  get bodyField() {
    return this.declared.display.body;
  }
  body(row) {
    return this.bodyField ? str(row[this.bodyField]) ?? "" : "";
  }
  get captionField() {
    return this.declared.display.caption;
  }
  caption(row) {
    return this.captionField ? str(row[this.captionField]) : null;
  }
  /** Declared metadata fields (field + human label) for cards / meta panels. */
  get metadataFields() {
    return this.declared.display.metadata;
  }
  /** Resolved metadata rows for one hit (absent/empty values dropped). */
  metadata(row) {
    const out = [];
    for (const { field, label } of this.metadataFields) {
      const value = str(row[field]);
      if (value) out.push({ field, label, value });
    }
    return out;
  }
  // ── time ──────────────────────────────────────────────────────────
  get hasTime() {
    return this.declared.time !== null;
  }
  time(row) {
    const t = this.declared.time;
    if (!t) return null;
    const start = num(row[t.start]);
    const end = num(row[t.end]);
    return start === null || end === null ? null : { start, end };
  }
  duration(row) {
    const d = num(row["duration"]);
    if (d !== null) return d;
    const t = this.time(row);
    return t ? t.end - t.start : null;
  }
  // ── search ────────────────────────────────────────────────────────
  get filterFields() {
    return this.declared.search?.filterable ?? [];
  }
  get hasFts() {
    return this.declared.search?.fts != null;
  }
  get rerankable() {
    return this.declared.search?.rerank ?? false;
  }
  /** String columns that carry an FTS index — the `text` (full-text) category. */
  get ftsColumns() {
    const fts = this.declared.search?.fts;
    return new Set(fts ? [fts.column] : []);
  }
  /** Category of one declared column, from its Lance type (null if unknown). */
  columnCategory(table, column) {
    const info = this.descriptor.tables[table]?.columns.find((c) => c.name === column);
    return info ? categoryOf(info, this.ftsColumns) : null;
  }
  /** Every declared embedding space, uniform — text/pixel/audio all alike. The
   *  only per-space behaviour (direct vs frame-join) rides on `onRowTable`. */
  get vectorSpaces() {
    const search2 = this.declared.search;
    if (!search2) return [];
    return Object.entries(search2.vectors).map(([key, b]) => ({
      key,
      table: b.table,
      column: b.column,
      dim: b.dim,
      encoder: b.query_encoder,
      captionSource: b.caption_source,
      onRowTable: b.table === search2.row_table
    }));
  }
  /** The modes the search bar should offer, derived GENERICALLY from the
   *  declared bindings: fts + one mode per embedding space (by its own key,
   *  whatever it is) + a `_fts` variant for spaces with a caption source +
   *  hybrid/all composites. A dataset that declares a new embedding key gets a
   *  new mode with no code change here. */
  get searchModes() {
    const search2 = this.declared.search;
    if (!search2) return [];
    const queryable = this.vectorSpaces.filter(
      (s) => s.encoder === "text" || s.encoder === "image"
    );
    const modes = [];
    if (search2.fts != null) modes.push("fts");
    for (const s of queryable) modes.push(s.key);
    for (const s of queryable) if (s.captionSource) modes.push(`${s.key}_fts`);
    const hasRowTextSpace = queryable.some((s) => s.encoder === "text" && s.onRowTable);
    if (search2.fts != null && hasRowTextSpace) modes.push("hybrid");
    if (modes.length > 1) modes.push("all");
    return modes;
  }
  hasMode(mode) {
    return this.searchModes.includes(mode);
  }
  /** One comparable relevance number (higher = better), or null if unranked. */
  relevanceOf(row, mode) {
    switch (mode) {
      case "fts":
      case "scene_fts":
        return row._score ?? null;
      case "semantic":
      case "visual":
        return row._distance != null ? COSINE_DISTANCE_MAX$1 - row._distance : null;
      case "hybrid":
        return row._relevance_score ?? null;
      case "scene":
        return null;
      default:
        if (row._relevance_score != null) return row._relevance_score;
        if (row._score != null) return row._score;
        if (row._distance != null) return COSINE_DISTANCE_MAX$1 - row._distance;
        return null;
    }
  }
  // ── media ─────────────────────────────────────────────────────────
  get hasMedia() {
    return this.declared.document !== null;
  }
  get hasThumbnail() {
    return this.declared.document?.thumbnail != null;
  }
  mediaUrl(row) {
    return `/api/media/${encodeURIComponent(this.docId(row))}${this.datasetQuery("?")}`;
  }
  thumbnailUrl(row) {
    return `/api/thumbnail/${encodeURIComponent(this.docId(row))}${this.datasetQuery("?")}`;
  }
  /** Per-row frame image (route arity = identity key fields). */
  frameUrl(row) {
    return `/api/chunk-frame/${this.keyPath(row).join("/")}${this.datasetQuery("?")}`;
  }
  // ── capabilities ──────────────────────────────────────────────────
  hasCapability(name) {
    return name in this.declared.capabilities;
  }
  // ── atlas ─────────────────────────────────────────────────────────
  get atlasSpaces() {
    return this.declared.atlas;
  }
  atlasSpace(name) {
    return this.atlasSpaces.find((s) => s.name === name) ?? null;
  }
  /** The categorical channel output-names an atlas space ships (for legends). */
  atlasChannels(name) {
    return this.atlasSpace(name)?.channels.map((c) => c.name) ?? [];
  }
  // ── dataset selector (non-default datasets ride a `?dataset=` param) ──
  isDefault = true;
  /** Mark this view as a non-default dataset so its URLs carry `?dataset=`. */
  withDatasetParam(isDefault) {
    this.isDefault = isDefault;
    return this;
  }
  datasetQuery(prefix) {
    return this.isDefault ? "" : `${prefix}dataset=${encodeURIComponent(this.id)}`;
  }
  datasetParam() {
    return this.isDefault ? null : this.id;
  }
}
let _active = null;
function setActiveView(view) {
  _active = view;
}
function activeView() {
  if (_active === null) throw new Error("dataset descriptor not loaded");
  return _active;
}
const COSINE_DISTANCE_MAX = 2;
function relevanceOf(hit, mode) {
  switch (mode) {
    case "fts":
    case "scene_fts":
      return hit._score ?? null;
    case "semantic":
    case "visual":
      return hit._distance != null ? COSINE_DISTANCE_MAX - hit._distance : null;
    case "hybrid":
      return hit._relevance_score ?? null;
    case "scene":
      return null;
    default:
      if (hit._relevance_score != null) return hit._relevance_score;
      if (hit._score != null) return hit._score;
      if (hit._distance != null) return COSINE_DISTANCE_MAX - hit._distance;
      return null;
  }
}
class ApiError extends Error {
  constructor(status, detail) {
    super(`api ${status}: ${detail}`);
    this.status = status;
    this.detail = detail;
    this.name = "ApiError";
  }
  status;
  detail;
}
const ProblemSchema = v.object({ detail: v.optional(v.string()), title: v.optional(v.string()) });
async function apiErrorFrom(r) {
  const body = await r.json().catch(() => null);
  const parsed = v.safeParse(ProblemSchema, body);
  const detail = (parsed.success ? parsed.output.detail ?? parsed.output.title : void 0) || r.statusText || `HTTP ${r.status}`;
  return new ApiError(r.status, detail);
}
async function asJson(r, schema) {
  if (!r.ok) throw await apiErrorFrom(r);
  return v.parse(schema, await r.json());
}
const HitsArraySchema = v.array(RowSchema);
function datasetParam(spec) {
  if (spec?.dataset) return spec.dataset;
  return activeView().datasetParam();
}
function appendCommonSearchParams(out, spec) {
  if (spec.rerank) out.append("rerank", "true");
  if (spec.rerank && spec.rerankN !== void 0) out.append("rerank_n", String(spec.rerankN));
  if (spec.weight !== void 0) out.append("weight", String(spec.weight));
  if (spec.qVec) out.append("q_vec", spec.qVec);
  if (spec.where) out.append("where", spec.where);
  if (spec.prefilter === false) out.append("prefilter", "false");
  for (const [field, value] of Object.entries(spec.filters ?? {})) {
    if (value) out.append(field, value);
  }
  if (spec.topic) out.append("topic", spec.topic);
  const ds = datasetParam(spec);
  if (ds) out.append("dataset", ds);
}
async function search(spec, fetcher = fetch) {
  const n = String(spec.n ?? 30);
  const mode = spec.mode ?? "fts";
  if (spec.image) {
    const fd = new FormData();
    fd.append("image", spec.image);
    if (spec.q) fd.append("q", spec.q);
    fd.append("n", n);
    fd.append("mode", mode);
    appendCommonSearchParams(fd, spec);
    const r2 = await fetcher("/api/search", { method: "POST", body: fd });
    return asJson(r2, HitsArraySchema);
  }
  const params = new URLSearchParams({ q: spec.q, n, mode });
  if (spec.fuzziness) params.append("fuzziness", String(spec.fuzziness));
  if (spec.phrase) params.append("phrase", "true");
  appendCommonSearchParams(params, spec);
  const r = await fetcher(`/api/search?${params}`);
  return asJson(r, HitsArraySchema);
}
function datasetSuffix() {
  const ds = activeView().datasetParam();
  return ds ? `?dataset=${encodeURIComponent(ds)}` : "";
}
v.object({ alignments: v.array(AlignmentSchema) });
const DocTranscriptChunkSchema = RowSchema;
v.object({
  doc_id: v.string(),
  chunks: v.array(DocTranscriptChunkSchema)
});
const DiarTurnSchema = v.object({
  turn_id: v.pipe(v.number(), v.integer()),
  speaker: v.string(),
  start: v.number(),
  end: v.number()
});
v.object({
  built: v.boolean(),
  doc_id: v.string(),
  turns: v.array(DiarTurnSchema),
  speakers: v.array(v.string())
});
const PingSchema = v.object({ ok: v.boolean(), url: v.string(), error: v.nullish(v.string()) });
const HealthSchema = v.object({
  db: v.object({
    path: v.string(),
    tables: v.array(v.string()),
    chunks: v.number(),
    documents: v.number()
  }),
  embed: PingSchema,
  rerank: PingSchema
});
async function getHealth(fetcher = fetch) {
  const r = await fetcher("/api/health");
  return asJson(r, HealthSchema);
}
const DocumentSchema = RowSchema;
v.object({
  total: v.pipe(v.number(), v.integer()),
  page: v.pipe(v.number(), v.integer()),
  docs: v.array(DocumentSchema)
});
v.object({ name: v.string(), type: v.string() });
async function getDatasetView(datasetId, isDefault, fetcher = fetch) {
  const r = await fetcher(`/api/datasets/${encodeURIComponent(datasetId)}/descriptor`);
  if (!r.ok) throw await apiErrorFrom(r);
  const descriptor = v.parse(DatasetDescriptorSchema, await r.json());
  return new DatasetView(descriptor).withDatasetParam(isDefault);
}
const thumbnailUrl = (row) => activeView().thumbnailUrl(row);
const chunkFrameUrl = (row) => activeView().frameUrl(row);
const mediaUrl = (row) => activeView().mediaUrl(row);
v.object({
  projected: v.boolean(),
  rows: v.pipe(v.number(), v.integer()),
  space: v.optional(v.string()),
  // Which named spaces are built (gates the space toggle).
  spaces: v.optional(v.record(v.string(), v.boolean()))
});
async function getAtlasChunks(rowids, fetcher = fetch) {
  const ds = activeView().datasetParam();
  const url = ds ? `/api/atlas/chunks?dataset=${encodeURIComponent(ds)}` : "/api/atlas/chunks";
  const r = await fetcher(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ rowids })
  });
  return asJson(r, HitsArraySchema);
}
const TopicNodeSchema = v.lazy(
  () => v.object({
    name: v.string(),
    value: v.optional(v.number()),
    children: v.optional(v.array(TopicNodeSchema))
  })
);
v.object({
  built: v.boolean(),
  layers: v.pipe(v.number(), v.integer()),
  n_chunks: v.pipe(v.number(), v.integer()),
  hierarchy: v.nullable(TopicNodeSchema),
  noise_label: v.optional(v.string())
});
v.object({
  built: v.boolean(),
  entities: v.pipe(v.number(), v.integer()),
  relations: v.pipe(v.number(), v.integer()),
  mentions: v.pipe(v.number(), v.integer()),
  videos: v.pipe(v.number(), v.integer())
});
const CypherValueSchema = v.union([v.string(), v.number(), v.null_()]);
const GraphCypherResponseSchema = v.object({
  built: v.boolean(),
  columns: v.array(v.string()),
  rows: v.array(v.array(CypherValueSchema)),
  error: v.nullable(v.string())
});
async function runGraphCypher(query, limit = 200, fetcher = fetch) {
  const ds = activeView().datasetParam();
  const url = ds ? `/api/graph/cypher?dataset=${encodeURIComponent(ds)}` : "/api/graph/cypher";
  const r = await fetcher(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, limit })
  });
  return asJson(r, GraphCypherResponseSchema);
}
const GraphMatchSchema = v.object({
  entity_id: v.string(),
  name: v.string(),
  entity_type: v.string(),
  mention_count: v.pipe(v.number(), v.integer()),
  videos: v.pipe(v.number(), v.integer())
});
v.object({
  built: v.boolean(),
  matches: v.array(GraphMatchSchema)
});
const GraphEntitySchema = v.object({
  entity_id: v.string(),
  name: v.string(),
  entity_type: v.string(),
  mention_count: v.pipe(v.number(), v.integer())
});
const GraphClipSchema = v.object({
  chunk_id: v.string(),
  doc_id: v.string(),
  title: v.string(),
  start: v.number(),
  end: v.number(),
  text: v.string()
});
const GraphNeighborSchema = v.object({
  entity_id: v.string(),
  name: v.string(),
  entity_type: v.string(),
  direction: v.picklist(["out", "in"]),
  description: v.string()
});
const GraphCooccurSchema = v.object({
  entity_id: v.string(),
  name: v.string(),
  shared: v.pipe(v.number(), v.integer())
});
v.object({
  built: v.boolean(),
  entity: v.nullable(GraphEntitySchema),
  clips: v.array(GraphClipSchema),
  neighbors: v.array(GraphNeighborSchema),
  cooccur: v.array(GraphCooccurSchema)
});
const GraphNodeSchema = v.object({
  id: v.string(),
  name: v.string(),
  type: v.string(),
  mentions: v.pipe(v.number(), v.integer()),
  videos: v.pipe(v.number(), v.integer())
});
const GraphEdgeSchema = v.object({
  source: v.string(),
  target: v.string(),
  description: v.string()
});
const GraphSubgraphResponseSchema = v.object({
  built: v.boolean(),
  nodes: v.array(GraphNodeSchema),
  edges: v.array(GraphEdgeSchema)
});
async function getGraphSubgraph(entityId, limit = 150, fetcher = fetch) {
  const params = new URLSearchParams({ limit: String(limit) });
  const ds = activeView().datasetParam();
  if (ds) params.set("dataset", ds);
  return asJson(await fetcher(`/api/graph/subgraph?${params}`), GraphSubgraphResponseSchema);
}
const VoiceStatusSchema = v.object({
  built: v.boolean(),
  turns: v.pipe(v.number(), v.integer()),
  speakers: v.pipe(v.number(), v.integer())
});
async function getVoiceStatus(fetcher = fetch) {
  return asJson(await fetcher(`/api/voice/status${datasetSuffix()}`), VoiceStatusSchema);
}
const VoiceHitSchema = v.intersect([
  RowSchema,
  v.object({
    speaker_label: v.string(),
    turn_id: v.pipe(v.number(), v.integer()),
    turn_start: v.number(),
    turn_end: v.number(),
    _distance: v.number(),
    turn_score: v.number()
  })
]);
const VoiceQueryInfoSchema = v.object({
  doc_id: v.nullable(v.string()),
  speaker_label: v.nullable(v.string()),
  turn_id: v.nullable(v.pipe(v.number(), v.integer())),
  turn_start: v.nullable(v.number()),
  turn_end: v.nullable(v.number())
});
const VoiceSimilarResponseSchema = v.object({
  query: VoiceQueryInfoSchema,
  hits: v.array(VoiceHitSchema)
});
async function voiceSimilar(anchor, opts = {}, fetcher = fetch) {
  const params = new URLSearchParams({ doc_id: anchor.docId });
  if ("turnId" in anchor) params.set("turn_id", String(anchor.turnId));
  else if ("speaker" in anchor) params.set("speaker", anchor.speaker);
  else params.set("t", String(anchor.t));
  if (opts.n !== void 0) params.set("n", String(opts.n));
  if (opts.excludeSameDoc !== void 0)
    params.set("exclude_same_doc", String(opts.excludeSameDoc));
  const ds = activeView().datasetParam();
  if (ds) params.set("dataset", ds);
  return asJson(await fetcher(`/api/voice/similar?${params}`), VoiceSimilarResponseSchema);
}
async function voiceSimilarUpload(file, opts = {}, fetcher = fetch) {
  const params = new URLSearchParams();
  if (opts.n !== void 0) params.set("n", String(opts.n));
  const ds = activeView().datasetParam();
  if (ds) params.set("dataset", ds);
  const fd = new FormData();
  fd.append("file", file);
  const url = params.size > 0 ? `/api/voice/similar?${params}` : "/api/voice/similar";
  const r = await fetcher(url, { method: "POST", body: fd });
  return asJson(r, VoiceSimilarResponseSchema);
}
function voiceBandOf(turnScore) {
  if (turnScore >= 0.7) return "strong";
  if (turnScore >= 0.6) return "possible";
  return null;
}
function isVoiceHit(hit) {
  return "turn_score" in hit && "speaker_label" in hit;
}
export {
  ApiError as A,
  getDatasetView as a,
  getVoiceStatus as b,
  activeView as c,
  chunkFrameUrl as d,
  search as e,
  voiceSimilarUpload as f,
  getHealth as g,
  voiceSimilar as h,
  isVoiceHit as i,
  getAtlasChunks as j,
  getGraphSubgraph as k,
  runGraphCypher as l,
  mediaUrl as m,
  relevanceOf as r,
  setActiveView as s,
  thumbnailUrl as t,
  voiceBandOf as v
};
