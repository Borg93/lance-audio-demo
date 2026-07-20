import * as v from "valibot";
import { R as RowSchema, A as AlignmentSchema, a as activeView, D as DatasetDescriptorSchema, b as DatasetView } from "./descriptor.js";
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
  chunkFrameUrl as c,
  voiceSimilarUpload as d,
  voiceSimilar as e,
  getAtlasChunks as f,
  getHealth as g,
  getGraphSubgraph as h,
  isVoiceHit as i,
  runGraphCypher as j,
  mediaUrl as m,
  relevanceOf as r,
  search as s,
  thumbnailUrl as t,
  voiceBandOf as v
};
