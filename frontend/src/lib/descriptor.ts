/**
 * Dataset descriptor — the schema-agnostic contract the whole UI renders from.
 *
 * The backend (`GET /api/datasets/{id}/descriptor`) merges *discovered* facts
 * (tables, column types, vector dims, indexes) with *declared* semantic roles
 * (identity, media binding, display, search wiring, atlas spaces,
 * capabilities). These zod schemas describe THAT envelope — never a corpus's
 * column names — so the same build renders any dataset.
 *
 * :class:`DatasetView` wraps a parsed descriptor with typed accessors
 * (`rowKey`, `title`, `body`, media URLs, search modes, …). Components read
 * rows through it instead of hardcoding field names.
 */

import { z } from 'zod';

// ─────────────────────────────────────────────────────────────────────
// Descriptor envelope (mirrors backend/lancekit/descriptor.py)
// ─────────────────────────────────────────────────────────────────────

const ColumnInfoSchema = z.object({
  name: z.string(),
  arrow_type: z.string(),
  nullable: z.boolean(),
  vector_dim: z.number().int().nullable().default(null),
  is_blob: z.boolean().default(false),
});
export type ColumnInfo = z.infer<typeof ColumnInfoSchema>;

const TableInfoSchema = z.object({
  name: z.string(),
  row_count: z.number().int(),
  version: z.number().int(),
  columns: z.array(ColumnInfoSchema),
  indexes: z.array(z.object({ name: z.string(), index_type: z.string(), columns: z.array(z.string()) })),
});
export type TableInfo = z.infer<typeof TableInfoSchema>;

const IdentitySchema = z.object({
  key_fields: z.array(z.string()).min(1),
  doc_key: z.string().default('doc_id'),
  doc_key_pattern: z.string().default('.*'),
});

const DocumentBindingSchema = z.object({
  table: z.string(),
  media_blob: z.string(),
  mime: z.string().nullable().default(null),
  thumbnail: z.string().nullable().default(null),
  thumbnail_mime: z.string().nullable().default(null),
});

const TimeBindingSchema = z.object({ start: z.string(), end: z.string() });

const MetadataFieldSchema = z.object({ field: z.string(), label: z.string() });

const DisplaySchema = z.object({
  title: z.array(z.string()).default([]),
  body: z.string().nullable().default(null),
  caption: z.string().nullable().default(null),
  metadata: z.array(MetadataFieldSchema).default([]),
});

const VectorBindingSchema = z.object({
  table: z.string(),
  column: z.string(),
  dim: z.number().int(),
  query_encoder: z.string(),
  caption_source: z.string().nullable().default(null),
});

const SearchSchema = z.object({
  row_table: z.string(),
  fts: z.object({ table: z.string(), column: z.string(), language: z.string().default('English') }).nullable().default(null),
  vectors: z.record(z.string(), VectorBindingSchema).default({}),
  filterable: z.array(z.string()).default([]),
  rerank: z.boolean().default(false),
});

const AtlasChannelSchema = z.object({
  name: z.string(),
  column: z.string().nullable().default(null),
  broadest_prefix: z.string().nullable().default(null),
});

const AtlasSpaceSchema = z.object({
  name: z.string(),
  x: z.string(),
  y: z.string(),
  cluster: z.string(),
  source_column: z.string(),
  table: z.string(),
  channels: z.array(AtlasChannelSchema).default([]),
});
export type AtlasSpaceDecl = z.infer<typeof AtlasSpaceSchema>;

const DeclaredSchema = z
  .object({
    identity: IdentitySchema,
    document: DocumentBindingSchema.nullable().default(null),
    time: TimeBindingSchema.nullable().default(null),
    display: DisplaySchema.default({}),
    search: SearchSchema.nullable().default(null),
    atlas: z.array(AtlasSpaceSchema).default([]),
    capabilities: z.record(z.string(), z.string()).default({}),
  })
  .passthrough();

export const DatasetDescriptorSchema = z.object({
  id: z.string(),
  tables: z.record(z.string(), TableInfoSchema),
  declared: DeclaredSchema,
});
export type DatasetDescriptor = z.infer<typeof DatasetDescriptorSchema>;

export const DatasetSummarySchema = z.object({
  id: z.string(),
  tables: z.record(z.string(), z.object({ row_count: z.number().int(), version: z.number().int() })).default({}),
  capabilities: z.array(z.string()).default([]),
});
export const DatasetsResponseSchema = z.object({ datasets: z.array(DatasetSummarySchema) });

// ─────────────────────────────────────────────────────────────────────
// Row envelope — the ranking + alignments contract, NOT corpus columns
// ─────────────────────────────────────────────────────────────────────

const WordSchema = z.object({
  text: z.string(),
  start: z.number(),
  end: z.number(),
  score: z.number().optional(),
});
export type Word = z.infer<typeof WordSchema>;

export const AlignmentSchema = z.object({
  start: z.number(),
  end: z.number(),
  text: z.string(),
  duration: z.number().optional(),
  score: z.number().optional(),
  words: z.array(WordSchema).optional(),
});
export type Alignment = z.infer<typeof AlignmentSchema>;

/** A search/browse result row. The parse validates the ENVELOPE — ranking
 *  signals + the alignments blob — and passes every other (corpus-specific)
 *  column through untouched, so the schema names no corpus column. Typed
 *  field access goes through {@link DatasetView}. */
export const RowSchema = z
  .object({
    _score: z.number().optional(),
    _distance: z.number().optional(),
    _relevance_score: z.number().optional(),
    alignments: z.array(AlignmentSchema).optional(),
    tags: z.array(z.string()).optional(),
  })
  .passthrough();
export type Row = z.infer<typeof RowSchema> & Record<string, unknown>;

/** The search modes a dataset supports, derived from its declared bindings. */
export type SearchMode = 'fts' | 'semantic' | 'visual' | 'scene' | 'scene_fts' | 'hybrid' | 'all';

const COSINE_DISTANCE_MAX = 2;

// ─────────────────────────────────────────────────────────────────────
// DatasetView — typed accessors over a parsed descriptor
// ─────────────────────────────────────────────────────────────────────

function str(value: unknown): string | null {
  return value === null || value === undefined ? null : String(value);
}

function num(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

export class DatasetView {
  constructor(readonly descriptor: DatasetDescriptor) {}

  get id(): string {
    return this.descriptor.id;
  }

  private get declared() {
    return this.descriptor.declared;
  }

  // ── identity ──────────────────────────────────────────────────────
  get keyFields(): string[] {
    return this.declared.identity.key_fields;
  }

  get docKeyField(): string {
    return this.declared.identity.doc_key;
  }

  /** Stable per-row identity key (the descriptor's key fields joined). */
  rowKey(row: Row): string {
    return this.keyFields.map((k) => String(row[k] ?? '')).join('|');
  }

  docId(row: Row): string {
    return String(row[this.docKeyField] ?? '');
  }

  /** The path segments a media route takes (doc key first, then the rest). */
  keyPath(row: Row): string[] {
    return this.keyFields.map((k) => encodeURIComponent(String(row[k] ?? '')));
  }

  // ── display ───────────────────────────────────────────────────────
  /** First non-empty declared title field, else the doc id. */
  title(row: Row): string {
    for (const field of this.declared.display.title) {
      const value = str(row[field]);
      if (value) return value;
    }
    return this.docId(row);
  }

  get bodyField(): string | null {
    return this.declared.display.body;
  }

  body(row: Row): string {
    return this.bodyField ? (str(row[this.bodyField]) ?? '') : '';
  }

  get captionField(): string | null {
    return this.declared.display.caption;
  }

  caption(row: Row): string | null {
    return this.captionField ? str(row[this.captionField]) : null;
  }

  /** Declared metadata fields (field + human label) for cards / meta panels. */
  get metadataFields(): { field: string; label: string }[] {
    return this.declared.display.metadata;
  }

  /** Resolved metadata rows for one hit (absent/empty values dropped). */
  metadata(row: Row): { field: string; label: string; value: string }[] {
    const out: { field: string; label: string; value: string }[] = [];
    for (const { field, label } of this.metadataFields) {
      const value = str(row[field]);
      if (value) out.push({ field, label, value });
    }
    return out;
  }

  // ── time ──────────────────────────────────────────────────────────
  get hasTime(): boolean {
    return this.declared.time !== null;
  }

  time(row: Row): { start: number; end: number } | null {
    const t = this.declared.time;
    if (!t) return null;
    const start = num(row[t.start]);
    const end = num(row[t.end]);
    return start === null || end === null ? null : { start, end };
  }

  duration(row: Row): number | null {
    const d = num(row['duration']);
    if (d !== null) return d;
    const t = this.time(row);
    return t ? t.end - t.start : null;
  }

  // ── search ────────────────────────────────────────────────────────
  get filterFields(): string[] {
    return this.declared.search?.filterable ?? [];
  }

  get hasFts(): boolean {
    return this.declared.search?.fts != null;
  }

  get rerankable(): boolean {
    return this.declared.search?.rerank ?? false;
  }

  /** The modes the search bar should offer, from declared bindings. */
  get searchModes(): SearchMode[] {
    const search = this.declared.search;
    if (!search) return [];
    const modes: SearchMode[] = [];
    if (search.fts != null) modes.push('fts');
    const vectors = search.vectors;
    if ('semantic' in vectors) modes.push('semantic');
    if ('visual' in vectors) modes.push('visual');
    if ('scene' in vectors) {
      modes.push('scene');
      if (vectors['scene']?.caption_source) modes.push('scene_fts');
    }
    if (search.fts != null && 'semantic' in vectors) modes.push('hybrid');
    if (modes.length > 1) modes.push('all');
    return modes;
  }

  hasMode(mode: SearchMode): boolean {
    return this.searchModes.includes(mode);
  }

  /** One comparable relevance number (higher = better), or null if unranked. */
  relevanceOf(row: Row, mode?: SearchMode): number | null {
    switch (mode) {
      case 'fts':
      case 'scene_fts':
        return row._score ?? null;
      case 'semantic':
      case 'visual':
        return row._distance != null ? COSINE_DISTANCE_MAX - row._distance : null;
      case 'hybrid':
        return row._relevance_score ?? null;
      case 'scene':
        return null;
      default:
        if (row._relevance_score != null) return row._relevance_score;
        if (row._score != null) return row._score;
        if (row._distance != null) return COSINE_DISTANCE_MAX - row._distance;
        return null;
    }
  }

  // ── media ─────────────────────────────────────────────────────────
  get hasMedia(): boolean {
    return this.declared.document !== null;
  }

  get hasThumbnail(): boolean {
    return this.declared.document?.thumbnail != null;
  }

  mediaUrl(row: Row): string {
    return `/api/media/${encodeURIComponent(this.docId(row))}${this.datasetQuery('?')}`;
  }

  thumbnailUrl(row: Row): string {
    return `/api/thumbnail/${encodeURIComponent(this.docId(row))}${this.datasetQuery('?')}`;
  }

  /** Per-row frame image (route arity = identity key fields). */
  frameUrl(row: Row): string {
    return `/api/chunk-frame/${this.keyPath(row).join('/')}${this.datasetQuery('?')}`;
  }

  // ── capabilities ──────────────────────────────────────────────────
  hasCapability(name: string): boolean {
    return name in this.declared.capabilities;
  }

  // ── atlas ─────────────────────────────────────────────────────────
  get atlasSpaces(): AtlasSpaceDecl[] {
    return this.declared.atlas;
  }

  atlasSpace(name: string): AtlasSpaceDecl | null {
    return this.atlasSpaces.find((s) => s.name === name) ?? null;
  }

  /** The categorical channel output-names an atlas space ships (for legends). */
  atlasChannels(name: string): string[] {
    return this.atlasSpace(name)?.channels.map((c) => c.name) ?? [];
  }

  // ── dataset selector (non-default datasets ride a `?dataset=` param) ──
  private isDefault = true;

  /** Mark this view as a non-default dataset so its URLs carry `?dataset=`. */
  withDatasetParam(isDefault: boolean): this {
    this.isDefault = isDefault;
    return this;
  }

  private datasetQuery(prefix: '?' | '&'): string {
    return this.isDefault ? '' : `${prefix}dataset=${encodeURIComponent(this.id)}`;
  }

  datasetParam(): string | null {
    return this.isDefault ? null : this.id;
  }
}
