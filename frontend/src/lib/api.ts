/**
 * Typed client for the FastAPI backend (`backend/app.py`).
 *
 * Schemas are zod-defined here and runtime-validated, so backend schema
 * drift surfaces as a clean error in the UI instead of silent rendering
 * bugs (the old plain-HTML frontend had several of those).
 */

import { z } from 'zod';

// ─────────────────────────────────────────────────────────────────────
// Schemas (mirror src/raudio/schema.py:CHUNK_SCHEMA + DOC_SCHEMA)
// ─────────────────────────────────────────────────────────────────────

export const SearchModeSchema = z.enum(['fts', 'semantic', 'visual', 'hybrid', 'all']);
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
    const r = await fetcher(`/api/search?${params}`);
    return asJson(r, HitsArraySchema);
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
