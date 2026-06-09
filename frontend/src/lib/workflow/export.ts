/**
 * Export the workflow's result hits to a downloadable file (client-side only —
 * the hits already live in the browser, so no backend round-trip). An Export
 * node picks the format (JSON / CSV) and which columns to include; both formats
 * honour that column selection.
 */
import type { Hit } from '$lib/api';

/** The chunk-level columns an Export node can include (the per-word
 *  `alignments`/`words` arrays are intentionally excluded — a flat export wants
 *  scalar fields). `_score` is the rank score. This order IS the export order. */
export const EXPORT_COLUMNS = [
  'doc_id',
  'audio_path',
  'speech_id',
  'chunk_id',
  'start',
  'end',
  'duration',
  'text',
  'caption',
  'language',
  'namn',
  'referenskod',
  'bildid',
  'extraid',
  '_score',
  'tags',
] as const;

export type ExportColumn = (typeof EXPORT_COLUMNS)[number];

/** Keep only known columns, in canonical EXPORT_COLUMNS order — defends against
 *  stale persisted selections and guarantees a stable column order regardless
 *  of the order the user toggled them in. */
function orderColumns(columns: readonly string[]): string[] {
  const want = new Set(columns);
  return EXPORT_COLUMNS.filter((c) => want.has(c));
}

function csvCell(value: unknown): string {
  if (value == null) return '';
  // Arrays (e.g. `tags`) flatten to a `; `-joined cell so commas don't split it.
  const s = Array.isArray(value) ? value.join('; ') : String(value);
  // Quote + escape only when needed (comma, quote, or newline).
  return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

export function hitsToJson(
  hits: readonly Hit[],
  columns: readonly string[] = EXPORT_COLUMNS,
): string {
  const cols = orderColumns(columns);
  const rows = hits.map((h) => {
    const rec = h as Record<string, unknown>;
    const out: Record<string, unknown> = {};
    for (const c of cols) out[c] = rec[c] ?? null;
    return out;
  });
  return JSON.stringify(rows, null, 2);
}

export function hitsToCsv(
  hits: readonly Hit[],
  columns: readonly string[] = EXPORT_COLUMNS,
): string {
  const cols = orderColumns(columns);
  const header = cols.join(',');
  const rows = hits.map((h) =>
    cols.map((c) => csvCell((h as Record<string, unknown>)[c])).join(','),
  );
  return [header, ...rows].join('\n');
}

/** Trigger a browser download of `content` as `filename`. */
export function downloadFile(filename: string, content: string, mime: string): void {
  const url = URL.createObjectURL(new Blob([content], { type: mime }));
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

/** Download the result hits as JSON or CSV with a timestamped filename, limited
 *  to the chosen columns. */
export function exportHits(
  hits: readonly Hit[],
  format: 'json' | 'csv',
  now: Date,
  columns: readonly string[] = EXPORT_COLUMNS,
): void {
  if (hits.length === 0) return;
  const stamp = now.toISOString().slice(0, 19).replace(/[:T]/g, '-');
  if (format === 'json') {
    downloadFile(`raudio-results-${stamp}.json`, hitsToJson(hits, columns), 'application/json');
  } else {
    downloadFile(`raudio-results-${stamp}.csv`, hitsToCsv(hits, columns), 'text/csv;charset=utf-8');
  }
}
