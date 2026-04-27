import { type ClassValue, clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

/** shadcn-svelte's standard `cn` helper. */
export function cn(...inputs: ClassValue[]): string {
    return twMerge(clsx(inputs));
}

/** Format seconds as `H:MM:SS` (or `M:SS` under an hour). */
export function fmtTime(s: number): string {
    const total = Math.round(s);
    const h = Math.floor(total / 3600);
    const m = Math.floor((total % 3600) / 60);
    const sec = total % 60;
    const pad = (n: number) => String(n).padStart(2, '0');
    return h > 0 ? `${h}:${pad(m)}:${pad(sec)}` : `${m}:${pad(sec)}`;
}

/** Escape HTML special characters for safe innerHTML interpolation. */
export function escapeHtml(s: string | null | undefined): string {
    return String(s ?? '').replace(
        /[&<>]/g,
        (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' })[c] ?? c,
    );
}

/** Lowercased content words from a Tantivy-style query. */
export function queryTerms(q: string): string[] {
    const stop = new Set(['and', 'or', 'not', 'near']);
    // `\w` excludes Unicode letters (ö/å/ä) even with /u — must use \p{L}.
    return (q.match(/[\p{L}\p{N}_]+/gu) ?? []).map((t) => t.toLowerCase()).filter((t) => !stop.has(t));
}
