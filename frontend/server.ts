/**
 * Bun static server + reverse proxy for /api/* → FastAPI backend.
 *
 *   bun install
 *   bun run server.ts --root ./build --api http://127.0.0.1:8000 --port 3000
 *
 * The Python backend owns Lance. This process only:
 *   - serves the static SvelteKit build from --root
 *   - falls back to 200.html for unknown paths so client-side routing works
 *   - forwards /api/* requests (including HTTP Range for video streaming)
 *     to the backend, preserving headers both ways.
 */

import { existsSync } from "node:fs";
import { resolve, dirname, extname } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));

// ─── CLI args ────────────────────────────────────────────────────────────
const args = Object.fromEntries(
    process.argv
        .slice(2)
        .map((a, i, all) =>
            a.startsWith("--") ? [a.slice(2), all[i + 1]] : null,
        )
        .filter((x): x is [string, string] => x !== null),
);
const API_BASE = (args.api ?? "http://127.0.0.1:8000").replace(/\/$/, "");
const PORT = Number(args.port ?? 3000);
// Default to the SvelteKit build dir adjacent to this file.
const ROOT = resolve(here, args.root ?? "./build");
const SPA_FALLBACK = resolve(ROOT, "200.html");

// ─── MIME helper ────────────────────────────────────────────────────────
const MIME: Record<string, string> = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".ico": "image/x-icon",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
};

function contentType(path: string): string {
    return MIME[extname(path).toLowerCase()] ?? "application/octet-stream";
}

function fileResponse(abs: string): Response {
    return new Response(Bun.file(abs), {
        headers: { "Content-Type": contentType(abs) },
    });
}

function serveStatic(pathname: string): Response {
    // Prevent path traversal — confine resolved paths to ROOT.
    const safe = pathname.replace(/^\/+/, "").replace(/\.\./g, "");
    const abs = resolve(ROOT, safe);
    if (!abs.startsWith(ROOT)) {
        return new Response("forbidden", { status: 403 });
    }
    if (existsSync(abs)) return fileResponse(abs);

    // Try the prerendered route variant SvelteKit emits (e.g. /gallery → /gallery.html).
    const htmlVariant = `${abs.replace(/\/+$/, "")}.html`;
    if (existsSync(htmlVariant)) return fileResponse(htmlVariant);

    // SPA fallback for client-side routes.
    if (existsSync(SPA_FALLBACK)) return fileResponse(SPA_FALLBACK);

    return new Response("not found", { status: 404 });
}

// ─── /api/* proxy (streams requests + responses; Range headers flow through) ──
async function proxy(req: Request): Promise<Response> {
    const url = new URL(req.url);
    const target = `${API_BASE}${url.pathname}${url.search}`;

    const headers = new Headers(req.headers);
    headers.delete("host");

    const upstream = await fetch(target, {
        method: req.method,
        headers,
        body: req.method === "GET" || req.method === "HEAD" ? undefined : req.body,
    });

    return new Response(upstream.body, {
        status: upstream.status,
        headers: upstream.headers,
    });
}

// ─── Router ──────────────────────────────────────────────────────────────
Bun.serve({
    port: PORT,
    async fetch(req) {
        const url = new URL(req.url);

        if (url.pathname.startsWith("/api/")) return proxy(req);

        if (url.pathname === "/") return serveStatic("index.html");

        return serveStatic(url.pathname);
    },
});

console.log(`→ frontend:  http://localhost:${PORT}`);
console.log(`  serving:   ${ROOT}`);
console.log(`  proxying /api/*  →  ${API_BASE}`);
