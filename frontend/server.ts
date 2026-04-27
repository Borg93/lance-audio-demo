/**
 * Bun static server + reverse proxy for /api/* → FastAPI backend.
 *
 *   bun install
 *   bun run server.ts --api http://127.0.0.1:8000 --port 3000
 *
 * The Python backend owns Lance. This process only:
 *   - serves index.html / gallery.html / static assets
 *   - forwards /api/* requests (including HTTP Range for video streaming)
 *     to the backend, preserving headers both ways.
 */

import { readFileSync, existsSync } from "node:fs";
import { resolve, dirname, join, extname } from "node:path";
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

function serveStatic(relativePath: string): Response {
    const safe = relativePath.replace(/^\/+/, "").replace(/\.\./g, "");
    const abs = resolve(here, safe);
    if (!abs.startsWith(here)) return new Response("forbidden", { status: 403 });
    if (!existsSync(abs)) return new Response("not found", { status: 404 });
    return new Response(Bun.file(abs), {
        headers: { "Content-Type": contentType(abs) },
    });
}

// ─── /api/* proxy (streams requests + responses; Range headers flow through) ──
async function proxy(req: Request): Promise<Response> {
    const url = new URL(req.url);
    const target = `${API_BASE}${url.pathname}${url.search}`;

    // Strip headers that shouldn't be forwarded by a proxy.
    const headers = new Headers(req.headers);
    headers.delete("host");

    const upstream = await fetch(target, {
        method: req.method,
        headers,
        body: req.method === "GET" || req.method === "HEAD" ? undefined : req.body,
    });

    // Stream the response back. Range, Content-Length, Content-Type etc.
    // pass through unchanged.
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

        if (url.pathname === "/" || url.pathname === "/index.html") {
            return serveStatic("index.html");
        }
        if (url.pathname === "/gallery" || url.pathname === "/gallery.html") {
            return serveStatic("gallery.html");
        }

        // Any other path → static file under this dir.
        return serveStatic(url.pathname);
    },
});

console.log(`→ frontend:  http://localhost:${PORT}`);
console.log(`  proxying /api/*  →  ${API_BASE}`);
