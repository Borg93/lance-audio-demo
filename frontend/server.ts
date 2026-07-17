/**
 * Local production serve for the Bun-server build (svelte-adapter-bun): run the
 * generated app request-handler AND reverse-proxy /api/* → the FastAPI backend,
 * in ONE process. This mirrors the rask topology (the MFE's Bun server serves
 * the app; a gateway routes /api elsewhere) — collapsed here into one thin
 * server for local reproduction.
 *
 *   make frontend-build     # produces ./build (adapter-bun output)
 *   bun run server.ts --api http://127.0.0.1:8000 --port 5274
 *
 * The Python backend owns Lance; this process only serves the app + forwards
 * /api/* (including HTTP Range for video streaming), preserving headers.
 */

import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));

// ─── CLI args ────────────────────────────────────────────────────────────
const args = Object.fromEntries(
  process.argv
    .slice(2)
    .map((a, i, all) => (a.startsWith('--') ? [a.slice(2), all[i + 1]] : null))
    .filter((x): x is [string, string] => x !== null),
);
const API_BASE = (args.api ?? 'http://127.0.0.1:8000').replace(/\/$/, '');
const PORT = Number(args.port ?? 3000);

// ─── The adapter-bun build's request handler ───────────────────────────────
// Generated into ./build at build time; a dynamic import by absolute path keeps
// this file type-checkable/lintable without the build artifact present.
const { getHandler } = (await import(resolve(here, 'build/handler.js'))) as {
  getHandler: () => {
    fetch: (req: Request, server: unknown) => Response | Promise<Response>;
    websocket?: unknown;
  };
};
const app = getHandler();

// ─── /api/* proxy (streams requests + responses; Range headers flow through) ──
async function proxy(req: Request): Promise<Response> {
  const url = new URL(req.url);
  const headers = new Headers(req.headers);
  headers.delete('host');
  const upstream = await fetch(`${API_BASE}${url.pathname}${url.search}`, {
    method: req.method,
    headers,
    body: req.method === 'GET' || req.method === 'HEAD' ? undefined : req.body,
  });
  return new Response(upstream.body, { status: upstream.status, headers: upstream.headers });
}

// ─── Router: /api → backend, everything else → the adapter-bun app ──────────
Bun.serve({
  port: PORT,
  websocket: app.websocket as never,
  async fetch(req, server) {
    if (new URL(req.url).pathname.startsWith('/api/')) return proxy(req);
    return app.fetch(req, server);
  },
});

console.log(`→ frontend:  http://localhost:${PORT}  (svelte-adapter-bun)`);
console.log(`  proxying /api/*  →  ${API_BASE}`);
