/**
 * Annotator E2E regression suite — drives the REAL app in a real (headless, WebGPU)
 * Chromium: every drawing tool, the OpenCV tools, AI-assist (Detect + SAM), and
 * draw → save → persist. Failures here mean a user-visible feature broke.
 *
 * Preconditions (asserted at start):
 *   - backend at :8000  (`uv run --no-sync rmedia --db transcripts_v2.lance serve`)
 *   - dev server at :5175 (`bun run dev --port 5175` in apps/media)
 *   - a chromium with WebGPU: default = the ms-playwright cache; override with E2E_CHROME.
 *
 * Run: `bun run test:e2e` (from apps/media). Re-seeds the demo annotations before + after,
 * so the suite is deterministic and leaves the demo clean.
 */
import { execFileSync } from "node:child_process";
import { existsSync, readdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright-core";

const BASE = process.env.E2E_BASE ?? "http://127.0.0.1:5175";
const API = process.env.E2E_API ?? "http://127.0.0.1:8000";
const KEY = "fe00cd746463ad2c/0/19";
const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../../..");

function chromePath() {
  if (process.env.E2E_CHROME) return process.env.E2E_CHROME;
  const cache = path.join(process.env.HOME ?? "", ".cache/ms-playwright");
  const builds = existsSync(cache)
    ? readdirSync(cache).filter((d) => /^chromium-\d+$/.test(d)).sort()
    : [];
  for (const b of builds.reverse()) {
    const p = path.join(cache, b, "chrome-linux/chrome");
    if (existsSync(p)) return p;
  }
  throw new Error("no chromium found — set E2E_CHROME or `npx playwright-core install chromium`");
}

const seed = () =>
  execFileSync("make", ["seed-annotations", "DB=transcripts_v2.lance"], { cwd: REPO, stdio: "pipe" });

const results = [];
const ok = (name, pass, detail = "") => {
  results.push({ name, pass });
  console.log(`${pass ? "PASS" : "FAIL"}  ${name}${detail ? " — " + detail : ""}`);
};

// ── preconditions ──
for (const [name, url] of [["backend", `${API}/api/datasets`], ["dev server", `${BASE}/`]]) {
  const res = await fetch(url).catch(() => null);
  if (!res?.ok) {
    console.error(`PRECONDITION FAILED: ${name} not reachable at ${url}`);
    process.exit(2);
  }
}
seed();

const browser = await chromium.launch({
  executablePath: chromePath(),
  headless: false, // new-headless below keeps WebGPU/Vulkan available
  args: [
    "--headless=new",
    "--no-sandbox",
    "--enable-unsafe-webgpu",
    "--enable-features=Vulkan",
    "--use-angle=vulkan",
    "--ignore-gpu-blocklist",
  ],
});
const page = await browser.newPage();
const pageErrors = [];
page.on("pageerror", (e) => pageErrors.push(e.message));

const count = async () => {
  const t = await page.locator('[title="Annotation count"]').first().textContent().catch(() => null);
  return t ? parseInt(t.trim(), 10) : NaN;
};

await page.goto(`${BASE}/annotate?keys=${KEY}`, { waitUntil: "networkidle", timeout: 60000 });
await page.waitForTimeout(2500);
const status = await page.locator('[data-testid="annotate-status"]').textContent().catch(() => "");
ok("annotator loads", /annotations from Lance/.test(status ?? ""), status ?? "");
const box = await page.locator("canvas").first().boundingBox();
ok("canvas mounted", !!box && box.width > 100);

const cx = box.x + box.width / 2;
const cy = box.y + box.height / 2;
const pt = (fx, fy) => [box.x + box.width * fx, box.y + box.height * fy];
async function drag(ax, ay, bx, by, steps = 8) {
  await page.mouse.move(ax, ay);
  await page.mouse.down();
  await page.mouse.move((ax + bx) / 2, (ay + by) / 2, { steps: 3 });
  await page.mouse.move(bx, by, { steps });
  await page.mouse.up();
}
async function tool(key) {
  await page.mouse.move(cx, cy);
  await page.keyboard.press(key);
  await page.waitForTimeout(200);
}
/** Assert a gesture on tool `key` grows the annotation count by ≥1. */
async function draws(name, key, gesture) {
  await tool(key);
  const before = await count();
  await gesture();
  await page.waitForTimeout(500);
  const after = await count();
  ok(`tool '${name}' commits a shape`, Number.isFinite(after) && after > before, `${before} → ${after}`);
}

// ── simple drawing tools ──
await draws("rect", "3", () => drag(...pt(0.30, 0.30), ...pt(0.42, 0.42)));
await draws("point", "5", () => page.mouse.click(cx + 30, cy + 30));
await draws("line", "6", () => drag(...pt(0.30, 0.60), ...pt(0.48, 0.63)));
await draws("polygon", "4", async () => {
  const [px, py] = pt(0.55, 0.55);
  await page.mouse.click(px, py);
  await page.mouse.click(px + 40, py);
  await page.mouse.click(px + 40, py + 40);
  await page.keyboard.press("Enter");
});
// pencil: freehand drag commits a simplified polygon on release
await draws("pencil", "8", async () => {
  const [px, py] = pt(0.60, 0.65);
  await page.mouse.move(px, py);
  await page.mouse.down();
  for (const [dx, dy] of [[30, 5], [55, 25], [40, 50], [10, 40]]) {
    await page.mouse.move(px + dx, py + dy, { steps: 4 });
  }
  await page.mouse.up();
});
// brush: strokes accumulate; Enter commits the mask
await draws("brush", "b", async () => {
  await drag(...pt(0.68, 0.30), ...pt(0.75, 0.38), 10);
  await page.keyboard.press("Enter");
});

// ── lasso is a SELECTION tool: loop around the canvas → annotations get selected ──
await tool("7");
{
  const [sx, sy] = pt(0.10, 0.10);
  const [ex, ey] = pt(0.90, 0.85);
  await page.mouse.move(sx, sy);
  await page.mouse.down();
  await page.mouse.move(ex, sy, { steps: 8 });
  await page.mouse.move(ex, ey, { steps: 8 });
  await page.mouse.move(sx, ey, { steps: 8 });
  await page.mouse.move(sx, sy, { steps: 8 });
  await page.mouse.up();
  await page.waitForTimeout(500);
  const detail = await page.locator('[data-testid="annotation-sidebar"]').getByText("Back to list").count();
  ok("tool 'lasso' selects enclosed annotations", detail > 0, "detail pane opened");
  await page.keyboard.press("Escape"); // clear selection for the next tools
  await page.waitForTimeout(200);
}

// ── OpenCV magnetic tool: lazy 8MB wasm init, then corner-snap commits ──
// (An IntelligentScissors edge-trace tool was removed 2026-07-21 — the opencv-js
// binding pathfinds degenerately; see InteractionManager. Magnetic is the CV tool.)
await tool("9"); // magnetic — arms lazy init (wasm + corner detection)
const magReady = await page
  .waitForSelector('[data-testid="annotator-toolbar"] button[data-cvready="true"]', { timeout: 90000 })
  .then(() => true)
  .catch(() => false);
ok("magnetic: OpenCV init completes", magReady);
if (magReady) {
  const before = await count();
  const [mx, my] = pt(0.35, 0.70);
  await page.mouse.click(mx, my);
  await page.mouse.click(mx + 50, my);
  await page.mouse.click(mx + 50, my + 40);
  await page.mouse.dblclick(mx + 50, my + 40);
  await page.waitForTimeout(600);
  const after = await count();
  ok("tool 'magnetic' commits a corner-snapped polygon", Number.isFinite(after) && after > before, `${before} → ${after}`);
}

// ── AI-assist: Detect (GroundingDINO text) + Segment (SAM box) ──
const assistBar = page.locator('[data-testid="ai-assist"]');
{
  const before = await count();
  await assistBar.getByText("Detect").click();
  await assistBar.locator("input").first().fill("text line");
  const called = page.waitForResponse((r) => r.url().includes("/api/assist/"), { timeout: 10000 }).then(() => true).catch(() => false);
  await assistBar.getByText("Run").click();
  ok("AI-assist Detect calls /api/assist", await called);
  await page.waitForTimeout(600);
  ok("AI-assist Detect adds predictions", (await count()) > before, `${before} → ${await count()}`);
}
{
  await assistBar.getByText("Segment").click();
  await page.waitForTimeout(300);
  const before = await count();
  const called = page.waitForResponse((r) => r.url().includes("/api/assist/"), { timeout: 10000 }).then(() => true).catch(() => false);
  await drag(...pt(0.15, 0.75), ...pt(0.28, 0.88), 6);
  ok("SAM Segment calls /api/assist", await called);
  await page.waitForTimeout(600);
  ok("SAM Segment adds a mask shape", (await count()) > before, `${before} → ${await count()}`);
  await assistBar.getByText("Detect").click(); // disarm SAM
}

// ── save → reload → everything persisted ──
const beforeSave = await count();
const saved = page
  .waitForResponse((r) => r.url().includes("/api/annotations/") && r.request().method() === "POST", { timeout: 10000 })
  .then((r) => r.ok())
  .catch(() => false);
await page.keyboard.press("Control+s");
ok("save POSTs the batch", await saved);
await page.waitForTimeout(600);
await page.goto(`${BASE}/annotate?keys=${KEY}`, { waitUntil: "networkidle", timeout: 60000 });
await page.waitForTimeout(2500);
const persisted = await count();
ok("all shapes persist across reload", Number.isFinite(persisted) && persisted === beforeSave, `${persisted} (expected ${beforeSave})`);

ok("no page-level JS errors", pageErrors.length === 0, pageErrors.slice(0, 3).join(" | "));

await browser.close();
seed(); // leave the demo clean
const passed = results.filter((r) => r.pass).length;
console.log(`\n=== E2E: ${passed}/${results.length} passed ===`);
process.exit(passed === results.length ? 0 : 1);
