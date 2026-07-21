/**
 * Read-plane E2E — the front page (WebGPU adapter, SavedViews save/apply) and the
 * annotator's provenance surfaces (compare-versions History panel fetching /versions).
 * Graduated from the session scratchpad smoke into the committed suite. Same
 * preconditions as the annotator suite (see lib.mjs / TESTING.md).
 */
import { BASE, KEY, assertPreconditions, collector, launchBrowser, seed } from "./lib.mjs";

const { ok, finish } = collector("read-plane E2E");

await assertPreconditions();
seed();

const browser = await launchBrowser();
const page = await browser.newPage();
const pageErrors = [];
page.on("pageerror", (e) => pageErrors.push(e.message));

async function suite() {
  // ── front page: WebGPU + SavedViews ──
  await page.goto(`${BASE}/`, { waitUntil: "networkidle", timeout: 60000 });
  // Adapter acquisition can transiently fail right after another browser instance
  // released the GPU (Vulkan device contention) — retry on a FRESH page before
  // declaring WebGPU broken. This is the suite's only explicit WebGPU assertion
  // (Pixi would silently fall back), so it must be robust, not deleted.
  let gpu = "unattempted";
  for (let attempt = 0; attempt < 3 && gpu !== "adapter-ok"; attempt++) {
    if (attempt > 0) await page.waitForTimeout(1500);
    const probe = attempt === 0 ? page : await browser.newPage();
    if (probe !== page) await probe.goto(`${BASE}/`, { waitUntil: "domcontentloaded", timeout: 30000 });
    gpu = await probe.evaluate(async () => {
      if (!navigator.gpu) return "no-navigator.gpu";
      try {
        const a = await navigator.gpu.requestAdapter();
        return a ? "adapter-ok" : "no-adapter";
      } catch (e) {
        return "err:" + e.message;
      }
    });
    if (probe !== page) await probe.close();
  }
  ok("WebGPU adapter available", gpu === "adapter-ok", gpu);
  await page.waitForTimeout(1200);

  const viewsBtn = page.locator('button:has-text("Views")').first();
  ok("SavedViews control renders", (await viewsBtn.count()) > 0);
  await viewsBtn.click();
  await page.waitForTimeout(300);
  const saveInput = page.locator('input[placeholder*="Save current view"]');
  ok("SavedViews dropdown opens", (await saveInput.count()) > 0);

  // save the current (empty) view under a name, re-open, APPLY it → runSearch fires
  await saveInput.fill("e2e-view");
  await page.keyboard.press("Enter");
  await page.waitForTimeout(300);
  const savedRow = page.locator("li", { hasText: "e2e-view" }).first();
  ok("view saves + lists", (await savedRow.count()) > 0);
  await savedRow.locator("button").first().click(); // apply → runSearch (browse no-op is fine)
  await page.waitForTimeout(300);
  // delete it again (leave localStorage clean)
  await page.locator('button:has-text("Views")').first().click();
  await page.waitForTimeout(200);
  const delBtn = page.locator("li", { hasText: "e2e-view" }).locator('button[title="Delete view"]');
  if (await delBtn.count()) await delBtn.first().click();
  ok("view deletes", (await page.locator("li", { hasText: "e2e-view" }).count()) === 0);

  // ── annotator provenance: the History panel fetches /versions + lists rows ──
  await page.goto(`${BASE}/annotate?keys=${KEY}`, { waitUntil: "networkidle", timeout: 60000 });
  await page.waitForTimeout(2000);
  const history = page.locator('[data-testid="version-history"]');
  ok("History panel present", (await history.count()) > 0);
  const versionsFetched = page
    .waitForResponse((r) => r.url().includes("/versions"), { timeout: 10000 })
    .then(() => true)
    .catch(() => false);
  await history.locator("button").first().click();
  ok("History expand fetches /versions", await versionsFetched);
  await page.waitForTimeout(500);
  const vrows = await history.locator("button", { hasText: /^v\d+/ }).count();
  ok("History lists version rows", vrows > 0, `${vrows} rows`);

  ok("no page-level JS errors", pageErrors.length === 0, pageErrors.slice(0, 3).join(" | "));
}

let crashed = null;
try {
  await suite();
} catch (e) {
  crashed = e;
  console.error("SUITE CRASH:", e?.message ?? e);
} finally {
  await browser.close().catch(() => {});
  try {
    seed();
  } catch {
    console.error("WARNING: post-run re-seed failed");
  }
  finish(crashed);
}
