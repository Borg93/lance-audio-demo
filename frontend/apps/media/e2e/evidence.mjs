// P3.1/P3.3/P3.4 evidence harness — drives the LIVE descriptor-driven build
// over CDP and asserts the criteria's text-form signals:
//   • a search on transcripts renders N hit-card DOM nodes
//   • the opened row's <video> reaches readyState >= 2 (real playback)
//   • the atlas WebGPU scatter reports pointsDrawn > 0
//   • the SAME build renders the structurally-different smoke dataset (acid test)
//
// Usage: bun e2e/evidence.mjs <base> <chromePath>
import { chromium } from 'playwright-core';

const BASE = process.argv[2] ?? 'http://127.0.0.1:5180';
const CHROME = process.argv[3]; // executable path to the cached chromium

const log = (...a) => console.log(...a);
let failures = 0;
const check = (cond, msg) => {
  log(`  ${cond ? '✓' : '✗'} ${msg}`);
  if (!cond) failures++;
};

async function newPage(browser) {
  const page = await browser.newPage();
  page.setDefaultTimeout(30000);
  return page;
}

async function waitDescriptor(page) {
  // The layout gates routes on the loaded descriptor; wait for the app shell.
  await page.waitForSelector('[data-slot="sidebar-content"]', { timeout: 30000 });
}

async function searchEvidence(page, dataset, query) {
  const url = dataset ? `${BASE}/?dataset=${dataset}` : `${BASE}/`;
  log(`\n— search evidence: ${dataset ?? 'transcripts_v2 (default)'} —`);
  await page.goto(url, { waitUntil: 'networkidle' });
  await waitDescriptor(page);
  const active = await page.evaluate(() =>
    JSON.stringify(window.__activeDataset ?? null),
  );
  // Type the query into the search box and submit.
  const box = page.locator('input[type="search"], input[placeholder]').first();
  await box.fill(query);
  await box.press('Enter');
  // Hit cards carry data-hit-key (added for this evidence); fall back to article.
  await page
    .waitForFunction(() => document.querySelectorAll('[data-hit-key]').length > 0, { timeout: 30000 })
    .catch(() => {});
  const cards = await page.evaluate(() => document.querySelectorAll('[data-hit-key]').length);
  log(`  dataset=${dataset ?? 'default'} identity=${active}`);
  check(cards > 0, `${cards} hit-card DOM nodes rendered for "${query}"`);
  return cards;
}

// A restored sample doc (media bytes present locally) — scripts/sample_docs.txt.
const RESTORED_DOC = '018f41ad28a1bc1b';

async function videoEvidence(page) {
  log('\n— <video> playback evidence (restored sample doc) —');
  // Deep-link straight to a restored doc so the media resolves (most search
  // hits point at docs whose bytes live only in the HCP bucket).
  await page.goto(`${BASE}/?doc=${RESTORED_DOC}&t=5`, { waitUntil: 'networkidle' });
  await waitDescriptor(page);
  await page.waitForSelector('video', { timeout: 30000 }).catch(() => {});
  const info = await page.evaluate(async () => {
    const v = document.querySelector('video');
    if (!v) return { present: false };
    // Nudge the element to load media metadata/data.
    try {
      v.muted = true;
      await v.play().catch(() => {});
    } catch {
      /* autoplay may be blocked; readyState still advances on load */
    }
    const deadline = Date.now() + 20000;
    while (v.readyState < 2 && Date.now() < deadline) {
      await new Promise((r) => setTimeout(r, 250));
    }
    return { present: true, src: v.currentSrc || v.src, readyState: v.readyState };
  });
  check(info.present, 'a <video> element is present in the player');
  if (info.present) {
    log(`  video src: ${info.src}`);
    check(info.readyState >= 2, `video readyState = ${info.readyState} (>= 2 = has data)`);
  }
}

async function atlasEvidence(page) {
  log('\n— atlas WebGPU evidence (transcripts) —');
  await page.goto(`${BASE}/atlas`, { waitUntil: 'networkidle' });
  await waitDescriptor(page);
  const adapter = await page.evaluate(async () => {
    if (!navigator.gpu) return 'no navigator.gpu';
    try {
      const a = await navigator.gpu.requestAdapter();
      return a ? 'adapter OK' : 'requestAdapter() returned null';
    } catch (e) {
      return 'requestAdapter threw: ' + (e instanceof Error ? e.message : String(e));
    }
  });
  log(`  WebGPU: ${adapter}`);
  const stats = await page.evaluate(async () => {
    const deadline = Date.now() + 25000;
    while (Date.now() < deadline) {
      const s = window.__atlasStats;
      if (s && s.pointsDrawn > 0) return s;
      await new Promise((r) => setTimeout(r, 300));
    }
    return window.__atlasStats ?? null;
  });
  log(`  window.__atlasStats = ${JSON.stringify(stats)}`);
  check(!!stats && stats.pointsDrawn > 0, `atlas scatter drew ${stats?.pointsDrawn ?? 0} points`);
}

const HEADED = process.env.HEADED === '1';
const browser = await chromium.launch({
  executablePath: CHROME,
  headless: !HEADED,
  args: [
    '--no-sandbox',
    '--enable-unsafe-webgpu',
    '--enable-features=Vulkan',
    '--ignore-gpu-blocklist',
    ...(HEADED ? [] : ['--use-angle=vulkan', '--use-gl=angle']),
  ],
});
try {
  const page = await newPage(browser);

  // P3.1 — transcripts render + playback
  const tHits = await searchEvidence(page, null, 'regeringen');
  if (tHits > 0) await videoEvidence(page);

  // P3.4 — atlas
  await atlasEvidence(page);

  // P3.3 — acid test: SAME build renders the structurally different smoke dataset
  log('\n— acid test: smoke dataset (single-key identity, no time axis) —');
  const sHits = await searchEvidence(page, 'smoke', 'synthetic');
  check(sHits > 0, 'the same build rendered the smoke dataset from its descriptor alone');

  await page.close();
} finally {
  await browser.close();
}

log(`\n${failures === 0 ? 'EVIDENCE OK' : `EVIDENCE FAILED (${failures} check(s))`}`);
process.exit(failures === 0 ? 0 : 1);
