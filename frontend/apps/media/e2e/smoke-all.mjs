// Honest full-app smoke: visit every route against the LIVE build, capturing
// console errors, page errors, and whether it rendered without crashing.
// Reports per route — every failure surfaced, nothing hidden.
import { chromium } from 'playwright-core';

const BASE = process.argv[2];
const CHROME = process.argv[3];
const browser = await chromium.launch({
  executablePath: CHROME,
  headless: true,
  args: ['--no-sandbox', '--enable-unsafe-webgpu', '--enable-features=Vulkan', '--ignore-gpu-blocklist'],
});

// `needs`: a selector that must appear on transcripts (rich dataset). On smoke
// (search + documents only) we require only "no crash / no error", since
// atlas/topics/graph capabilities are legitimately absent (built:false).
const ROUTES = [
  { path: '/', name: 'search', needs: 'input[type="search"]' },
  { path: '/atlas', name: 'atlas', needs: 'canvas' },
  { path: '/tree', name: 'tree (topics)', needs: 'main' },
  { path: '/graph', name: 'graph (KG)', needs: 'main' },
  { path: '/workflow', name: 'workflow', needs: 'main' },
  { path: '/guide', name: 'guide', needs: 'main' },
];

function urlFor(base, path, dataset) {
  const q = dataset ? `?dataset=${dataset}` : '';
  return `${base}${path}${q}`;
}

async function visit(page, url, needs, requireContent) {
  const errors = [];
  const onConsole = (m) => { if (m.type() === 'error') errors.push(m.text().slice(0, 200)); };
  const onPageError = (e) => errors.push('PAGEERROR: ' + e.message.slice(0, 200));
  page.on('console', onConsole);
  page.on('pageerror', onPageError);
  let status = 'ok';
  let note = '';
  try {
    await page.goto(url, { waitUntil: 'networkidle', timeout: 30000 });
    await page.waitForSelector('[data-slot="sidebar-content"]', { timeout: 20000 });
    if (requireContent) {
      const found = await page.waitForSelector(needs, { timeout: 20000 }).then(() => true).catch(() => false);
      const bodyLen = await page.evaluate(() => document.querySelector('main')?.innerText?.length ?? 0);
      if (!found) { status = 'NO-CONTENT'; note = `"${needs}" not found`; }
      else if (bodyLen < 20) { status = 'EMPTY'; note = `main text ${bodyLen}`; }
    } else {
      // smoke: just needs to render the shell without throwing.
      const bodyLen = await page.evaluate(() => document.querySelector('main')?.innerText?.length ?? 0);
      note = `rendered (main text ${bodyLen})`;
    }
  } catch (e) {
    status = 'THREW';
    note = e instanceof Error ? e.message.slice(0, 150) : String(e);
  }
  page.off('console', onConsole);
  page.off('pageerror', onPageError);
  const real = errors.filter((e) => !/favicon|ResizeObserver|\[vite\]|hmr|WebSocket|net::ERR_ABORTED/i.test(e));
  return { status, note, errors: real.slice(0, 4) };
}

async function runDataset(label, dataset, requireContent) {
  console.log(`\n===== ${label} =====`);
  const page = await browser.newPage();
  page.setDefaultTimeout(30000);
  let ok = 0;
  for (const r of ROUTES) {
    const res = await visit(page, urlFor(BASE, r.path, dataset), r.needs, requireContent);
    const clean = res.status === 'ok' && res.errors.length === 0;
    if (clean) ok++;
    console.log(`  ${clean ? '✓' : '✗'} ${r.name.padEnd(16)} [${res.status}] ${res.note}`);
    for (const e of res.errors) console.log(`      console: ${e}`);
  }
  await page.close();
  console.log(`  ${label}: ${ok}/${ROUTES.length} routes clean`);
  return ok;
}

const t = await runDataset('transcripts_v2 (default, content required)', null, true);
const s = await runDataset('smoke (?dataset=smoke, no-crash required)', 'smoke', false);
await browser.close();
console.log(`\nTOTAL: transcripts ${t}/${ROUTES.length}, smoke ${s}/${ROUTES.length} route-renders clean`);
// Exit code must reflect the counts so this can gate CI / a make target — both
// datasets must render every route (this doc's cited "6/6 + 6/6" evidence).
const allClean = t === ROUTES.length && s === ROUTES.length;
process.exit(allClean ? 0 : 1);
