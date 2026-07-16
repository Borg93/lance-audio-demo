import { chromium } from 'playwright-core';
const BASE = process.argv[2];
const CHROME = process.argv[3];
const browser = await chromium.launch({
  executablePath: CHROME,
  headless: true,
  args: ['--no-sandbox', '--enable-unsafe-webgpu', '--use-angle=swiftshader', '--ignore-gpu-blocklist'],
});
const page = await browser.newPage();
const errors = [];
page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
page.on('pageerror', (e) => errors.push('PAGEERROR: ' + e.message));
const requests = [];
page.on('response', (r) => { if (r.url().includes('/api/')) requests.push(`${r.status()} ${r.url().replace(BASE, '')}`); });

await page.goto(`${BASE}/`, { waitUntil: 'networkidle' });
await page.waitForSelector('[data-slot="sidebar-content"]');
await new Promise((r) => setTimeout(r, 1000));

// Enumerate candidate search inputs
const inputs = await page.evaluate(() =>
  [...document.querySelectorAll('input')].map((i) => ({
    type: i.type, placeholder: i.placeholder, ariaLabel: i.getAttribute('aria-label'), visible: i.offsetParent !== null,
  })),
);
console.log('INPUTS:', JSON.stringify(inputs, null, 1));

// Try the search
const box = page.locator('input[type="search"]').first();
await box.fill('regeringen');
await box.press('Enter');
await new Promise((r) => setTimeout(r, 3000));

console.log('API REQUESTS:', JSON.stringify(requests, null, 1));
console.log('CONSOLE ERRORS:', JSON.stringify(errors.slice(0, 10), null, 1));

const domState = await page.evaluate(() => ({
  hitKeys: document.querySelectorAll('[data-hit-key]').length,
  articles: document.querySelectorAll('article').length,
  buttons: document.querySelectorAll('main button, [data-slot] button').length,
  bodyText: document.querySelector('main')?.innerText?.slice(0, 300) ?? document.body.innerText.slice(0, 300),
}));
console.log('DOM STATE:', JSON.stringify(domState, null, 1));

await browser.close();
