import { chromium } from 'playwright-core';
const [BASE, CHROME] = process.argv.slice(2);
const browser = await chromium.launch({ executablePath: CHROME, headless: true,
  args: ['--headless=new','--no-sandbox','--enable-unsafe-webgpu','--enable-features=Vulkan','--use-angle=vulkan'] });
const page = await browser.newPage();
await page.goto(`${BASE}/annotate`, { waitUntil: 'networkidle', timeout: 45000 });
let status = '';
for (let i=0;i<30;i++){ status=(await page.locator('[data-testid=annotate-status]').textContent().catch(()=> ''))??''; if(/annotations from Lance/.test(status)) break; await page.waitForTimeout(500); }
const canvas = await page.locator('canvas').count();
// ported ra-anno layout — the five surfaces the goal requires
const toolbar   = await page.locator('[data-testid=annotator-toolbar]').count();
const sidebar   = await page.locator('[data-testid=annotation-sidebar]').count();
const layers    = await page.locator('[data-testid=layer-panel]').count();
const zoom      = await page.locator('[data-testid=zoom-controls]').count();
const pagenav   = await page.locator('[data-testid=page-nav]').count();
// the 3 seeded annotations render in the review queue, predictions first
const listItems = await page.locator('[data-testid=annotation-list] ul > li').count();
const topOfQueue = (await page.locator('[data-testid=annotation-list] ul > li').first().textContent().catch(()=> ''))??'';
console.log('viewer status :', status.trim());
console.log('canvas count  :', canvas);
console.log('toolbar       :', toolbar, '| sidebar', sidebar, '| layers', layers, '| zoom', zoom, '| pagenav', pagenav);
console.log('queue items   :', listItems);
console.log('top of queue  :', topOfQueue.replace(/\s+/g,' ').trim().slice(0,60));
const layoutOk = toolbar===1 && sidebar===1 && layers===1 && zoom===1 && pagenav===1;
const dataOk = canvas>0 && listItems===3 && /annotations from Lance/.test(status);
console.log(layoutOk && dataOk ? 'BOTH OK' : 'BOTH FAIL');
await browser.close();
process.exit(layoutOk && dataOk ? 0 : 1);
