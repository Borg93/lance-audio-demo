import { chromium } from 'playwright-core';
const [BASE, CHROME] = process.argv.slice(2);
const browser = await chromium.launch({ executablePath: CHROME, headless: true,
  args: ['--headless=new','--no-sandbox','--enable-unsafe-webgpu','--enable-features=Vulkan','--use-angle=vulkan'] });
const page = await browser.newPage();
await page.goto(`${BASE}/annotate`, { waitUntil: 'networkidle', timeout: 45000 });
let status = '';
for (let i=0;i<30;i++){ status=(await page.locator('[data-testid=annotate-status]').textContent().catch(()=> ''))??''; if(/annotations from Lance/.test(status)) break; await page.waitForTimeout(500); }
const canvas = await page.locator('canvas').count();
const tableRows = await page.locator('[data-testid=annotation-table] tbody tr').count();
const firstRowStatus = (await page.locator('[data-testid=annotation-table] tbody tr').first().textContent().catch(()=> ''))??'';
console.log('viewer status :', status.trim());
console.log('canvas count  :', canvas);
console.log('table rows    :', tableRows);
console.log('top of queue  :', firstRowStatus.replace(/\s+/g,' ').trim().slice(0,60));
console.log(canvas>0 && tableRows===3 && /prediction/.test(firstRowStatus) ? 'BOTH OK' : 'BOTH FAIL');
await browser.close();
