import { chromium } from "playwright-core";
const [BASE, CHROME] = process.argv.slice(2);
const browser = await chromium.launch({
  executablePath: CHROME,
  headless: true,
  args: [
    "--headless=new",
    "--no-sandbox",
    "--enable-unsafe-webgpu",
    "--enable-features=Vulkan",
    "--use-angle=vulkan",
  ],
});
const page = await browser.newPage();
await page.goto(`${BASE}/annotate`, { waitUntil: "networkidle", timeout: 45000 });
let status = "";
for (let i = 0; i < 30; i++) {
  status =
    (await page
      .locator("[data-testid=annotate-status]")
      .textContent()
      .catch(() => "")) ?? "";
  if (/annotations from Lance/.test(status)) break;
  await page.waitForTimeout(500);
}
const canvas = await page.locator("canvas").count();
// ported ra-anno layout — the five surfaces the goal requires
const toolbar = await page.locator("[data-testid=annotator-toolbar]").count();
const sidebar = await page.locator("[data-testid=annotation-sidebar]").count();
const layers = await page.locator("[data-testid=layer-panel]").count();
const zoom = await page.locator("[data-testid=zoom-controls]").count();
const pagenav = await page.locator("[data-testid=page-nav]").count();
const listItems = await page.locator("[data-testid=annotation-list] ul > li").count();

// review loop: select a prediction → accept-and-advance (A) bumps the accepted count
// + advances the selection; Ctrl+Z reverts. Read the count off the sidebar summary.
const acceptedCount = async () => {
  const t =
    (await page
      .locator("[data-testid=annotation-sidebar]")
      .textContent()
      .catch(() => "")) || "";
  const m = t.match(/accepted\s+(\d+)/);
  return m ? Number(m[1]) : -1;
};
const acc0 = await acceptedCount();
await page.locator("[data-testid=annotation-list] ul > li button").first().click();
await page.waitForSelector("[data-testid=annotation-detail]", { timeout: 5000 }).catch(() => {});
const advancedFrom = (
  (await page
    .locator("[data-testid=annotation-detail]")
    .textContent()
    .catch(() => "")) || ""
).replace(/\s+/g, " ");
const saveEnabledBefore = await page
  .getByTitle(/^Save to Lance/)
  .isEnabled()
  .catch(() => false);
await page.locator("body").press("a"); // accept-and-advance
await page.waitForTimeout(250);
const acc1 = await acceptedCount();
const advancedTo = (
  (await page
    .locator("[data-testid=annotation-detail]")
    .textContent()
    .catch(() => "")) || ""
).replace(/\s+/g, " ");
await page.locator("body").press("Control+z"); // undo the accept
await page.waitForTimeout(250);
const acc2 = await acceptedCount();
const reviewOk =
  acc0 === 1 &&
  acc1 === 2 &&
  acc2 === 1 &&
  saveEnabledBefore === false &&
  advancedTo !== advancedFrom;

console.log("viewer status :", status.trim());
console.log("canvas count  :", canvas);
console.log(
  "toolbar       :",
  toolbar,
  "| sidebar",
  sidebar,
  "| layers",
  layers,
  "| zoom",
  zoom,
  "| pagenav",
  pagenav,
);
console.log("queue items   :", listItems);
console.log(
  "accept-advance:",
  reviewOk ? "OK" : `FAIL acc ${acc0}->${acc1}->${acc2} advanced=${advancedTo !== advancedFrom}`,
);
const layoutOk = toolbar === 1 && sidebar === 1 && layers === 1 && zoom === 1 && pagenav === 1;
const dataOk = canvas > 0 && listItems === 3 && /annotations from Lance/.test(status);
console.log(layoutOk && dataOk && reviewOk ? "BOTH OK" : "BOTH FAIL");
await browser.close();
process.exit(layoutOk && dataOk && reviewOk ? 0 : 1);
