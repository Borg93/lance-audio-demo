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

// table view: the List/Table toggle shows the sortable table, then back to list
await page
  .getByTitle("Table")
  .click()
  .catch(() => {});
await page.waitForTimeout(150);
const tableRows = await page.locator("[data-testid=annotation-table] tbody tr").count();
await page
  .getByTitle("List")
  .click()
  .catch(() => {});
await page.waitForTimeout(150);

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

// multi-select: Shift-click two list items → the bulk-actions bar shows "2 selected"
await page.locator("body").press("Escape");
await page.waitForTimeout(150);
await page
  .locator("[data-testid=annotation-list] ul > li button")
  .nth(0)
  .click({ modifiers: ["Shift"] });
await page
  .locator("[data-testid=annotation-list] ul > li button")
  .nth(1)
  .click({ modifiers: ["Shift"] });
await page.waitForTimeout(200);
const bulkText = (
  (await page
    .locator("[data-testid=bulk-actions]")
    .textContent()
    .catch(() => "")) || ""
).replace(/\s+/g, " ");
const multiOk = /2 selected/.test(bulkText);

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
console.log("queue items   :", listItems, "| table rows", tableRows);
console.log(
  "accept-advance:",
  reviewOk ? "OK" : `FAIL acc ${acc0}->${acc1}->${acc2} advanced=${advancedTo !== advancedFrom}`,
);
console.log("multi-select :", multiOk ? "OK" : `FAIL bulk="${bulkText.slice(0, 40)}"`);
const layoutOk = toolbar === 1 && sidebar === 1 && layers === 1 && zoom === 1 && pagenav === 1;
const dataOk =
  canvas > 0 && listItems === 3 && tableRows === 3 && /annotations from Lance/.test(status);
console.log(layoutOk && dataOk && reviewOk && multiOk ? "BOTH OK" : "BOTH FAIL");
await browser.close();
process.exit(layoutOk && dataOk && reviewOk && multiOk ? 0 : 1);
