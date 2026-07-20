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
// the 3 seeded annotations render in the review queue, predictions first
const listItems = await page.locator("[data-testid=annotation-list] ul > li").count();
const topOfQueue =
  (await page
    .locator("[data-testid=annotation-list] ul > li")
    .first()
    .textContent()
    .catch(() => "")) ?? "";

// functional: undo/redo a review edit (select → Accept → Undo reverts)
const detailText = async () =>
  (
    (await page
      .locator("[data-testid=annotation-detail]")
      .textContent()
      .catch(() => "")) || ""
  ).replace(/\s+/g, " ");
await page.locator("[data-testid=annotation-list] ul > li button").first().click();
await page.waitForSelector("[data-testid=annotation-detail]", { timeout: 5000 }).catch(() => {});
const before = await detailText();
await page
  .locator("[data-testid=annotation-detail] button", { hasText: "Accept" })
  .click()
  .catch(() => {});
await page.waitForTimeout(200);
const afterAccept = await detailText();
// Save button becomes enabled once dirty (non-destructive check — we don't click it;
// the live POST→merge_insert round-trip is proven separately).
const saveEnabled = await page.getByTitle(/^Save to Lance/).isEnabled().catch(() => false);
await page
  .getByTitle(/^Undo/)
  .click()
  .catch(() => {});
await page.waitForTimeout(200);
const afterUndo = await detailText();
const undoOk =
  /prediction/.test(before) &&
  /accepted/.test(afterAccept) &&
  /prediction/.test(afterUndo) &&
  saveEnabled;

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
console.log("top of queue  :", topOfQueue.replace(/\s+/g, " ").trim().slice(0, 60));
console.log(
  "undo/redo     :",
  undoOk
    ? "OK"
    : `FAIL before=${before.slice(0, 30)} accept=${afterAccept.slice(0, 30)} undo=${afterUndo.slice(0, 30)}`,
);
const layoutOk = toolbar === 1 && sidebar === 1 && layers === 1 && zoom === 1 && pagenav === 1;
const dataOk = canvas > 0 && listItems === 3 && /annotations from Lance/.test(status);
console.log(layoutOk && dataOk && undoOk ? "BOTH OK" : "BOTH FAIL");
await browser.close();
process.exit(layoutOk && dataOk && undoOk ? 0 : 1);
