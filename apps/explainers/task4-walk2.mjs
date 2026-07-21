import { chromium } from "@playwright/test";
import fs from "node:fs";
const OUT = "/tmp/task4-shots2";
fs.rmSync(OUT, { recursive: true, force: true });
fs.mkdirSync(OUT, { recursive: true });
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
await page.goto("http://localhost:5231/#/rust-architecture-deck-port", { waitUntil: "networkidle" });
await page.waitForTimeout(500);
const silent = page.getByRole("button", { name: "Play without audio" });
if (await silent.count()) { await silent.first().click(); await page.waitForTimeout(400); }
const dots = page.locator('[aria-label^="Go to slide"]');
const targets = [3,10,25,26,29,38,39,40,41,42,44,46,47]; // 1-based
for (const t of targets) {
  await dots.nth(t - 1).click();
  await page.waitForTimeout(3200); // allow entrance reveal
  const num = String(t).padStart(2, "0");
  await page.screenshot({ path: `${OUT}/slide-${num}.png` });
  console.log("captured", num);
}
await browser.close();
console.log("DONE");
