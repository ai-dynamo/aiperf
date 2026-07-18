import { chromium } from "@playwright/test";

const base = "http://localhost:5178";
const shots = [
  ["/story?audience=executive", "story-exec-1"],
  ["/story?audience=developer", "story-dev-1"],
  ["/story?audience=maintainer", "story-maint-1"],
];

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
for (const [path, name] of shots) {
  await page.goto(`${base}${path}`, { waitUntil: "networkidle" });
  await page.waitForTimeout(1200);
  await page.screenshot({ path: `artifacts/${name}.png`, fullPage: false });
  console.log(`shot ${name}`);
}

// Step to chapter 4 (Clock) on the developer view to capture the trait fan.
await page.goto(`${base}/story?audience=developer`, { waitUntil: "networkidle" });
await page.waitForTimeout(600);
for (let i = 0; i < 3; i += 1) {
  await page.getByRole("button", { name: /^Next/ }).click();
  await page.waitForTimeout(300);
}
await page.screenshot({ path: "artifacts/story-dev-clock.png", fullPage: false });
console.log("shot story-dev-clock");

// Maintainer view, dispatch chapter (RequestSink fan with evidence paths).
await page.goto(`${base}/story?audience=maintainer`, { waitUntil: "networkidle" });
await page.waitForTimeout(600);
for (let i = 0; i < 6; i += 1) {
  await page.getByRole("button", { name: /^Next/ }).click();
  await page.waitForTimeout(250);
}
await page.screenshot({ path: "artifacts/story-maint-dispatch.png", fullPage: false });
console.log("shot story-maint-dispatch");

await browser.close();
