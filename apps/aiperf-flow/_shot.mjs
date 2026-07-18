import { chromium } from "@playwright/test";
const url = process.argv[2]; const out = process.argv[3] || "/tmp/flow.png";
const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 2 });
await p.goto(url, { waitUntil: "networkidle" });
try { await p.getByRole("button", { name: "Play without audio" }).click({ timeout: 3000 }); } catch {}
await p.waitForTimeout(700);
await p.screenshot({ path: out });
await b.close();
console.log("shot", out);
