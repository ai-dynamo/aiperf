import { chromium } from "@playwright/test";
const url = process.argv[2] || "http://127.0.0.1:5188/?scene=request-investigation&beat=evidence";
const out = process.argv[3] || "/tmp/flow.png";
const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 2 });
await p.goto(url, { waitUntil: "networkidle" });
const btn = p.getByRole("button", { name: "Play without audio" });
try { await btn.click({ timeout: 3000 }); } catch {}
await p.waitForTimeout(500);
// pause playback so HUD stays visible
await p.waitForTimeout(300);
await p.screenshot({ path: out });
await b.close();
console.log("shot", out);
