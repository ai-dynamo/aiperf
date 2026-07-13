import { chromium } from "@playwright/test";
const b = await chromium.launch({ executablePath: "/usr/bin/google-chrome-stable" });
const p = await b.newPage({ viewport: { width: 1500, height: 950 } });
await p.goto("http://localhost:4187/", { waitUntil: "networkidle" });
await p.waitForTimeout(3000);
console.log("VIEWPORT on load:", await p.$eval(".react-flow__viewport", e=>e.style.transform));
await p.screenshot({ path: "artifacts/_fit.png" });
await b.close();
