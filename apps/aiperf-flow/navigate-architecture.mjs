import { chromium } from "@playwright/test";

const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 2 });

// Navigate to aiperf-flow
await p.goto("http://127.0.0.1:5188", { waitUntil: "networkidle" });

// Wait for the page to load
await p.waitForTimeout(1000);

// Click on architecture.flow in the sidebar (first match)
const archLink = p.locator('text="architecture.flow"').first();
await archLink.click();

// Wait for navigation
await p.waitForTimeout(1500);

// Take screenshot
await p.screenshot({ path: "/tmp/aiperf-flow-architecture-slide1.png" });

await b.close();
console.log("Shot architecture.flow slide 1");
