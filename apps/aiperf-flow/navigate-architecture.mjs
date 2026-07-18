import { chromium } from "@playwright/test";

const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 2 });

// Navigate to aiperf-flow architecture flow
await p.goto("http://127.0.0.1:5188#/architecture.flow/Control%20plane", { waitUntil: "networkidle" });

// Wait for page to load
await p.waitForTimeout(1500);

// Dismiss audio consent modal
try { 
  await p.getByRole("button", { name: "Play without audio" }).click({ timeout: 5000 }); 
} catch (e) {
  console.log("Audio consent not found or already dismissed:", e.message);
}

// Wait for modal to disappear
await p.waitForTimeout(800);

// Take screenshot
await p.screenshot({ path: "/tmp/aiperf-flow-architecture-slide1.png" });

await b.close();
console.log("Shot architecture.flow slide 1");
