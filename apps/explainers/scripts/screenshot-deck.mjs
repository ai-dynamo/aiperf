#!/usr/bin/env node
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Screenshot every slide (+ finalCard when present) of one explainer deck.
 *
 * Boots `npm run build` + `vite preview` (no HMR) unless `--base-url` is set.
 *
 * Usage:
 *   node apps/explainers/scripts/screenshot-deck.mjs --deck flow-sdk-examples
 *   node apps/explainers/scripts/screenshot-deck.mjs --deck flow-sdk-examples --base-url http://127.0.0.1:4173
 *   node apps/explainers/scripts/screenshot-deck.mjs --deck flow-sdk-examples --viewport 1280x720
 */

import { mkdir } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { createRequire } from "node:module";
import { createServer } from "node:http";
import { spawn } from "node:child_process";

const __dirname = dirname(fileURLToPath(import.meta.url));
const EXPLAINERS_ROOT = resolve(__dirname, "..");

function parseArgs(argv) {
  const options = {
    deck: "flow-sdk-examples",
    baseUrl: null,
    out: null,
    viewport: { width: 1440, height: 1100 },
  };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--deck") options.deck = argv[++i];
    else if (arg.startsWith("--deck=")) options.deck = arg.slice(7);
    else if (arg === "--base-url") options.baseUrl = argv[++i];
    else if (arg.startsWith("--base-url=")) options.baseUrl = arg.slice(11);
    else if (arg === "--out") options.out = argv[++i];
    else if (arg.startsWith("--out=")) options.out = arg.slice(6);
    else if (arg === "--viewport") options.viewport = parseViewport(argv[++i]);
    else if (arg.startsWith("--viewport=")) {
      options.viewport = parseViewport(arg.slice("--viewport=".length));
    }
  }
  return options;
}

function parseViewport(value) {
  const match = /^(\d+)x(\d+)$/.exec(value ?? "");
  if (!match) {
    throw new Error(`invalid --viewport "${value}"; expected WIDTHxHEIGHT`);
  }
  const viewport = { width: Number(match[1]), height: Number(match[2]) };
  if (viewport.width === 0 || viewport.height === 0) {
    throw new Error(`invalid --viewport "${value}"; dimensions must be positive`);
  }
  return viewport;
}

function resolvePlaywright() {
  const require = createRequire(import.meta.url);
  try {
    return require.resolve("playwright", { paths: [EXPLAINERS_ROOT] });
  } catch {
    return null;
  }
}

async function loadChromium() {
  const id = resolvePlaywright();
  if (!id) {
    throw new Error(
      "playwright not found; run `cd apps/explainers && npm install && npx playwright install chromium`",
    );
  }
  const mod = await import(pathToFileURL(id).href);
  const chromium = mod.chromium ?? mod.default?.chromium;
  if (!chromium) throw new Error("playwright has no chromium export");
  return chromium;
}

function waitForUrl(url, timeoutMs = 60_000) {
  const start = Date.now();
  return new Promise((resolveWait, reject) => {
    const tick = async () => {
      try {
        const res = await fetch(url);
        if (res.ok || res.status === 404) {
          resolveWait();
          return;
        }
      } catch {
        // retry
      }
      if (Date.now() - start > timeoutMs) {
        reject(new Error(`timed out waiting for ${url}`));
        return;
      }
      setTimeout(tick, 400);
    };
    tick();
  });
}

async function freePort() {
  return await new Promise((resolvePort) => {
    const server = createServer();
    server.listen(0, "127.0.0.1", () => {
      const { port } = server.address();
      server.close(() => resolvePort(port));
    });
  });
}

function spawnLogged(command, args, cwd) {
  const child = spawn(command, args, {
    cwd,
    stdio: ["ignore", "pipe", "pipe"],
    env: { ...process.env, BROWSER: "none" },
  });
  let logs = "";
  child.stdout.on("data", (d) => {
    logs += d.toString();
  });
  child.stderr.on("data", (d) => {
    logs += d.toString();
  });
  return { child, getLogs: () => logs };
}

async function waitForExit(child, label) {
  return await new Promise((resolve, reject) => {
    child.on("error", reject);
    child.on("exit", (code, signal) => {
      if (code === 0) {
        resolve();
        return;
      }
      reject(
        new Error(
          `${label} exited with code ${code}${signal ? ` signal ${signal}` : ""}`,
        ),
      );
    });
  });
}

/**
 * Serve a production build via `vite preview` — no HMR / file watchers.
 * Dev-server HMR remounts ExplainerShell mid-run and re-shows the start gate.
 */
async function startPreviewServer() {
  console.error("building explainers (no HMR preview)…");
  const build = spawnLogged("npm", ["run", "build"], EXPLAINERS_ROOT);
  try {
    await waitForExit(build.child, "npm run build");
  } catch (error) {
    throw new Error(`${error.message}\n${build.getLogs().slice(-4000)}`);
  }

  const port = await freePort();
  const baseUrl = `http://127.0.0.1:${port}`;
  const preview = spawnLogged(
    "npm",
    [
      "run",
      "preview",
      "--",
      "--host",
      "127.0.0.1",
      "--port",
      String(port),
      "--strictPort",
    ],
    EXPLAINERS_ROOT,
  );
  try {
    await waitForUrl(baseUrl);
  } catch (error) {
    preview.child.kill("SIGTERM");
    throw new Error(`${error.message}\n${preview.getLogs().slice(-2000)}`);
  }
  return {
    baseUrl,
    stop: () => preview.child.kill("SIGTERM"),
  };
}

async function suppressStartGateCss(page) {
  // Persist across Vite HMR remounts that reset `started` mid-run.
  await page.addStyleTag({
    content: ".ex-gate { display: none !important; pointer-events: none !important; }",
  });
}

async function dismissStartGate(page) {
  // Prefer an explicit click so ExplainerShell sets `started` / `playing`.
  for (let attempt = 0; attempt < 5; attempt += 1) {
    const silent = page.getByRole("button", { name: /Play without audio/i });
    if (await silent.count()) {
      await silent.first().click({ timeout: 5000, force: true }).catch(() => {});
    } else {
      const withAudio = page.getByRole("button", { name: /Play with audio/i });
      if (await withAudio.count()) {
        await withAudio.first().click({ timeout: 5000, force: true }).catch(() => {});
      }
    }
    await page.waitForTimeout(250);
    const gateVisible = await page.locator(".ex-gate").isVisible().catch(() => false);
    if (!gateVisible) {
      break;
    }
  }
  await suppressStartGateCss(page);
}

async function pausePlayback(page) {
  // Timed narration otherwise auto-advances while we wait for scenes to settle.
  const pause = page.getByRole("button", { name: /^Pause$/i });
  if (await pause.count()) {
    await pause.first().click({ force: true }).catch(() => {});
    await page.waitForTimeout(150);
  }
}

async function slideCount(page) {
  const text = await page
    .locator("text=/\\d+\\s*\\/\\s*\\d+/")
    .first()
    .textContent()
    .catch(() => null);
  if (!text) return null;
  const m = text.match(/(\d+)\s*\/\s*(\d+)/);
  return m ? Number(m[2]) : null;
}

async function goToSlide(page, slideIndex1Based) {
  const step = page.getByRole("button", {
    name: new RegExp(`^Go to slide ${slideIndex1Based}:`, "i"),
  });
  if (await step.count()) {
    await step.first().click({ force: true });
    await page.waitForTimeout(350);
    return true;
  }
  return false;
}

async function goNext(page) {
  // ExplainerShell's bottom-nav control is "Next →" (arrow glyph included in
  // the accessible name); anchoring the full string missed the button
  // entirely and silently truncated every route to slide 0. Match the prefix
  // instead so a glyph/label tweak doesn't do the same again.
  const next = page.getByRole("button", { name: /^Next/i });
  if (await next.count()) {
    const disabled = await next.first().isDisabled().catch(() => true);
    if (!disabled) {
      await next.first().click({ force: true });
      return true;
    }
  }
  return false;
}

async function waitForSceneSettle(page, ms = 2800) {
  await page.locator("svg.scene-renderer").first().waitFor({ timeout: 15_000 }).catch(() => {});
  await page.waitForTimeout(ms);
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const outDir =
    options.out ??
    join(EXPLAINERS_ROOT, "artifacts", "screenshots", options.deck);
  await mkdir(outDir, { recursive: true });

  let stop = null;
  let baseUrl = options.baseUrl;
  if (!baseUrl) {
    const server = await startPreviewServer();
    baseUrl = server.baseUrl;
    stop = server.stop;
  }

  const chromium = await loadChromium();
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({
    viewport: options.viewport,
    deviceScaleFactor: 1,
  });
  // Prefer reduced motion so scenes settle quickly for still frames.
  await page.emulateMedia({ reducedMotion: "reduce" });

  const route = `/#/${options.deck}`;
  const paths = [];
  try {
    await page.goto(`${baseUrl}${route}`, {
      waitUntil: "networkidle",
      timeout: 90_000,
    });
    await page.waitForTimeout(600);
    await dismissStartGate(page);
    await pausePlayback(page);
    await page.waitForTimeout(400);

    const total = (await slideCount(page)) ?? 1;
    console.error(`screenshotting ${options.deck}: ${total} slides → ${outDir}`);

    for (let i = 0; i < total; i += 1) {
      await dismissStartGate(page);
      await pausePlayback(page);
      const jumped = await goToSlide(page, i + 1);
      if (!jumped && i > 0) {
        const moved = await goNext(page);
        if (!moved) {
          console.error(`  stopped: cannot reach slide ${i + 1}/${total}`);
          break;
        }
      }
      await pausePlayback(page);
      await waitForSceneSettle(page, 1800);
      const name = `slide-${String(i + 1).padStart(2, "0")}.png`;
      const path = join(outDir, name);
      await page.screenshot({ path, fullPage: true });
      paths.push(path);
      console.error(`  wrote ${name}`);

      if (i === total - 1) {
        const finalCard = page.locator(".explainer-final-card");
        if (await finalCard.count()) {
          await page.waitForTimeout(600);
          const finalPath = join(outDir, "final-card.png");
          await finalCard.first().screenshot({ path: finalPath }).catch(async () => {
            await page.screenshot({ path: finalPath, fullPage: true });
          });
          paths.push(finalPath);
          console.error("  wrote final-card.png");
        }
      }
    }
  } finally {
    await browser.close().catch(() => {});
    if (stop) stop();
  }

  console.log(
    JSON.stringify(
      { deck: options.deck, viewport: options.viewport, outDir, paths },
      null,
      2,
    ),
  );
}

main().catch((error) => {
  console.error(String(error?.stack ?? error));
  process.exit(1);
});
