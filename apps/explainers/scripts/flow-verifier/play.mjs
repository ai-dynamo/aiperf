/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { createServer } from "node:http";
import { dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { createRequire } from "node:module";
import { spawn } from "node:child_process";

const __dirname = dirname(fileURLToPath(import.meta.url));
const EXPLAINERS_ROOT = resolve(__dirname, "../..");

const DECK_ROUTES = [
  "/#/rust-architecture",
  "/#/rust-architecture-atlas",
  "/#/segment-pools",
  "/#/slurm-velo",
  "/#/velo-deep-dive",
  "/#/cellular-internals",
  "/#/cellular-algorithms",
  "/#/dynosim",
  "/#/tstar-warmup",
  "/#/synthetic-dataset-generator",
];

function resolvePlaywright() {
  const require = createRequire(import.meta.url);
  try {
    return require.resolve("playwright", {
      paths: [EXPLAINERS_ROOT],
    });
  } catch {
    return null;
  }
}

async function loadChromium() {
  const id = resolvePlaywright();
  if (!id) {
    throw new Error(
      "playwright not found; install in apps/explainers (`npm install && npx playwright install chromium`) or pass --ir-only",
    );
  }
  const mod = await import(pathToFileURL(id).href);
  const chromium = mod.chromium ?? mod.default?.chromium;
  if (!chromium) {
    throw new Error(`playwright module at ${id} has no chromium export`);
  }
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

function runNpm(args, cwd) {
  return new Promise((resolveRun, reject) => {
    const child = spawn("npm", args, { cwd, stdio: "pipe", shell: false });
    let logs = "";
    child.stdout.on("data", (d) => {
      logs += d.toString();
    });
    child.stderr.on("data", (d) => {
      logs += d.toString();
    });
    child.on("error", reject);
    child.on("exit", (code) => {
      if (code === 0) resolveRun();
      else
        reject(
          new Error(
            `npm ${args.join(" ")} exited ${code}\n${logs.slice(-4000)}`,
          ),
        );
    });
  });
}

/**
 * Spawn the explainer app server for Playwright to visit.
 *
 * Defaults to `vite dev` (live `.flow` compile — matches the browser-owned
 * compiler path that `deck-registry.ts` uses at runtime). Passing
 * `usePreview: true` runs `npm run build && npm run preview` instead, so the
 * gate also proves production bundling. Preview is opt-in, not default:
 * `build` runs a workspace-wide `tsc --noEmit` ahead of `vite build`, and
 * during active deck/compiler migrations unrelated in-flight files can fail
 * that typecheck even when the runtime path under test is fine — see
 * `.superpowers/sdd/playwright-hardening-report.md`.
 */
async function startViteServer({ usePreview = false } = {}) {
  const port = await freePort();
  const baseUrl = `http://127.0.0.1:${port}`;

  if (usePreview) {
    await runNpm(["run", "build"], EXPLAINERS_ROOT);
  }

  const args = usePreview
    ? [
        "run",
        "preview",
        "--",
        "--host",
        "127.0.0.1",
        "--port",
        String(port),
        "--strictPort",
      ]
    : ["run", "dev", "--", "--host", "127.0.0.1", "--port", String(port)];
  const child = spawn("npm", args, {
    cwd: EXPLAINERS_ROOT,
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
  try {
    await waitForUrl(baseUrl);
  } catch (error) {
    child.kill("SIGTERM");
    throw new Error(`${error.message}\n${logs.slice(-2000)}`);
  }
  return {
    baseUrl,
    stop: () => {
      child.kill("SIGTERM");
    },
  };
}

async function dismissStartGate(page) {
  const silent = page.getByRole("button", { name: /Play without audio/i });
  if (await silent.count()) {
    await silent.first().click({ timeout: 5000 });
    return;
  }
  const withAudio = page.getByRole("button", { name: /Play with audio/i });
  if (await withAudio.count()) {
    await withAudio.first().click({ timeout: 5000 }).catch(() => {});
    return;
  }
  const fallback = page
    .getByRole("button")
    .filter({ hasText: /start|begin|continue|enter|play/i })
    .first();
  if (await fallback.count()) {
    await fallback.click({ timeout: 2000 }).catch(() => {});
  }
}

/**
 * `ExplainerShell`'s single play/pause toggle renders exactly one of the
 * labels "Play", "Pause", or "Replay" (no "slideshow" suffix — see
 * `ExplainerShell.tsx`). Anchor the regexes to the whole accessible name so
 * this doesn't silently no-op if the control text drifts again.
 */
const PLAYBACK_TOGGLE_START_RE = /^(Play|Replay)$/i;
const PLAYBACK_TOGGLE_PAUSE_RE = /^Pause$/i;
const PLAYBACK_TOGGLE_ANY_RE = /^(Play|Pause|Replay)$/i;

async function clickPlay(page) {
  const start = page.getByRole("button", { name: PLAYBACK_TOGGLE_START_RE });
  if (await start.count()) {
    await start.first().click();
    return;
  }
  const pause = page.getByRole("button", { name: PLAYBACK_TOGGLE_PAUSE_RE });
  if (await pause.count()) return;
  throw new Error("Play/Pause/Replay control not found");
}

/**
 * Live SVG assertions (design layer B).
 * Samples while playing so mid-draw arrowhead deferral is actually exercised.
 *
 * `rootSelector` scopes the lookup (e.g. to `.explainer-final-card`) so the
 * finalCard's own scene is asserted directly instead of relying on DOM order
 * versus the slide diagram's `svg.scene-renderer`.
 */
async function collectSvgFindings(page, deckRoute, slideIndex, options = {}) {
  const { rootSelector = null } = options;
  return await page.evaluate(
    ({ route, slide, rootSelector: rootSel }) => {
      const findings = [];
      const root = rootSel ? document.querySelector(rootSel) : document;
      const svg = root?.querySelector("svg.scene-renderer, svg");
      if (!svg) {
        findings.push({
          severity: "error",
          deck: route,
          slide: String(slide),
          code: "missing-svg",
          message: "no scene SVG on stage",
        });
        return findings;
      }
      const viewBox = svg.viewBox?.baseVal;
      const vb = viewBox
        ? { x: viewBox.x, y: viewBox.y, w: viewBox.width, h: viewBox.height }
        : { x: 0, y: 0, w: 700, h: 400 };

      const reduced =
        svg.getAttribute("data-scene-reduced-motion") === "true";
      const nodes = svg.querySelectorAll("[data-flow-node-id]");
      if (nodes.length === 0) {
        findings.push({
          severity: "error",
          deck: route,
          slide: String(slide),
          code: "empty-stage",
          message: "SVG has no data-flow-node-id nodes",
        });
      }

      // SceneRenderer contract: heads only when draw is complete (unless reduced).
      if (!reduced) {
        for (const el of svg.querySelectorAll("path[data-flow-arrowhead]")) {
          const head = el.getAttribute("data-flow-arrowhead");
          const dashoffset = el.getAttribute("stroke-dashoffset");
          if (head === "true" && dashoffset != null) {
            const off = Number(dashoffset);
            if (Number.isFinite(off) && off > 0.02 && off < 0.98) {
              findings.push({
                severity: "error",
                deck: route,
                slide: String(slide),
                code: "arrowhead-leads-stroke",
                message: `path shows arrowhead while stroke-dashoffset=${off}`,
              });
            }
          }
        }
      }

      const ctm = svg.getScreenCTM?.();
      for (const el of svg.querySelectorAll(
        "[data-flow-dot], [data-flow-motion-signal], circle.motion-signal",
      )) {
        const box = el.getBoundingClientRect();
        if (!Number.isFinite(box.x) || !Number.isFinite(box.y)) {
          findings.push({
            severity: "error",
            deck: route,
            slide: String(slide),
            code: "invalid-dot-box",
            message: "motion dot has non-finite screen box",
          });
          continue;
        }
        if (box.width === 0 && box.height === 0) {
          continue;
        }
        // Map screen center into SVG user space when CTM is available.
        if (ctm && typeof DOMPointReadOnly !== "undefined") {
          try {
            const inv = ctm.inverse();
            const pt = new DOMPointReadOnly(
              box.left + box.width / 2,
              box.top + box.height / 2,
            ).matrixTransform(inv);
            const margin = 48;
            const outside =
              pt.x < vb.x - margin ||
              pt.y < vb.y - margin ||
              pt.x > vb.x + vb.w + margin ||
              pt.y > vb.y + vb.h + margin;
            if (outside) {
              findings.push({
                severity: "warn",
                deck: route,
                slide: String(slide),
                code: "dot-out-of-viewbox",
                message: `motion dot at (${pt.x.toFixed(1)},${pt.y.toFixed(1)}) outside viewBox`,
              });
            }
          } catch {
            // CTM inverse can fail on detached nodes; skip.
          }
        }
      }

      for (const el of nodes) {
        const kind = el.getAttribute("data-flow-kind") ?? "";
        if (
          kind === "path" ||
          kind === "line" ||
          kind === "arrow" ||
          kind === "connector"
        ) {
          continue;
        }
        const box = el.getBoundingClientRect();
        const svgBox = svg.getBoundingClientRect();
        if (box.width <= 0 || box.height <= 0) continue;
        const outside =
          box.right < svgBox.left - 40 ||
          box.left > svgBox.right + 40 ||
          box.bottom < svgBox.top - 40 ||
          box.top > svgBox.bottom + 40;
        if (outside) {
          findings.push({
            severity: "warn",
            deck: route,
            slide: String(slide),
            code: "dom-out-of-stage",
            message: `node ${el.getAttribute("data-flow-node-id")} renders outside SVG stage`,
          });
        }
      }

      if (!(vb.w > 0 && vb.h > 0)) {
        findings.push({
          severity: "error",
          deck: route,
          slide: String(slide),
          code: "bad-viewbox",
          message: "SVG viewBox has non-positive size",
        });
      }
      return findings;
    },
    { route: deckRoute, slide: slideIndex, rootSelector },
  );
}

/**
 * finalCard scene assertions (G2 for Play): `ExplainerShell` mounts `.explainer-final-card`
 * only on the last slide when the DeckPackage authored one. Scene-only cards mount
 * `SceneRenderer` directly; chrome-only cards (title/summary/cta with no scene) are valid
 * and have nothing to sample. Absence of the region entirely is also valid — most decks
 * have no finalCard.
 */
async function collectFinalCardFindings(page, deckRoute) {
  const probe = await page.evaluate(() => {
    const region = document.querySelector(".explainer-final-card");
    if (!region) return { present: false, hasSvg: false };
    return { present: true, hasSvg: region.querySelector("svg") != null };
  });
  if (!probe.present || !probe.hasSvg) return [];
  return await collectSvgFindings(page, deckRoute, "finalCard", {
    rootSelector: ".explainer-final-card",
  });
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

async function goNext(page) {
  // ExplainerShell's bottom-nav control is "Next →" (arrow glyph included in
  // the accessible name); anchoring the full string missed the button
  // entirely and silently truncated every route to slide 0. Match the prefix
  // instead so a glyph/label tweak doesn't do the same again.
  const next = page.getByRole("button", { name: /^Next/i });
  if (await next.count()) {
    const disabled = await next.first().isDisabled().catch(() => true);
    if (!disabled) {
      await next.first().click();
      return true;
    }
  }
  return false;
}

async function sampleWhilePlaying(page, deckRoute, slideIndex, findings) {
  // Two samples spaced so a draw cue mid-flight is likely caught.
  findings.push(...(await collectSvgFindings(page, deckRoute, slideIndex)));
  await page.waitForTimeout(350);
  findings.push(...(await collectSvgFindings(page, deckRoute, slideIndex)));
}

/**
 * Attach console/runtime error capture to a page. Browser-side errors (a
 * failed live `.flow` compile, an uncaught SceneRenderer exception, a React
 * warning promoted to `console.error`, ...) would otherwise pass silently —
 * the SVG-shape assertions can't see them. `slideRef` is a mutable box so
 * findings recorded asynchronously (console/pageerror fire off the main
 * await chain) still land against the slide being visited at the time.
 */
function attachErrorCapture(page, route, slideRef, findings) {
  page.on("console", (msg) => {
    if (msg.type() !== "error") return;
    findings.push({
      severity: "error",
      deck: route,
      slide: slideRef.label,
      code: "console-error",
      message: msg.text(),
    });
  });
  page.on("pageerror", (error) => {
    findings.push({
      severity: "error",
      deck: route,
      slide: slideRef.label,
      code: "page-error",
      message: String(error?.message ?? error),
    });
  });
}

/**
 * Play every deck route and collect live SVG findings.
 *
 * @param {{ deckRoute?: string; baseUrl?: string; preview?: boolean }} options
 *   `preview: true` gates against `npm run build && npm run preview` instead
 *   of `vite dev` — see `startViteServer` for why this isn't the default.
 */
export async function verifyPlayAll(options = {}) {
  const findings = [];
  const deckFilter = options.deckRoute ?? null;
  const routes = deckFilter
    ? DECK_ROUTES.filter((r) =>
        r.includes(deckFilter.replace(/^\/?#?\/?/, "")),
      )
    : DECK_ROUTES;

  let stop = null;
  let baseUrl = options.baseUrl ?? null;
  if (!baseUrl) {
    const server = await startViteServer({ usePreview: options.preview });
    baseUrl = server.baseUrl;
    stop = server.stop;
  }

  const chromium = await loadChromium();
  // Force full motion so arrowhead-deferral checks are meaningful.
  const browser = await chromium.launch({ headless: true });
  try {
    for (const route of routes) {
      const page = await browser.newPage();
      await page.emulateMedia({ reducedMotion: "no-preference" });
      const slideRef = { label: "pre-play" };
      attachErrorCapture(page, route, slideRef, findings);
      try {
        await page.goto(`${baseUrl}${route}`, {
          waitUntil: "networkidle",
          timeout: 60_000,
        });
        await page.waitForTimeout(500);
        await dismissStartGate(page);
        await page
          .getByRole("button", { name: PLAYBACK_TOGGLE_ANY_RE })
          .first()
          .waitFor({ timeout: 15_000 })
          .catch(() => {});
        await clickPlay(page).catch((error) => {
          findings.push({
            severity: "error",
            deck: route,
            slide: "*",
            code: "play-failed",
            message: String(error.message ?? error),
          });
        });

        const total = (await slideCount(page)) ?? 1;
        for (let i = 0; i < total; i += 1) {
          slideRef.label = String(i);
          await sampleWhilePlaying(page, route, i, findings);
          if (i === total - 1) {
            slideRef.label = "finalCard";
            findings.push(...(await collectFinalCardFindings(page, route)));
          } else {
            const moved = await goNext(page);
            if (!moved) {
              // A silent break here would under-report slide coverage as if
              // the deck were shorter than it is; surface it instead.
              findings.push({
                severity: "error",
                deck: route,
                slide: slideRef.label,
                code: "next-unavailable",
                message: `Next control missing/disabled at slide ${i} of ${total}; walk stopped short`,
              });
              break;
            }
            await clickPlay(page).catch(() => {});
          }
        }
      } catch (error) {
        findings.push({
          severity: "error",
          deck: route,
          slide: slideRef.label,
          code: "play-exception",
          message: String(error.message ?? error),
        });
      } finally {
        await page.close().catch(() => {});
      }
    }
  } finally {
    await browser.close().catch(() => {});
    if (stop) stop();
  }

  return findings;
}

export { DECK_ROUTES };
