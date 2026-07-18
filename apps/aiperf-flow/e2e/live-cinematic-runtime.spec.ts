// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { execFile } from "node:child_process";
import {
  copyFile,
  mkdtemp,
  readFile,
  rm,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { extname, join, relative, resolve, sep } from "node:path";
import { promisify } from "node:util";

import { expect, test, type Page } from "@playwright/test";

import {
  clearRuntimeMetrics,
  collectRuntimeMetrics,
} from "./helpers/runtime-metrics.js";

const execFileAsync = promisify(execFile);
const e2eDir = __dirname;
const flowRoot = resolve(e2eDir, "..");
const fixturePath = join(e2eDir, "fixtures", "cinematic-foundation.flow");
const cliPath = join(flowRoot, "packages", "cli", "dist", "main.js");
const siteBundlePath = join(flowRoot, "packages", "runtime", "dist", "site.js");
const themePath = join(flowRoot, "packages", "runtime", "src", "theme.css");
const origin = "http://cinematic.test";

const contentTypes: Readonly<Record<string, string>> = {
  ".css": "text/css; charset=utf-8",
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".map": "application/json; charset=utf-8",
  ".txt": "text/plain; charset=utf-8",
};

type SemanticSnapshot = Readonly<{
  sceneId: string | null;
  entities: readonly Readonly<{
    id: string | null;
    label: string | null;
    kind: string | null;
  }>[];
  relations: readonly Readonly<{
    id: string | null;
    from: string | null;
    to: string | null;
  }>[];
}>;

type CinematicBrowserSnapshot = Readonly<{
  semantics: SemanticSnapshot;
  activeBeatId: string | null;
  backend: string | null;
  subtitleCueId: string | null;
}>;

let staticSiteRoot: string | undefined;

test.use({
  launchOptions:
    process.env.PLAYWRIGHT_EXECUTABLE_PATH === undefined
      ? {}
      : { executablePath: process.env.PLAYWRIGHT_EXECUTABLE_PATH },
});

async function buildStaticSite(): Promise<string> {
  if (staticSiteRoot !== undefined) {
    return staticSiteRoot;
  }

  const root = await mkdtemp(join(tmpdir(), "aiperf-flow-cinematic-"));
  try {
    await execFileAsync(
      process.execPath,
      [cliPath, "build", fixturePath, "--out", root],
      { cwd: flowRoot },
    );
    await Promise.all([
      copyFile(siteBundlePath, join(root, "site.js")),
      copyFile(themePath, join(root, "theme.css")),
      writeFile(
        join(root, "index.html"),
        `<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Cinematic foundation</title>
    <link rel="stylesheet" href="./theme.css" />
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="./site.js"></script>
  </body>
</html>
`,
        "utf8",
      ),
    ]);
  } catch (error) {
    await rm(root, { recursive: true, force: true });
    throw error;
  }
  staticSiteRoot = root;
  return root;
}

async function routeStaticSite(page: Page, root: string): Promise<void> {
  await page.route(`${origin}/**`, async (route) => {
    const url = new URL(route.request().url());
    const requestPath =
      url.pathname === "/" ? "index.html" : decodeURIComponent(url.pathname.slice(1));
    const filePath = resolve(root, requestPath);
    const fromRoot = relative(root, filePath);
    if (
      fromRoot === ".." ||
      fromRoot.startsWith(`..${sep}`) ||
      fromRoot.includes(`..${sep}`)
    ) {
      await route.fulfill({ status: 400, body: "Invalid path" });
      return;
    }
    try {
      await route.fulfill({
        status: 200,
        contentType: contentTypes[extname(filePath)] ?? "application/octet-stream",
        body: await readFile(filePath),
      });
    } catch {
      await route.fulfill({ status: 404, body: "Not found" });
    }
  });
}

async function openCinematicRuntime(
  page: Page,
  options: Readonly<{
    canvas?: boolean;
    forcedColors?: "active" | "none";
    reducedMotion?: "no-preference" | "reduce";
    reducedTransparency?: boolean;
  }> = {},
): Promise<void> {
  await page.emulateMedia({
    forcedColors: options.forcedColors ?? "none",
    reducedMotion: options.reducedMotion ?? "no-preference",
  });
  if (options.reducedTransparency === true) {
    await page.addInitScript(() => {
      const browserMatchMedia = window.matchMedia.bind(window);
      window.matchMedia = (query) => {
        const mediaQuery = browserMatchMedia(query);
        if (query !== "(prefers-reduced-transparency: reduce)") {
          return mediaQuery;
        }
        return new Proxy(mediaQuery, {
          get(target, property) {
            if (property === "matches") {
              return true;
            }
            const value = Reflect.get(target, property, target);
            return typeof value === "function" ? value.bind(target) : value;
          },
        });
      };
    });
  }
  if (options.canvas === false) {
    await page.addInitScript(() => {
      HTMLCanvasElement.prototype.getContext = () => null;
    });
  }
  await routeStaticSite(page, await buildStaticSite());
  await page.goto(origin);
  await expect(
    page.getByRole("heading", { level: 1, name: "Request execution" }),
  ).toBeVisible();
  await expect(page.getByRole("region", { name: "Scene field" })).toBeVisible();
  await expect(page.getByRole("navigation", { name: "Causal path" })).toBeVisible();
  await expect(
    page.getByRole("region", { name: "Immersive controls" }),
  ).toBeVisible();
  await expect(page.getByRole("region", { name: "Scene stage" })).toBeVisible();
  await expect(page.getByRole("region", { name: "Semantic outline" })).toBeVisible();
  await expect(page.locator('input[type="range"]')).toHaveCount(0);
}

async function semanticSnapshot(page: Page): Promise<SemanticSnapshot> {
  return page.getByRole("region", { name: "Semantic outline" }).evaluate((outline) => ({
    sceneId: outline.getAttribute("data-scene-id"),
    entities: [...outline.querySelectorAll<HTMLElement>("[data-entity-id]")].map(
      (entity) => ({
        id: entity.getAttribute("data-entity-id"),
        label: entity.getAttribute("aria-label"),
        kind: entity.getAttribute("data-kind"),
      }),
    ),
    relations: [
      ...outline.querySelectorAll<HTMLElement>("[data-relation-id]"),
    ].map((relationElement) => ({
      id: relationElement.getAttribute("data-relation-id"),
      from: relationElement.getAttribute("data-from"),
      to: relationElement.getAttribute("data-to"),
    })),
  }));
}

async function cinematicBrowserSnapshot(
  page: Page,
): Promise<CinematicBrowserSnapshot> {
  const activeBeat = page
    .getByRole("navigation", { name: "Causal path" })
    .locator('[aria-current="step"]');
  const stage = page.getByRole("region", { name: "Scene stage" });
  const subtitles = page.getByRole("region", { name: "Subtitles" });

  return {
    semantics: await semanticSnapshot(page),
    activeBeatId: await activeBeat.getAttribute("data-beat-id"),
    backend: await stage.getAttribute("data-backend"),
    subtitleCueId: await subtitles.getAttribute("data-cue-id"),
  };
}

async function preparePausedRuntime(page: Page): Promise<void> {
  const playWithoutAudio = page.getByRole("button", {
    name: "Play without audio",
  });
  if (await playWithoutAudio.isVisible()) {
    await playWithoutAudio.click();
  }
  const pause = page.getByRole("button", { name: "Pause", exact: true });
  if (await pause.isVisible()) {
    await pause.click();
  }
  await expect(page.getByRole("button", { name: "Play", exact: true })).toBeEnabled();
}

test.describe("live cinematic runtime", () => {
  test.describe.configure({ mode: "serial" });

  test.afterAll(async () => {
    if (staticSiteRoot !== undefined) {
      await rm(staticSiteRoot, { recursive: true, force: true });
      staticSiteRoot = undefined;
    }
  });

  test("keeps the stage and semantic meaning available at reference and compact sizes", async ({
    page,
  }) => {
    const viewports = [
      { width: 3840, height: 2160 },
      { width: 1920, height: 1080 },
      { width: 1024, height: 768 },
      { width: 390, height: 844 },
    ] as const;
    let referenceSemantics: SemanticSnapshot | undefined;

    for (const viewport of viewports) {
      await page.setViewportSize(viewport);
      await openCinematicRuntime(page);

      const semantics = await semanticSnapshot(page);
      referenceSemantics ??= semantics;
      expect(semantics).toEqual(referenceSemantics);
      await expect(page.getByRole("region", { name: "Playback controls" })).toBeVisible();
      await expect(page.getByRole("region", { name: "Narration transcript" })).toBeVisible();
      await expect(page.locator('[data-backend="canvas"]').first()).toBeVisible();
    }
  });

  test("exposes keyboard-operable reading order, relations, and transcript access", async ({
    page,
  }) => {
    await openCinematicRuntime(page);
    const outline = page.getByRole("region", { name: "Semantic outline" });
    const entityButtons = outline.getByRole("button");

    await expect(entityButtons).toHaveCount(2);
    await expect(entityButtons.nth(0)).toHaveAttribute("data-entity-id", "admission");
    await expect(entityButtons.nth(1)).toHaveAttribute("data-entity-id", "worker");
    await expect(outline.getByRole("list", { name: "Relations" })).toContainText(
      "dispatch request",
    );

    await page.keyboard.press("Tab");
    const skipLink = page.getByRole("link", { name: "Skip to transcript" });
    await expect(skipLink).toBeFocused();
    await page.keyboard.press("Enter");
    await expect(
      page.getByRole("region", { name: "Narration transcript" }),
    ).toBeFocused();

    const worker = entityButtons.nth(1);
    await worker.focus();
    await page.keyboard.press("Enter");
    await expect(worker).toHaveAttribute("aria-selected", "true");
    await expect(worker).toHaveAttribute("data-focused", "true");
  });

  test("preserves semantic content when reduced motion and forced colors are active", async ({
    page,
  }) => {
    await openCinematicRuntime(page);
    const referenceSemantics = await semanticSnapshot(page);

    await page.close();
    const variantPage = await page.context().newPage();
    await openCinematicRuntime(variantPage, {
      forcedColors: "active",
      reducedMotion: "reduce",
    });

    expect(await semanticSnapshot(variantPage)).toEqual(referenceSemantics);
    expect(
      await variantPage.evaluate(
        () =>
          matchMedia("(prefers-reduced-motion: reduce)").matches &&
          matchMedia("(forced-colors: active)").matches,
      ),
    ).toBe(true);
    await expect(
      variantPage.getByRole("button", { name: "Play", exact: true }),
    ).toBeVisible();
    await expect(
      variantPage.getByRole("region", { name: "Narration transcript" }),
    ).toContainText("Admission follows the shared clock");
  });

  test("uses the SVG fallback without losing controls, semantics, or selection", async ({
    page,
  }) => {
    await openCinematicRuntime(page, { canvas: false });

    await expect(page.getByRole("region", { name: "Scene stage" })).toHaveAttribute(
      "data-backend",
      "svg",
    );
    await expect(page.locator(".aiperf-flow__svg-fallback")).toBeVisible();
    await expect(page.locator("canvas")).toHaveCount(0);
    expect(await semanticSnapshot(page)).toMatchObject({
      sceneId: "request-execution",
      entities: [
        { id: "admission", label: "Admission queue" },
        { id: "worker", label: "Worker sink" },
      ],
      relations: [{ id: "dispatch", from: "admission", to: "worker" }],
    });

    const admission = page
      .getByRole("region", { name: "Semantic outline" })
      .getByRole("button", { name: "Admission queue" });
    await admission.focus();
    await page.keyboard.press("Enter");
    await expect(admission).toHaveAttribute("aria-selected", "true");
    await expect(page.getByRole("region", { name: "Playback controls" })).toBeVisible();
    await expect(
      page.getByRole("region", { name: "Narration transcript" }),
    ).toBeVisible();
  });

  test("direct seek and continuous playback produce equal browser state", async ({
    page,
  }) => {
    await openCinematicRuntime(page);
    await preparePausedRuntime(page);
    const causalPath = page.getByRole("navigation", { name: "Causal path" });
    const beats = causalPath.getByRole("button");
    await expect(beats).toHaveCount(3);

    const targetBeat = beats.last();
    await expect(targetBeat).toHaveAttribute("data-beat-id", /\S/u);
    const targetBeatId = await targetBeat.getAttribute("data-beat-id");
    await targetBeat.click();
    await expect(targetBeat).toHaveAttribute("aria-current", "step");
    const directState = await cinematicBrowserSnapshot(page);
    expect(directState.activeBeatId).toBe(targetBeatId);

    await page.getByRole("button", { name: "Restart" }).click();
    await page.getByRole("button", { name: "Play", exact: true }).click();
    await expect
      .poll(async () =>
        causalPath
          .locator('[aria-current="step"]')
          .getAttribute("data-beat-id"),
      )
      .toBe(targetBeatId);
    await page.getByRole("button", { name: "Pause", exact: true }).click();

    expect(await cinematicBrowserSnapshot(page)).toEqual(directState);
  });

  test("applies reduced-transparency and no-depth quality variants", async ({
    page,
  }) => {
    await openCinematicRuntime(page, {
      forcedColors: "active",
      reducedMotion: "reduce",
      reducedTransparency: true,
    });
    const sceneField = page.getByRole("region", { name: "Scene field" });

    await expect(sceneField).toHaveAttribute("data-transparency", "reduced");
    await expect(sceneField).toHaveAttribute("data-depth", "none");
    await expect(page.getByRole("region", { name: "Semantic outline" })).toBeVisible();
    await expect(page.getByRole("navigation", { name: "Causal path" })).toBeVisible();
    await expect(
      page.getByRole("region", { name: "Narration transcript" }),
    ).toBeVisible();
  });

  test("reports evaluation, draw, and total frame metrics", async ({ page }) => {
    await openCinematicRuntime(page);
    await preparePausedRuntime(page);
    await clearRuntimeMetrics(page);
    await page.getByRole("button", { name: "Play", exact: true }).click();
    const samples = await collectRuntimeMetrics(page, { sampleCount: 2 });
    await page.getByRole("button", { name: "Pause", exact: true }).click();

    expect(samples).toHaveLength(2);
    for (const sample of samples) {
      expect(
        [sample.evaluationMs, sample.drawMs, sample.totalMs].every(
          (duration) => Number.isFinite(duration) && duration >= 0,
        ),
      ).toBe(true);
      expect(sample.totalMs).toBeGreaterThanOrEqual(sample.evaluationMs);
      expect(sample.totalMs).toBeGreaterThanOrEqual(sample.drawMs);
    }
  });
});
