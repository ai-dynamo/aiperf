// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { expect, test, type Locator, type Page } from "@playwright/test";

const desktopViewport = { width: 1440, height: 900 } as const;
const mobileViewport = { width: 390, height: 844 } as const;
const referenceViewport = { width: 3840, height: 2160 } as const;

test.use({
  colorScheme: "dark",
  deviceScaleFactor: 1,
  locale: "en-US",
  reducedMotion: "no-preference",
  timezoneId: "UTC",
  viewport: desktopViewport,
});

function sceneField(page: Page): Locator {
  return page.getByRole("region", { name: "Scene field" });
}

function causalPath(page: Page): Locator {
  return page.getByRole("navigation", { name: /causal (?:path|replay)/iu });
}

function semanticTwin(page: Page): Locator {
  return page.getByRole("region", { name: "Semantic outline" });
}

function commandDialog(page: Page): Locator {
  return page.getByRole("dialog", { name: "Command Constellation" });
}

async function openPreview(
  page: Page,
  options: Readonly<{
    search?: string;
    canvas?: boolean;
    reducedMotion?: boolean;
    forcedColors?: boolean;
  }> = {},
): Promise<void> {
  if (options.canvas === false) {
    await page.addInitScript(() => {
      HTMLCanvasElement.prototype.getContext = () => null;
    });
  }
  await page.emulateMedia({
    forcedColors: options.forcedColors === true ? "active" : "none",
    reducedMotion: options.reducedMotion === true ? "reduce" : "no-preference",
  });
  await page.goto(`/${options.search ?? ""}`);

  const withoutAudio = page.getByRole("button", { name: "Play without audio" });
  if (await withoutAudio.isVisible()) {
    await withoutAudio.click();
  }
  const pause = page.getByRole("button", { name: /^Pause(?: scene)?$/u });
  if (await pause.isVisible()) {
    await pause.click();
  }

  await expect(sceneField(page)).toBeVisible();
  await expect(causalPath(page)).toBeVisible();
  await expect(semanticTwin(page)).toBeAttached();
  await page.evaluate(async () => document.fonts.ready);
}

async function openCommands(page: Page): Promise<Locator> {
  await page.keyboard.press("ControlOrMeta+K");
  const dialog = commandDialog(page);
  await expect(dialog).toBeVisible();
  return dialog;
}

async function selectBeat(page: Page, index: number): Promise<Locator> {
  const beat = causalPath(page).getByRole("button").nth(index);
  await beat.click();
  await expect(beat).toHaveAttribute("aria-current", "step");
  return beat;
}

async function currentState(page: Page): Promise<unknown> {
  return page.evaluate(() => {
    const path = document.querySelector(
      '[aria-label="Causal path"], [aria-label="Causal Replay"]',
    );
    const twin = document.querySelector('[aria-label="Semantic outline"]');
    const field = document.querySelector('[aria-label="Scene field"]');
    const currentBeat = path?.querySelector(
      '[aria-current="step"], [data-state="active"]',
    );
    return {
      backend: field?.getAttribute("data-backend"),
      beatId: currentBeat?.getAttribute("data-beat-id"),
      sceneId: twin?.getAttribute("data-scene-id"),
      entities: [...(twin?.querySelectorAll("[data-entity-id]") ?? [])].map(
        (entity) => ({
          id: entity.getAttribute("data-entity-id"),
          selected: entity.getAttribute("aria-selected"),
        }),
      ),
      relations: [...(twin?.querySelectorAll("[data-relation-id]") ?? [])].map(
        (relation) => relation.getAttribute("data-relation-id"),
      ),
    };
  });
}

async function openContextLens(page: Page): Promise<Locator> {
  const entity = semanticTwin(page).locator("button[data-entity-id]").first();
  const entityId = await entity.getAttribute("data-entity-id");
  expect(entityId).not.toBeNull();
  await entity.click();
  const lens = page.getByRole("region", { name: "Context Lens" });
  await expect(lens).toBeVisible();
  await expect(lens).toHaveAttribute("data-entity-id", entityId!);
  return lens;
}

async function freezeBrowserTime(page: Page): Promise<void> {
  await page.clock.install({ time: new Date("2026-01-01T00:00:00.000Z") });
  await page.addInitScript(() => {
    let state = 0x9e3779b9;
    Math.random = () => {
      state = Math.imul(state ^ (state >>> 16), 0x21f0aaad);
      state = Math.imul(state ^ (state >>> 15), 0x735a2d97);
      return ((state ^= state >>> 15) >>> 0) / 0x1_0000_0000;
    };
  });
}

test.beforeEach(async ({ page }) => {
  await freezeBrowserTime(page);
});

test("causal replay makes direct seek equal continuous playback", async ({
  page,
}) => {
  await openPreview(page);
  const beats = causalPath(page).getByRole("button");
  expect(await beats.count()).toBeGreaterThan(1);

  const target = beats.nth(1);
  const targetTimeAttribute = await target.getAttribute("data-time-ms");
  expect(targetTimeAttribute).not.toBeNull();
  const targetTime = Number(targetTimeAttribute);
  expect(Number.isSafeInteger(targetTime)).toBe(true);
  await target.click();
  const directlySeeked = await currentState(page);

  await beats.first().click();
  await page.getByRole("button", { name: /^Play(?: scene)?$/u }).click();
  await page.clock.runFor(targetTime);
  await expect(target).toHaveAttribute("aria-current", "step");
  expect(await currentState(page)).toEqual(directlySeeked);
});

test("command search executes actions and restores invoking focus", async ({
  page,
}) => {
  await openPreview(page);
  const trigger = page.getByRole("button", { name: /open commands/iu });
  await trigger.focus();
  await trigger.click();

  const dialog = commandDialog(page);
  const search = dialog.getByRole("searchbox", { name: /search commands/iu });
  await expect(search).toBeFocused();
  await page.keyboard.press("ArrowDown");
  await expect(dialog.getByRole("option").nth(1)).toHaveAttribute(
    "aria-selected",
    "true",
  );
  await search.fill("last beat");
  await expect(dialog.getByRole("option").first()).toContainText(/last beat/iu);
  await dialog.getByRole("option").first().click();
  await expect(causalPath(page).getByRole("button").last()).toHaveAttribute(
    "aria-current",
    "step",
  );

  await trigger.click();
  await page.keyboard.press("Escape");
  await expect(dialog).toBeHidden();
  await expect(trigger).toBeFocused();

  const entityDialog = await openCommands(page);
  await entityDialog
    .getByRole("searchbox", { name: /search commands/iu })
    .fill("Worker sink");
  await entityDialog.getByRole("option").first().click();
  await expect(
    semanticTwin(page).getByRole("button", { name: "Worker sink" }),
  ).toHaveAttribute("aria-selected", "true");

  const twinDialog = await openCommands(page);
  await twinDialog
    .getByRole("searchbox", { name: /search commands/iu })
    .fill("semantic twin");
  await twinDialog.getByRole("option").first().click();
  await expect(semanticTwin(page)).toBeVisible();
});

test("Context Lens and Focus World share selection and restore the beat", async ({
  page,
}) => {
  await openPreview(page);
  const beat = await selectBeat(page, 1);
  const beatId = await beat.getAttribute("data-beat-id");
  expect(beatId).not.toBeNull();
  const lens = await openContextLens(page);
  const selectedId = await lens.getAttribute("data-entity-id");
  expect(selectedId).not.toBeNull();

  await lens.getByRole("button", { name: /focus world/iu }).click();
  const focusWorld = page.getByRole("region", { name: "Focus World" });
  await expect(focusWorld).toBeVisible();
  await expect(focusWorld).toHaveAttribute("data-entity-id", selectedId!);
  await expect(
    semanticTwin(page).locator(`[data-entity-id="${selectedId!}"]`),
  ).toHaveAttribute("aria-selected", "true");

  await page.keyboard.press("Escape");
  await expect(focusWorld).toBeHidden();
  await expect(beat).toHaveAttribute("data-beat-id", beatId!);
  await expect(beat).toHaveAttribute("aria-current", "step");
});

test("fullscreen falls back to layout mode when the API is unavailable", async ({
  page,
}) => {
  await page.addInitScript(() => {
    Object.defineProperty(Document.prototype, "fullscreenEnabled", {
      configurable: true,
      get: () => false,
    });
    Object.defineProperty(Element.prototype, "requestFullscreen", {
      configurable: true,
      value: undefined,
    });
  });
  await openPreview(page);

  await page.getByRole("button", { name: /enter fullscreen/iu }).click();
  expect(
    await sceneField(page).evaluate(
      (field) =>
        field.getAttribute("data-fullscreen") ??
        field.getAttribute("data-fullscreen-state"),
    ),
  ).toBe("layout");
  await expect(page.locator(":fullscreen")).toHaveCount(0);
  await expect(causalPath(page)).toBeVisible();
});

test("restores valid scene, beat, and selected entity from the URL", async ({
  page,
}) => {
  await openPreview(page, {
    search: "?scene=execution&beat=trace-dispatch&entity=worker",
  });

  await expect(semanticTwin(page)).toHaveAttribute("data-scene-id", "execution");
  await expect(
    causalPath(page).locator('[data-beat-id="trace-dispatch"]'),
  ).toHaveAttribute("aria-current", "step");
  await expect(
    semanticTwin(page).locator('[data-entity-id="worker"]'),
  ).toHaveAttribute("aria-selected", "true");
  const restored = new URL(page.url()).searchParams;
  expect(restored.get("scene")).toBe("execution");
  expect(restored.get("beat")).toBe("trace-dispatch");
  expect(restored.get("entity")).toBe("worker");
});

test("supports keyboard traversal through beats and semantic entities", async ({
  page,
}) => {
  await openPreview(page);
  const beats = causalPath(page).getByRole("button");
  await beats.first().focus();
  await page.keyboard.press("End");
  await expect(beats.last()).toBeFocused();
  await expect(beats.last()).toHaveAttribute("aria-current", "step");
  await page.keyboard.press("Home");
  await expect(beats.first()).toBeFocused();
  await page.keyboard.press("ArrowRight");
  await expect(beats.nth(1)).toBeFocused();

  const entities = semanticTwin(page).locator("button[data-entity-id]");
  await entities.first().focus();
  await page.keyboard.press("ArrowDown");
  await expect(entities.nth(1)).toBeFocused();
  await page.keyboard.press("Enter");
  await expect(entities.nth(1)).toHaveAttribute("aria-selected", "true");
});

test("captions remain visible and synchronized at a named cue", async ({
  page,
}) => {
  await openPreview(page);
  const namedBeat = causalPath(page)
    .locator('[data-beat-id="trace-dispatch"], [data-beat-id="dispatch"]')
    .first();
  await namedBeat.click();

  const subtitles = page.getByRole("region", { name: "Subtitles" });
  await expect(subtitles).toBeVisible();
  await expect(subtitles).toHaveAttribute("data-cue-id", /\S/u);
  const cueId = await subtitles.getAttribute("data-cue-id");
  expect(cueId).not.toBeNull();
  await expect(semanticTwin(page).getByRole("status")).toHaveAttribute(
    "data-transcript-cue",
    cueId!,
  );

  await subtitles.getByRole("button", { name: /turn subtitles off/iu }).click();
  await expect(subtitles).not.toHaveAttribute("data-cue-id", /\S/u);
  await subtitles.getByRole("button", { name: /turn subtitles on/iu }).click();
  await expect(subtitles).toHaveAttribute("data-cue-id", cueId);
});

test("keeps the semantic twin mounted and aligned with selection", async ({
  page,
}) => {
  await openPreview(page);
  const twin = semanticTwin(page);
  await expect(twin).not.toHaveAttribute("aria-hidden", "true");
  await expect(twin.locator("[data-entity-id]")).not.toHaveCount(0);
  await expect(twin.locator("[data-relation-id]")).not.toHaveCount(0);

  const entity = twin.locator("button[data-entity-id]").first();
  const entityId = await entity.getAttribute("data-entity-id");
  expect(entityId).not.toBeNull();
  await entity.click();
  await expect(entity).toHaveAttribute("aria-selected", "true");
  await expect(sceneField(page)).toHaveAttribute("data-selected-entity-id", entityId!);
});

test("SVG fallback preserves replay, semantics, selection, and commands", async ({
  page,
}) => {
  await openPreview(page, { canvas: false });
  await expect(sceneField(page)).toHaveAttribute("data-backend", "svg");
  await expect(sceneField(page).locator("svg.aiperf-flow__svg-fallback")).toBeVisible();
  await expect(page.locator("canvas")).toHaveCount(0);

  const entity = semanticTwin(page).locator("button[data-entity-id]").first();
  await entity.click();
  await expect(entity).toHaveAttribute("aria-selected", "true");
  await expect(causalPath(page)).toBeVisible();
  await expect(await openCommands(page)).toBeVisible();
});

test("Flow browser opens as an overlay without resizing the scene", async ({
  page,
}) => {
  await openPreview(page);
  const before = await sceneField(page).boundingBox();
  expect(before).not.toBeNull();

  await page.getByRole("button", { name: "Open Flow browser" }).click();
  const drawer = page.getByRole("complementary", { name: "Flow browser" });
  await expect(drawer).toBeVisible();
  await expect(drawer).toHaveAttribute("data-overlay", "true");
  expect(await sceneField(page).boundingBox()).toEqual(before);

  await drawer.getByRole("button", { name: /close|collapse/iu }).click();
  await expect(drawer).toBeHidden();
});

test("mobile controls and command sheet do not create horizontal overflow", async ({
  page,
}) => {
  await page.setViewportSize(mobileViewport);
  await openPreview(page);
  await expect
    .poll(() =>
      page.evaluate(
        () => document.documentElement.scrollWidth <= window.innerWidth,
      ),
    )
    .toBe(true);

  const dialog = await openCommands(page);
  const box = await dialog.boundingBox();
  expect(box).not.toBeNull();
  expect(box!.x).toBe(0);
  expect(box!.width).toBe(mobileViewport.width);
  expect(box!.height).toBeGreaterThanOrEqual(mobileViewport.height - 1);
  await expect(causalPath(page)).toBeAttached();
});

type ScreenshotCase = Readonly<{
  name: string;
  viewport: Readonly<{ width: number; height: number }>;
  open?: Parameters<typeof openPreview>[1];
  prepare?(page: Page): Promise<void>;
}>;

const screenshotMatrix: readonly ScreenshotCase[] = [
  {
    name: "01-desktop-playing.png",
    viewport: desktopViewport,
    prepare: async (page) => {
      const play = page.getByRole("button", { name: /^Play(?: scene)?$/u });
      if (await play.isVisible()) {
        await play.click();
      }
      await expect(
        page.getByRole("button", { name: /^Pause(?: scene)?$/u }),
      ).toBeVisible();
    },
  },
  {
    name: "02-desktop-browser-drawer.png",
    viewport: desktopViewport,
    prepare: async (page) => {
      await page.getByRole("button", { name: "Open Flow browser" }).click();
    },
  },
  {
    name: "03-desktop-context-lens.png",
    viewport: desktopViewport,
    prepare: async (page) => {
      await openContextLens(page);
    },
  },
  {
    name: "04-desktop-focus-world.png",
    viewport: desktopViewport,
    prepare: async (page) => {
      const lens = await openContextLens(page);
      await lens.getByRole("button", { name: /focus world/iu }).click();
    },
  },
  {
    name: "05-desktop-command-constellation.png",
    viewport: desktopViewport,
    prepare: async (page) => {
      await openCommands(page);
    },
  },
  {
    name: "06-desktop-named-cue-captions.png",
    viewport: desktopViewport,
    prepare: async (page) => {
      await causalPath(page)
        .locator('[data-beat-id="trace-dispatch"], [data-beat-id="dispatch"]')
        .first()
        .click();
      await expect(page.getByRole("region", { name: "Subtitles" })).toBeVisible();
    },
  },
  {
    name: "07-mobile-playing.png",
    viewport: mobileViewport,
    prepare: async (page) => {
      const play = page.getByRole("button", { name: /^Play(?: scene)?$/u });
      if (await play.isVisible()) {
        await play.click();
      }
    },
  },
  {
    name: "08-mobile-command-sheet.png",
    viewport: mobileViewport,
    prepare: async (page) => {
      await openCommands(page);
    },
  },
  {
    name: "09-svg-fallback.png",
    viewport: desktopViewport,
    open: { canvas: false },
  },
  {
    name: "10-reduced-motion.png",
    viewport: desktopViewport,
    open: { reducedMotion: true },
  },
  {
    name: "11-forced-colors.png",
    viewport: desktopViewport,
    open: { forcedColors: true },
  },
  {
    name: "12-reference-3840x2160.png",
    viewport: referenceViewport,
  },
];

for (const screenshot of screenshotMatrix) {
  test(`matches deterministic immersive state: ${screenshot.name}`, async ({
    page,
  }) => {
    await page.setViewportSize(screenshot.viewport);
    await openPreview(page, screenshot.open);
    await screenshot.prepare?.(page);
    await expect(page).toHaveScreenshot(screenshot.name, {
      animations: "disabled",
      caret: "hide",
      fullPage: false,
      scale: "css",
    });
  });
}
