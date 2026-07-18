// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Locator, type Page } from "@playwright/test";

const seriousOrCritical = new Set(["serious", "critical"]);
const screenshotOptions = {
  animations: "disabled",
  caret: "hide",
  maxDiffPixels: 600,
  scale: "css",
} as const;

function withQuery(path: string, query: string): string {
  return query.trim().length === 0 ? path : `${path}?${query}`;
}

function sceneLocator(page: Page): Locator {
  return page.locator(".graph-scene-route");
}

async function expectProductionLayoutReady(page: Page) {
  await expect(page.getByRole("status", { name: "Graph layout status" })).toHaveText(
    "Graph layout ready.",
  );
}

async function disableNondeterministicMotion(page: Page) {
  await page.addStyleTag({
    content: `
      *, *::before, *::after {
        animation: none !important;
        animation-delay: 0s !important;
        animation-duration: 0s !important;
        transition: none !important;
        transition-delay: 0s !important;
        transition-duration: 0s !important;
        scroll-behavior: auto !important;
      }
    `,
  });
}

async function openFlightDeck(
  page: Page,
  input: {
    path?: string;
    query?: string;
    reducedMotion?: "no-preference" | "reduce";
    forcedColors?: "none" | "active";
    fitGraph?: boolean;
    extraZoomOut?: boolean;
  } = {},
) {
  await page.emulateMedia({
    reducedMotion: input.reducedMotion ?? "no-preference",
    forcedColors: input.forcedColors ?? "none",
  });
  await page.goto(withQuery(input.path ?? "/", input.query ?? "audience=developer"));
  await disableNondeterministicMotion(page);
  await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
  await expectProductionLayoutReady(page);
  if (input.fitGraph ?? true) {
    await page.getByRole("button", { name: "Fit View" }).click();
  }
  if (input.extraZoomOut) {
    await page.getByRole("button", { name: "Zoom Out" }).click();
  }
}

async function expectNoSeriousOrCriticalViolations(page: Page) {
  const results = await new AxeBuilder({ page }).analyze();
  const blockingViolations = results.violations.filter((violation) =>
    seriousOrCritical.has(violation.impact ?? ""),
  );
  expect(blockingViolations).toEqual([]);
}

async function expectGraphNodesFullyInViewport(page: Page) {
  const stageBox = await page.locator(".graph-canvas-stage").boundingBox();
  expect(stageBox).not.toBeNull();
  const roundingTolerance = 2;
  const nodes = page.locator(".react-flow__node-runtimeNode");
  const count = await nodes.count();
  expect(count).toBeGreaterThan(0);
  for (let index = 0; index < count; index += 1) {
    await expect
      .poll(async () => {
        const nodeBox = await nodes.nth(index).boundingBox();
        return nodeBox
          ? nodeBox.x >= stageBox!.x - roundingTolerance &&
              nodeBox.y >= stageBox!.y - roundingTolerance &&
              nodeBox.x + nodeBox.width <=
                stageBox!.x + stageBox!.width + roundingTolerance &&
              nodeBox.y + nodeBox.height <=
                stageBox!.y + stageBox!.height + roundingTolerance
          : false;
      })
      .toBe(true);
  }
}

test.describe("Flight Deck visual and accessibility slice", () => {
  const runtimeErrors: string[] = [];

  test.beforeEach(async ({ page }) => {
    runtimeErrors.length = 0;
    page.on("pageerror", (error) => runtimeErrors.push(error.message));
    page.on("console", (message) => {
      if (message.type() === "error") {
        runtimeErrors.push(message.text());
      }
    });
  });

  test.afterEach(() => {
    expect(runtimeErrors).toEqual([]);
  });

  test("axe serious/critical audits pass on runtime, overlay, and evidence states", async ({
    page,
  }) => {
    await openFlightDeck(page, {
      path: "/",
      query: "audience=developer&primary=native_http",
    });
    await expectNoSeriousOrCriticalViolations(page);

    await page.getByRole("combobox", { name: "Compare flavor" }).selectOption("dynamo_online");
    await expect(page).toHaveURL(/compare=dynamo_online/u);
    await expectNoSeriousOrCriticalViolations(page);

    await page
      .getByRole("button", { name: "Show graph accessibility outline" })
      .click();
    await page.getByRole("button", { name: "Select node Rust runtime composition" }).click();
    await expect(page.getByRole("dialog")).toBeVisible();
    await expectNoSeriousOrCriticalViolations(page);
  });

  test("supports keyboard-only outline and pulse controls with visible focus", async ({
    page,
  }) => {
    await openFlightDeck(page, {
      path: "/",
      query: "audience=developer&primary=native_http",
      fitGraph: false,
    });

    const skipLink = page.getByRole("link", { name: "Skip to content" });
    await page.keyboard.press("Tab");
    await expect(skipLink).toBeFocused();

    const outlineToggle = page.getByRole("button", {
      name: "Show graph accessibility outline",
    });
    await outlineToggle.focus();
    await expect(outlineToggle).toBeFocused();

    await page.keyboard.press("Enter");
    const outline = page.getByRole("tree", { name: "Visible graph outline" });
    await expect(outline).toBeVisible();

    const firstNodeTrigger = page.getByRole("button", { name: /^Select node /u }).first();
    await firstNodeTrigger.focus();
    await expect(firstNodeTrigger).toBeFocused();
    await page.keyboard.press("Enter");
    const drawer = page.getByRole("dialog");
    await expect(drawer).toBeVisible();

    await page.keyboard.press("Escape");
    await expect(drawer).toHaveCount(0);
    await expect(
      page.locator("[data-graph-entity-trigger='true']:focus"),
    ).toHaveCount(1);

    const playButton = page.getByRole("button", { name: "Play pulse timeline" });
    await playButton.focus();
    await expect(playButton).toBeFocused();
    await page.keyboard.press("Enter");
    const pauseButton = page.getByRole("button", { name: "Pause pulse timeline" });
    await expect(pauseButton).toBeVisible();
    await page.keyboard.press("Enter");
    await expect(playButton).toBeVisible();
  });

  test("smokes 200% zoom and narrow viewport command usability", async ({ page, isMobile }) => {
    test.skip(isMobile, "Desktop zoom check only.");
    await page.setViewportSize({ width: 820, height: 720 });
    await openFlightDeck(page, {
      path: "/",
      query: "audience=developer",
    });
    await page.evaluate(() => {
      const documentElement = (
        globalThis as {
          document?: { documentElement?: { style?: { zoom?: string } } };
        }
      ).document?.documentElement;
      if (documentElement?.style) {
        documentElement.style.zoom = "2";
      }
    });

    const collapseRail = page.getByRole("button", { name: "Collapse scene rail" });
    await collapseRail.scrollIntoViewIfNeeded();
    await expect(collapseRail).toBeVisible();
    await collapseRail.click();
    await expect(page.getByRole("button", { name: "Expand scene rail" })).toBeVisible();

    const shareButton = page.getByRole("button", { name: "Share graph state" });
    await shareButton.scrollIntoViewIfNeeded();
    await expect(shareButton).toBeVisible();
  });

  test("smokes forced-colors and reduced-motion rendering", async ({ page }) => {
    await openFlightDeck(page, {
      path: "/",
      query: "audience=developer&primary=native_http",
      forcedColors: "active",
      reducedMotion: "reduce",
    });

    await expect(
      page.getByText("Motion reduced: semantic playback only."),
    ).toBeVisible();
    await expect(page.getByTestId("pulse-active-particle")).toHaveAttribute(
      "data-motion",
      "reduced",
    );
    await expectNoSeriousOrCriticalViolations(page);
  });

  test("captures deterministic runtime baseline snapshots", async ({
    page,
    isMobile,
  }) => {
    await openFlightDeck(page, {
      path: "/scenes/metrics-telemetry",
      query: "audience=developer&primary=native_http",
      extraZoomOut: true,
    });
    if (!isMobile) {
      await expectGraphNodesFullyInViewport(page);
    }
    await expect(sceneLocator(page)).toHaveScreenshot(
      "runtime-flight-deck.png",
      screenshotOptions,
    );
  });

  test("captures the complete default journey overview", async ({
    page,
    isMobile,
  }) => {
    await openFlightDeck(page, {
      path: "/",
      query: "audience=developer&primary=native_http",
    });
    if (!isMobile) {
      await expectGraphNodesFullyInViewport(page);
    }
    await expect(sceneLocator(page)).toHaveScreenshot(
      "default-journey-flight-deck.png",
      screenshotOptions,
    );
  });

  test("captures deterministic comparison overlay snapshots", async ({ page }) => {
    await openFlightDeck(page, {
      path: "/scenes/runner-protocol-registries",
      query:
        "audience=developer&primary=native_http&compare=dynamo_online",
      extraZoomOut: true,
    });
    await expect(page).toHaveURL(/compare=dynamo_online/u);
    await expectGraphNodesFullyInViewport(page);
    await expect(sceneLocator(page)).toHaveScreenshot(
      "comparison-overlay-flight-deck.png",
      screenshotOptions,
    );
  });

  test("captures deterministic expanded maintainer scene snapshots", async ({
    page,
  }) => {
    await openFlightDeck(page, {
      path: "/scenes/runner-protocol-registries",
      query: "audience=maintainer&primary=native_http",
    });
    await page
      .getByRole("button", { name: "Show graph accessibility outline" })
      .click();
    await page.getByRole("button", { name: "Expand" }).first().click();
    await expectProductionLayoutReady(page);
    await page.getByRole("button", { name: "Fit View" }).click();
    await page.getByRole("button", { name: "Zoom Out" }).click();
    await expectGraphNodesFullyInViewport(page);
    await expect(sceneLocator(page)).toHaveScreenshot(
      "maintainer-expanded-flight-deck.png",
      screenshotOptions,
    );
  });

  test("captures deterministic evidence drawer snapshots", async ({ page }) => {
    await openFlightDeck(page, {
      path: "/",
      query: "audience=developer&primary=native_http",
    });
    await page
      .getByRole("button", { name: "Show graph accessibility outline" })
      .click();
    await page.getByRole("button", { name: /^Select node /u }).first().click();
    await expect(page.getByRole("dialog")).toBeVisible();
    await expect(sceneLocator(page)).toHaveScreenshot(
      "evidence-drawer-flight-deck.png",
      screenshotOptions,
    );
  });

  test("captures deterministic reduced-motion pulse snapshots", async ({ page }) => {
    await openFlightDeck(page, {
      path: "/scenes/metrics-telemetry",
      query: "audience=developer&primary=native_http",
      reducedMotion: "reduce",
    });
    await expect(page.getByTestId("pulse-active-particle")).toHaveAttribute(
      "data-motion",
      "reduced",
    );
    await expect(sceneLocator(page)).toHaveScreenshot(
      "reduced-motion-pulse-flight-deck.png",
      screenshotOptions,
    );
  });
});
