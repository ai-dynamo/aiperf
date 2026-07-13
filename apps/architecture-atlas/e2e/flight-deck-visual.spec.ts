// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Locator, type Page } from "@playwright/test";

const seriousOrCritical = new Set(["serious", "critical"]);
const screenshotOptions = {
  animations: "disabled",
  caret: "hide",
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
}

async function expectNoSeriousOrCriticalViolations(page: Page) {
  const results = await new AxeBuilder({ page }).analyze();
  const blockingViolations = results.violations.filter((violation) =>
    seriousOrCritical.has(violation.impact ?? ""),
  );
  expect(blockingViolations).toEqual([]);
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

    await page.getByTestId("graph-node-node.runtime-composition").click();
    await expect(page.getByRole("dialog")).toBeVisible();
    await expectNoSeriousOrCriticalViolations(page);
  });

  test("supports keyboard-only outline and pulse controls with visible focus", async ({
    page,
  }) => {
    await openFlightDeck(page, {
      path: "/",
      query: "audience=developer&primary=native_http",
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

  test("captures deterministic runtime baseline snapshots", async ({ page }) => {
    await openFlightDeck(page, {
      path: "/",
      query: "audience=developer&primary=native_http",
    });
    await expect(sceneLocator(page)).toHaveScreenshot(
      "runtime-flight-deck.png",
      screenshotOptions,
    );
  });

  test("captures deterministic comparison overlay snapshots", async ({ page }) => {
    await openFlightDeck(page, {
      path: "/",
      query:
        "audience=developer&primary=native_http&compare=dynamo_online",
    });
    await expect(page).toHaveURL(/compare=dynamo_online/u);
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
      path: "/",
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
