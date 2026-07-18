// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { expect, test, type Locator, type Page } from "@playwright/test";

const fixturePath = fileURLToPath(
  new URL("../examples/cinematic/request-lifecycle.flow", import.meta.url),
);
const namedBeat = /^(establish|teach|inspect|transition)$/u;

async function openRequestLifecycle(page: Page): Promise<void> {
  await page.goto("/");
  await expect(
    page.getByRole("heading", { level: 1, name: /request lifecycle/u }),
  ).toBeVisible();
  await expect(page.getByRole("region", { name: "Scene field" })).toBeVisible();
  await expect(
    page.getByRole("region", { name: "Semantic outline" }),
  ).toBeAttached();
}

function playbackButton(page: Page, state: "play" | "pause"): Locator {
  return page.getByRole("button", {
    name: state === "play" ? /^Play(?: scene)?$/u : /^Pause(?: scene)?$/u,
  });
}

async function ensurePlaying(page: Page): Promise<void> {
  const play = playbackButton(page, "play");
  if (await play.isVisible()) {
    await play.click();
  }
  await expect(playbackButton(page, "pause")).toBeVisible();
}

test.describe("RequestLifecycleWaterfall cinematic north star", () => {
  const runtimeErrors: string[] = [];

  test.skip(
    !existsSync(fixturePath),
    "The concurrently authored request-lifecycle.flow fixture is not available yet.",
  );

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

  test("plays authored named beats", async ({ page }) => {
    await openRequestLifecycle(page);

    const subtitles = page.getByRole("region", { name: "Subtitles" });
    await expect(subtitles).toHaveAttribute("data-cue-id", namedBeat);

    await ensurePlaying(page);
    await playbackButton(page, "pause").click();

    await expect(subtitles).toHaveAttribute("data-cue-id", namedBeat);
  });

  test("supports keyboard semantic activation and exposes evidence", async ({
    page,
  }) => {
    await openRequestLifecycle(page);

    const twin = page.getByRole("region", { name: "Semantic outline" });
    const evidenceEntity = twin
      .locator('button[data-evidence-ids]:not([data-evidence-ids=""])')
      .first();
    await expect(evidenceEntity).toBeAttached();

    await evidenceEntity.focus();
    await expect(evidenceEntity).toBeFocused();
    await page.keyboard.press("Enter");

    await expect(evidenceEntity).toHaveAttribute("aria-selected", "true");
    await expect(evidenceEntity).toHaveAttribute("data-selected", "true");
    await expect(evidenceEntity).toHaveAttribute("data-evidence-ids", /\S/u);
  });

  test("resumes from the exact paused beat", async ({ page }) => {
    await openRequestLifecycle(page);
    await ensurePlaying(page);

    await playbackButton(page, "pause").click();
    const subtitles = page.getByRole("region", { name: "Subtitles" });
    const pausedBeat = await subtitles.getAttribute("data-cue-id");
    expect(pausedBeat).toMatch(namedBeat);

    await page.getByRole("button", { name: "Explore" }).click();
    await expect(
      page.getByRole("button", { name: "Resume lesson" }),
    ).toBeVisible();
    await expect(subtitles).toHaveAttribute("data-cue-id", pausedBeat!);

    await page.getByRole("button", { name: "Resume lesson" }).click();
    await expect(subtitles).toHaveAttribute("data-cue-id", pausedBeat!);
  });

  test("preserves semantics through SVG fallback", async ({ page }) => {
    await page.addInitScript(() => {
      HTMLCanvasElement.prototype.getContext = () => null;
    });
    await openRequestLifecycle(page);

    const stage = page.getByRole("region", { name: "Scene field" });
    await expect(stage).toHaveAttribute("data-backend", "svg");
    await expect(stage.locator("svg.aiperf-flow__svg-fallback")).toBeVisible();
    await expect(
      page.getByRole("region", { name: "Semantic outline" }),
    ).toBeAttached();
  });

  test("keeps the cinematic stage usable at a mobile viewport", async ({
    page,
  }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await openRequestLifecycle(page);

    const stage = page.getByRole("region", { name: "Scene field" });
    const stageBox = await stage.boundingBox();
    expect(stageBox).not.toBeNull();
    expect(stageBox!.x).toBeGreaterThanOrEqual(0);
    expect(stageBox!.x + stageBox!.width).toBeLessThanOrEqual(390);
    await expect
      .poll(() =>
        page.evaluate(
          () => document.documentElement.scrollWidth <= window.innerWidth,
        ),
      )
      .toBe(true);
    await expect(
      playbackButton(page, "play").or(playbackButton(page, "pause")),
    ).toBeVisible();
  });

  test("offers keyboard traversal between semantic entities", async ({ page }) => {
    await openRequestLifecycle(page);

    const entities = page
      .getByRole("region", { name: "Semantic outline" })
      .getByRole("list", { name: "Entities" })
      .getByRole("button");
    const firstEntity = entities.first();
    const secondEntity = entities.nth(1);

    await firstEntity.focus();
    await expect(firstEntity).toBeFocused();
    await page.keyboard.press("ArrowDown");
    await expect(secondEntity).toBeFocused();
    await expect(secondEntity).toHaveAttribute("data-focused", "true");

    await page.keyboard.press("ArrowUp");
    await expect(firstEntity).toBeFocused();
    await expect(firstEntity).toHaveAttribute("data-focused", "true");
  });

  test("opens authored evidence from the semantic twin", async ({ page }) => {
    await openRequestLifecycle(page);

    const arrivalEvidence = page
      .getByRole("region", { name: "Semantic outline" })
      .getByRole("button", { name: "Arrival evidence", exact: true });
    await arrivalEvidence.focus();
    await page.keyboard.press("Enter");
    await expect(arrivalEvidence).toHaveAttribute("aria-selected", "true");

    const lens = page.getByRole("region", { name: "Context Lens" });
    await expect(lens).toBeVisible();
    await expect(lens).toContainText("Arrival evidence");
    await expect(lens).toContainText(
      "Evidence ID ev-arrival-req-017, captured from the scheduler at 1200 milliseconds.",
    );
  });
});
