/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Browser checks for the narrated decks.
//!
//! These cover exactly what jsdom cannot: real layout. The node types added for charts declare
//! their box up front (`style.width`/`height`, from a pure layout module) and the component then
//! draws into it — if the two ever disagree, nothing in a unit test notices, but the node visibly
//! overflows its card. The same goes for framing: `Slide` re-fits the view on every reveal tick,
//! and whether the result actually contains the diagram depends on measured geometry.

import { expect, test, type ConsoleMessage, type Page } from "@playwright/test";
import { ASYNC_DATAFLOW_ENGINE_DECK } from "../src/decks/async-dataflow-engine/deck.js";
import { PYTHON_GRAPH_WORKLOAD_DECK } from "../src/decks/python-graph-workload/deck.js";
import { METRICS_PLANE_DECK } from "../src/decks/metrics-plane/deck.js";
import { NATIVE_DIAGRAM_VOCABULARY_DECK } from "../src/decks/native-diagram-vocabulary/deck.js";

const DECKS = [
  ASYNC_DATAFLOW_ENGINE_DECK,
  PYTHON_GRAPH_WORKLOAD_DECK,
  METRICS_PLANE_DECK,
  NATIVE_DIAGRAM_VOCABULARY_DECK,
];

/**
 * How long the canvas must hold still before it counts as settled.
 *
 * `useReveal` steps one id every 220ms and `Slide` then animates `fitView` for 220ms, so any
 * window comfortably longer than one step distinguishes "mid-cascade" from "done". A flat sleep
 * cannot: it is dead time on a two-node slide and truncates a twelve-node one.
 */
const SETTLE_STABLE_MS = 400;

/** Upper bound for one slide to settle; the longest authored slide needs ~3s. */
const SETTLE_TIMEOUT_MS = 8_000;

/** Collect console errors and uncaught exceptions for the life of the page. */
function watchForErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on("console", (msg: ConsoleMessage) => {
    if (msg.type() === "error") errors.push(`console: ${msg.text()}`);
  });
  page.on("pageerror", (err) => errors.push(`pageerror: ${err.message}`));
  return errors;
}

/** Clear the audio start gate so the deck mounts its canvas and the reveal cascade runs. */
async function start(page: Page): Promise<void> {
  const silent = page.getByRole("button", { name: /Play without audio/i });
  await silent.click();
  await expect(silent).toBeHidden();
}

/** Key on `window` holding the last observed canvas signature and when it was first seen. */
const SETTLE_KEY = "__aiperfFlowSettleProbe";

/**
 * Wait for the reveal cascade and the `fitView` animation it triggers to both stop moving.
 *
 * The signature samples what the assertions below actually read — how many nodes exist, how many
 * are on screen, and the viewport transform — so this returns exactly when those stop changing
 * rather than after a guessed interval. A slide that never reveals its last node still settles;
 * the visible-count assertion then fails on the real discrepancy instead of on a timeout.
 */
async function waitForCascadeSettled(page: Page): Promise<void> {
  await page.waitForFunction(
    ([key, stableMs]: [string, number]) => {
      const store = window as unknown as Record<string, { sig: string; since: number } | undefined>;
      const nodes = Array.from(document.querySelectorAll<HTMLElement>(".react-flow__node"));
      const onScreen = nodes.filter((el) => el.getClientRects().length > 0).length;
      const viewport = document.querySelector<HTMLElement>(".react-flow__viewport");
      const sig = `${nodes.length}/${onScreen}/${viewport?.style.transform ?? ""}`;

      const now = performance.now();
      const prev = store[key];
      if (prev === undefined || prev.sig !== sig) {
        store[key] = { sig, since: now };
        return false;
      }
      return now - prev.since >= stableMs;
    },
    [SETTLE_KEY, SETTLE_STABLE_MS] as [string, number],
    { timeout: SETTLE_TIMEOUT_MS, polling: 100 },
  );
}

async function gotoSlide(page: Page, index: number, title: string): Promise<void> {
  // Drop the previous slide's sample so its signature can never be mistaken for a settled state
  // on the slide we are about to open.
  await page.evaluate((key) => {
    delete (window as unknown as Record<string, unknown>)[key];
  }, SETTLE_KEY);
  await page.getByRole("button", { name: `Go to slide ${index + 1}: ${title}` }).click();
  await waitForCascadeSettled(page);
}

type NodeBox = {
  id: string;
  declaredWidth: number;
  declaredHeight: number;
  contentWidth: number;
  contentHeight: number;
};

/**
 * Declared box versus rendered content for every node carrying an explicit size.
 *
 * `offsetWidth` is pre-transform, so both numbers are in the same unscaled CSS pixels regardless
 * of the canvas zoom.
 */
async function measureNodes(page: Page): Promise<NodeBox[]> {
  return await page.$$eval(".react-flow__node", (els) =>
    els
      .filter((el) => (el as HTMLElement).style.width !== "")
      .map((el) => {
        const wrapper = el as HTMLElement;
        const child = wrapper.firstElementChild as HTMLElement | null;
        return {
          id: wrapper.getAttribute("data-id") ?? "?",
          declaredWidth: wrapper.offsetWidth,
          declaredHeight: wrapper.offsetHeight,
          contentWidth: child?.offsetWidth ?? 0,
          contentHeight: child?.offsetHeight ?? 0,
        };
      }),
  );
}

for (const deck of DECKS) {
  test.describe(`deck ${deck.id}`, () => {
    test("renders every slide with no console errors and no clipped nodes", async ({ page }) => {
      const errors = watchForErrors(page);
      await page.goto(`/${deck.id}`);
      await start(page);

      for (const [index, slide] of deck.slides.entries()) {
        await gotoSlide(page, index, slide.title);

        const rendered = await page.locator(".react-flow__node").count();
        expect(rendered, `slide "${slide.id}" rendered no nodes`).toBeGreaterThan(0);

        // Every authored node must be on screen once the cascade has settled — a node missing
        // from revealOrder stays hidden forever, and nothing else surfaces that.
        expect(
          await page.locator(".react-flow__node:visible").count(),
          `slide "${slide.id}" hid nodes after the cascade settled`,
        ).toBe(slide.nodes.length);

        // The declared box and the rendered content must MATCH, not merely fit. Over-allocating
        // is as wrong as overflowing: it pads the node with dead space and feeds `fitView` a
        // box the diagram does not occupy — and it cannot be seen by eye, so only this catches it.
        //
        // `soft` so one bad node reports every other node too. A hard expect aborts the whole
        // deck at the first failure, which is exactly how two broken node types stayed hidden
        // behind a third on the slide before them.
        for (const box of await measureNodes(page)) {
          expect
            .soft(
              Math.abs(box.contentWidth - box.declaredWidth),
              `node "${box.id}" on slide "${slide.id}" width: declared ${box.declaredWidth}, rendered ${box.contentWidth}`,
            )
            .toBeLessThanOrEqual(1);
          expect
            .soft(
              Math.abs(box.contentHeight - box.declaredHeight),
              `node "${box.id}" on slide "${slide.id}" height: declared ${box.declaredHeight}, rendered ${box.contentHeight}`,
            )
            .toBeLessThanOrEqual(1);
        }
      }

      expect(errors).toEqual([]);
    });

    test("frames the whole diagram after the reveal cascade", async ({ page }) => {
      await page.goto(`/${deck.id}`);
      await start(page);

      for (const [index, slide] of deck.slides.entries()) {
        await gotoSlide(page, index, slide.title);

        const viewport = await page.locator(".react-flow__viewport").boundingBox();
        const pane = await page.locator(".react-flow__pane").boundingBox();
        expect(viewport, `slide "${slide.id}" has no viewport`).not.toBeNull();
        expect(pane, `slide "${slide.id}" has no pane`).not.toBeNull();

        // fitView is asked for padding 0.16, so the content box must sit inside the pane. A few
        // pixels of tolerance absorbs sub-pixel transform rounding.
        const slack = 4;
        expect(viewport!.width, `slide "${slide.id}" overflows the pane horizontally`)
          .toBeLessThanOrEqual(pane!.width + slack);
        expect(viewport!.height, `slide "${slide.id}" overflows the pane vertically`)
          .toBeLessThanOrEqual(pane!.height + slack);
      }
    });
  });
}

test("home lists every deck and each route resolves", async ({ page }) => {
  const errors = watchForErrors(page);
  await page.goto("/");

  for (const deck of DECKS) {
    const link = page.getByRole("link", { name: new RegExp(deck.title.split(":")[0]!, "i") });
    await expect(link.first()).toBeVisible();
  }

  for (const deck of DECKS) {
    await page.goto(`/${deck.id}`);
    // DeckRoute renders this literal when the registry has no entry for the id.
    await expect(page.getByText(`No deck registered for id "${deck.id}"`)).toHaveCount(0);
  }

  expect(errors).toEqual([]);
});
