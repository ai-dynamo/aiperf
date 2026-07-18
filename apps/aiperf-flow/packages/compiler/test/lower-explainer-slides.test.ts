/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { SourceRange } from "@aiperf/flow-schema";
import { describe, expect, test } from "vitest";

import {
  lowerExplainerSlides,
  slideIdFromTitle,
  type SlideTextAst,
} from "../src/lower-explainer-slides.js";

function range(): SourceRange {
  return {
    source: "deck.flow",
    start: { offset: 0, line: 1, column: 1 },
    end: { offset: 1, line: 1, column: 2 },
  };
}

function sampleSlide(overrides: Partial<SlideTextAst> = {}): SlideTextAst {
  return {
    kind: "slide",
    sourceMap: range(),
    eyebrow: "Product shell",
    title: "One binary is both CLI and engine",
    lede: "AIPerf ships as one native binary.",
    narration: "AIPerf ships as one native aiperf binary.",
    term: { word: "aiperf-cli", meaning: "Native CLI crate" },
    points: ["CLI and engine share one process."],
    caption: "Product shell overview",
    ...overrides,
  };
}

describe("slideIdFromTitle", () => {
  test("slugifies titles", () => {
    expect(slideIdFromTitle("Product shell", 0)).toBe("product-shell");
  });

  test("falls back to index when title is empty", () => {
    expect(slideIdFromTitle("   ", 2)).toBe("slide-2");
  });
});

describe("lowerExplainerSlides", () => {
  test("maps text fields into SlidePackage without render", () => {
    const result = lowerExplainerSlides([sampleSlide()]);

    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }

    expect(result.value).toHaveLength(1);
    const slide = result.value[0]!;
    expect(slide.id).toBe("one-binary-is-both-cli-and-engine");
    expect(slide.eyebrow).toBe("Product shell");
    expect(slide.title).toBe("One binary is both CLI and engine");
    expect(slide.lede).toBe("AIPerf ships as one native binary.");
    expect(slide.narration).toBe("AIPerf ships as one native aiperf binary.");
    expect(slide.term).toEqual({
      word: "aiperf-cli",
      meaning: "Native CLI crate",
    });
    expect(slide.points).toEqual(["CLI and engine share one process."]);
    expect(slide.caption).toBe("Product shell overview");
    expect(slide.render).toBeUndefined();
  });

  test("prefers an authored slide id when present", () => {
    const result = lowerExplainerSlides([
      sampleSlide({ id: "product-shell" }),
    ]);

    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }

    expect(result.value[0]?.id).toBe("product-shell");
  });

  test("omits term when absent", () => {
    const result = lowerExplainerSlides([
      sampleSlide({ term: undefined }),
    ]);

    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }

    expect(result.value[0]?.term).toBeUndefined();
  });

  test("rejects empty title", () => {
    const result = lowerExplainerSlides([sampleSlide({ title: "   " })]);

    expect(result.ok).toBe(false);
    if (result.ok) {
      return;
    }

    expect(result.diagnostics).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          code: "EXPLAINER_SLIDE_FIELD_REQUIRED",
          message: expect.stringContaining("title"),
        }),
      ]),
    );
  });

  test("rejects empty narration", () => {
    const result = lowerExplainerSlides([sampleSlide({ narration: "" })]);

    expect(result.ok).toBe(false);
    if (result.ok) {
      return;
    }

    expect(result.diagnostics).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          code: "EXPLAINER_SLIDE_FIELD_REQUIRED",
          message: expect.stringContaining("narration"),
        }),
      ]),
    );
  });

  test("lowers multiple slides in order", () => {
    const result = lowerExplainerSlides([
      sampleSlide({ id: "first", title: "First", narration: "First narration." }),
      sampleSlide({
        id: "second",
        title: "Second",
        narration: "Second narration.",
        term: undefined,
      }),
    ]);

    expect(result.ok).toBe(true);
    if (!result.ok) {
      return;
    }

    expect(result.value.map((slide) => slide.id)).toEqual(["first", "second"]);
    expect(result.value.map((slide) => slide.title)).toEqual([
      "First",
      "Second",
    ]);
  });

  test("lowers package-scene sceneIr into render with roots and timeline", () => {
    const result = lowerExplainerSlides([
      sampleSlide({
        id: "one-box",
        sceneIr: {
          kind: "package-scene",
          roots: [
            {
              id: "box",
              capability: "core.rect",
              layout: { x: 80, y: 120, width: 160, height: 72 },
              style: { fill: "#3FA266" },
            },
          ],
          timeline: [
            {
              id: "enter-box",
              at: 0,
              duration: 400,
              target: "box",
              action: "enter",
            },
          ],
          camera: [],
        },
      }),
    ]);

    expect(result.ok, JSON.stringify(result.diagnostics)).toBe(true);
    if (!result.ok) {
      return;
    }

    const scene = result.value[0]?.render?.scene;
    expect(result.value[0]?.render?.kind).toBe("scene");
    expect(scene?.roots.length).toBeGreaterThanOrEqual(1);
    expect(scene?.timeline.length).toBeGreaterThan(0);
  });
});
