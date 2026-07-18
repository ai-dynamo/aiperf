// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, it, expect } from "vitest";

import type { DeckPackage } from "@aiperf/flow-schema";

const packagePath = join(
  dirname(fileURLToPath(import.meta.url)),
  "../../../../explainers/src/decks-generated/dynosim.package.json",
);

const DYNOSIM_PACKAGE = JSON.parse(
  readFileSync(packagePath, "utf8"),
) as DeckPackage;

describe("dynosim.flow compiled deck", () => {
  it("compiles all 18 slides", () => {
    expect(DYNOSIM_PACKAGE.id).toBe("dynosim");
    expect(DYNOSIM_PACKAGE.slides.length).toBe(18);
  });

  it("covers every original topic in order", () => {
    const expectedTitles = [
      "Why Dynosim",
      "Feature gate",
      "Config seam",
      "Routing",
      "Composition",
      "Offline mode",
      "Online mode",
      "Clock compare",
      "Event queues",
      "Sim pump",
      "Ordering rule",
      "Step bounds",
      "Submission",
      "Token path",
      "Metrics",
      "Delivery modes",
      "Completion",
      "Recap",
    ];

    expect(DYNOSIM_PACKAGE.slides.map((s) => s.title)).toEqual(expectedTitles);
  });

  it("has non-empty narration for every slide", () => {
    for (const slide of DYNOSIM_PACKAGE.slides) {
      expect(slide.narration.length).toBeGreaterThan(10);
    }
  });

  it("has a compiled scene render for every slide", () => {
    for (const slide of DYNOSIM_PACKAGE.slides) {
      expect(slide.render?.kind).toBe("scene");
      expect(Array.isArray(slide.render?.scene?.roots)).toBe(true);
      expect(slide.render?.scene?.roots.length).toBeGreaterThan(0);
    }
  });

  it("covers key dynosim technical concepts across narration", () => {
    const combined = DYNOSIM_PACKAGE.slides.map((s) => s.narration).join(" ");

    expect(combined).toContain("SimClock");
    expect(combined).toContain("RealClock");
    expect(combined).toContain("deterministic");
    expect(combined.toLowerCase()).toContain("clock");
    expect(combined.toLowerCase()).toContain("observer");
  });

  it("discusses metrics accumulation", () => {
    const combined = DYNOSIM_PACKAGE.slides.map((s) => s.narration).join(" ");
    expect(combined.toLowerCase()).toContain("time to first token");
  });

  it("has scene rects with readable labels", () => {
    for (const slide of DYNOSIM_PACKAGE.slides) {
      const rectTexts = (slide.render?.scene?.roots ?? [])
        .flatMap((root: { children?: { text?: string }[] }) => root.children ?? [])
        .map((child) => child.text);
      expect(rectTexts.length).toBeGreaterThan(0);
    }
  });
});
