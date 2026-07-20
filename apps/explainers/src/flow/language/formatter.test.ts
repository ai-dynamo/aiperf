/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";

import { formatDocument } from "./formatter.js";
import { parseDocument } from "./parser.js";

const FIXTURE = `explainer "Round Trip Demo" {
  id: "round-trip-demo"
  route: "/round-trip-demo"
  topic: "authoring"
  storagePrefix: "round-trip-demo"
  classPrefix: "deck-round-trip"
  eyebrowLabel: "ROUND TRIP"
  startGateTitle: "Start the demo"

  hub: {
    highlight: "Preserve slides"
    title: "Formatter must keep explainers"
    description: "parse then format then parse again."
  }

  slide "Opening" {
    eyebrow: "SLIDE · ONE"
    title: "Opening"
    lede: "A short lede."
    narration: "This slide must survive formatDocument."
    points: ["keep metadata", "keep narration"]
    caption: "Round-trip caption."
    term: { word: "AST", meaning: "Abstract syntax tree" }
    render: @scene {
      sdk.Header(id = "h1", title = "DEMO", caption = "round trip", x = 18, y = 16, width = 664, height = 44)
      timeline main {
        at 0 reveal h1 duration 180
      }
    }
  }

  slide "Closing" {
    eyebrow: "SLIDE · TWO"
    title: "Closing"
    lede: "Second slide lede."
    narration: "Second slide narration stays intact."
    caption: "No points needed."
  }

  finalCard: @scene {
    sdk.Title(id = "final-title", text = "Done", x = 54, y = 105, width = 430, height = 42)
    timeline main {
      at 0 reveal final-title duration 160
    }
  }
}
`;

describe("formatDocument explainer round-trip", () => {
  it("preserves explainer slides and finalCard through parse → format → parse", () => {
    const first = parseDocument(FIXTURE, "round-trip.flow");
    expect(first.ok).toBe(true);
    if (!first.ok) return;

    expect(first.value.explainers).toHaveLength(1);
    expect(first.value.scenes).toEqual([]);
    expect(first.value.explainers![0]!.slides).toHaveLength(2);

    const formatted = formatDocument(first.value);
    expect(formatted).toMatch(/explainer\s+/);
    expect(formatted).toMatch(/slide\s+"Opening"/);
    expect(formatted).toMatch(/slide\s+"Closing"/);
    expect(formatted).toMatch(/finalCard\s*:/);

    const second = parseDocument(formatted, "round-trip.formatted.flow");
    expect(second.ok).toBe(true);
    if (!second.ok) {
      throw new Error(
        second.diagnostics.map((d) => d.message).join("\n") || "reparse failed",
      );
    }

    const original = first.value.explainers![0]!;
    const roundTripped = second.value.explainers![0]!;

    expect(roundTripped.id).toBe(original.id);
    expect(roundTripped.metadata).toEqual(original.metadata);
    expect(roundTripped.slides).toHaveLength(original.slides.length);

    for (let i = 0; i < original.slides.length; i += 1) {
      const before = original.slides[i]!;
      const after = roundTripped.slides[i]!;
      expect(after.eyebrow).toBe(before.eyebrow);
      expect(after.title).toBe(before.title);
      expect(after.lede).toBe(before.lede);
      expect(after.narration).toBe(before.narration);
      expect(after.caption).toBe(before.caption);
      expect(after.points).toEqual(before.points);
      expect(after.term).toEqual(before.term);
      expect(Boolean(after.sceneIr)).toBe(Boolean(before.sceneIr));
    }

    expect(Boolean(roundTripped.finalCard)).toBe(true);
    expect(Boolean(original.finalCard)).toBe(true);
  });
});
