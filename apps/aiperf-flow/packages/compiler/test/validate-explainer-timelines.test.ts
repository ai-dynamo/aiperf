/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { DeckPackage, SceneIr, SourceRange } from "@aiperf/flow-schema";
import { describe, expect, test } from "vitest";

import { validateExplainerTimelines } from "../src/validate-explainer-timelines.js";

const sourceMap: SourceRange = {
  source: "deck.flow",
  start: { offset: 0, line: 1, column: 1 },
  end: { offset: 1, line: 1, column: 2 },
};

function scene(timeline: SceneIr["timeline"]): SceneIr {
  return {
    id: "main",
    title: "Main",
    summary: "A diagram slide",
    roots: [
      {
        kind: "rect",
        id: "box",
        geometry: { x: 0, y: 0, width: 100, height: 40 },
        style: {},
        accessibility: { label: "Box" },
        fallback: "Box unavailable",
        sourceMap,
      },
    ],
    camera: [],
    timeline,
    narration: "",
    interactions: [],
    responsive: [],
    accessibility: { label: "Main scene", readingOrder: ["box"] },
    fallback: "Scene unavailable",
    sourceMap,
  };
}

function deck(overrides: Partial<DeckPackage> = {}): DeckPackage {
  return {
    schemaVersion: 1,
    id: "rust-architecture",
    route: "/rust-architecture",
    topic: "architecture",
    storagePrefix: "rust-arch-explainer",
    classPrefix: "rust-arch",
    eyebrowLabel: "RUST ARCHITECTURE",
    startGateTitle: "Rust architecture walkthrough",
    hub: {
      title: "from scratch",
      highlight: "Rust architecture",
      description: "Narrated walkthrough of the native workspace.",
    },
    slides: [
      {
        id: "product-shell",
        eyebrow: "Product shell",
        title: "One binary is both CLI and engine",
        lede: "AIPerf ships as one native binary.",
        narration: "AIPerf ships as one native aiperf binary.",
        points: ["CLI and engine share one process."],
        caption: "Product shell overview",
        render: {
          kind: "scene",
          scene: scene([
            {
              id: "enter-box",
              at: 0,
              duration: 200,
              target: "box",
              action: "enter",
              sourceMap,
            },
          ]),
        },
      },
    ],
    glossary: [{ word: "aiperf-cli", meaning: "Native CLI crate" }],
    ...overrides,
  };
}

describe("validateExplainerTimelines", () => {
  test("accepts scene slides with a non-empty timeline", () => {
    const result = validateExplainerTimelines(deck());

    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.value.id).toBe("rust-architecture");
      expect(result.diagnostics).toEqual([]);
    }
  });

  test("accepts text-only slides without render", () => {
    const result = validateExplainerTimelines(
      deck({
        slides: [
          {
            id: "text-only",
            eyebrow: "Text",
            title: "No diagram",
            lede: "Words only",
            narration: "This slide has no diagram.",
            points: [],
            caption: "Caption",
          },
        ],
      }),
    );

    expect(result.ok).toBe(true);
  });

  test("rejects scene slides with an empty timeline", () => {
    const result = validateExplainerTimelines(
      deck({
        slides: [
          {
            id: "static-diagram",
            eyebrow: "Diagram",
            title: "Missing motion",
            lede: "Has a scene but no cues",
            narration: "This diagram should fail closed.",
            points: [],
            caption: "Caption",
            render: { kind: "scene", scene: scene([]) },
          },
        ],
      }),
    );

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.diagnostics).toEqual([
        expect.objectContaining({
          code: "EXPLAINER_TIMELINE_REQUIRED",
          severity: "error",
          message: expect.stringContaining("static-diagram"),
        }),
      ]);
    }
  });

  test("reports every empty-timeline scene slide", () => {
    const result = validateExplainerTimelines(
      deck({
        slides: [
          {
            id: "ok",
            eyebrow: "Ok",
            title: "Animated",
            lede: "Has cues",
            narration: "Good.",
            points: [],
            caption: "Caption",
            render: {
              kind: "scene",
              scene: scene([
                {
                  id: "enter-box",
                  at: 0,
                  duration: 200,
                  target: "box",
                  action: "enter",
                  sourceMap,
                },
              ]),
            },
          },
          {
            id: "empty-a",
            eyebrow: "A",
            title: "Empty A",
            lede: "No cues",
            narration: "Bad A.",
            points: [],
            caption: "Caption",
            render: { kind: "scene", scene: scene([]) },
          },
          {
            id: "empty-b",
            eyebrow: "B",
            title: "Empty B",
            lede: "No cues",
            narration: "Bad B.",
            points: [],
            caption: "Caption",
            render: { kind: "scene", scene: scene([]) },
          },
        ],
      }),
    );

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.diagnostics.map((d) => d.message)).toEqual([
        expect.stringContaining("empty-a"),
        expect.stringContaining("empty-b"),
      ]);
    }
  });
});
