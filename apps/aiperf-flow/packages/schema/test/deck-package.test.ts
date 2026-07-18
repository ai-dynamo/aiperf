// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, test } from "vitest";

import {
  deckPackageSchema,
  safeParseDeckPackage,
  type DeckPackage,
} from "../src/deck-package.js";

const sourceMap = {
  source: "deck.flow",
  start: { offset: 0, line: 1, column: 1 },
  end: { offset: 1, line: 1, column: 2 },
};

const minimalScene = {
  id: "main",
  title: "Main",
  summary: "A diagram slide",
  roots: [
    {
      kind: "rect" as const,
      id: "box",
      geometry: { x: 0, y: 0, width: 100, height: 40 },
      style: {},
      accessibility: { label: "Box" },
      fallback: "Box unavailable",
      sourceMap,
    },
  ],
  camera: [],
  timeline: [
    {
      id: "enter-box",
      at: 0,
      duration: 200,
      target: "box",
      action: "enter",
      sourceMap,
    },
  ],
  narration: "",
  interactions: [],
  responsive: [],
  accessibility: { label: "Main scene", readingOrder: ["box"] },
  fallback: "Scene unavailable",
  sourceMap,
};

function validPackage(
  overrides: Partial<DeckPackage> & {
    slides?: DeckPackage["slides"];
  } = {},
): unknown {
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
        render: { kind: "scene", scene: minimalScene },
      },
    ],
    glossary: [{ word: "aiperf-cli", meaning: "Native CLI crate" }],
    ...overrides,
  };
}

describe("DeckPackage", () => {
  test("parses a valid package with a scene render", () => {
    const parsed = deckPackageSchema.parse(validPackage());

    expect(parsed.schemaVersion).toBe(1);
    expect(parsed.id).toBe("rust-architecture");
    expect(parsed.slides).toHaveLength(1);
    expect(parsed.slides[0]?.narration).toBe(
      "AIPerf ships as one native aiperf binary.",
    );
    expect(parsed.slides[0]?.render?.kind).toBe("scene");
    expect(parsed.slides[0]?.render?.scene.id).toBe("main");
  });

  test("allows slides without render", () => {
    const result = safeParseDeckPackage(
      validPackage({
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
    if (result.ok) {
      expect(result.value.slides[0]?.render).toBeUndefined();
    }
  });

  test("rejects unknown fields", () => {
    const result = safeParseDeckPackage({
      ...validPackage(),
      extra: true,
    });

    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.diagnostics[0]?.code).toBe("DECK_PACKAGE_INVALID");
    }
  });

  test('rejects render.kind "mental_model"', () => {
    const result = safeParseDeckPackage(
      validPackage({
        slides: [
          {
            id: "bad-render",
            eyebrow: "Bad",
            title: "Mental model escape hatch",
            lede: "Not allowed",
            narration: "Should fail.",
            points: [],
            caption: "Caption",
            render: { kind: "mental_model", component: "MentalModel" } as never,
          },
        ],
      }),
    );

    expect(result.ok).toBe(false);
  });

  test("requires narration string on slides", () => {
    const result = safeParseDeckPackage(
      validPackage({
        slides: [
          {
            id: "missing-narration",
            eyebrow: "Missing",
            title: "No narration",
            lede: "Lede",
            points: [],
            caption: "Caption",
          } as never,
        ],
      }),
    );

    expect(result.ok).toBe(false);
  });
});
