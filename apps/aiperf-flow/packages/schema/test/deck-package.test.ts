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
      capability: "core.rect" as const,
      geometry: { x: 0, y: 0, width: 100, height: 40 },
      style: {},
      accessibility: { label: "Box" },
      fallback: "Box unavailable",
      sourceMap,
    },
    {
      kind: "text" as const,
      id: "label",
      capability: "core.text" as const,
      text: "Coordinator",
      geometry: { x: 8, y: 8, width: 84, height: 24 },
      style: { fontSize: 14 },
      accessibility: { label: "Coordinator label" },
      fallback: "Label unavailable",
      sourceMap,
    },
    {
      kind: "connector" as const,
      id: "arrow",
      capability: "core.connector" as const,
      path: "M100 20 H160",
      points: [
        { x: 100, y: 20 },
        { x: 160, y: 20 },
      ],
      from: { nodeId: "box", anchor: "right" },
      to: { nodeId: "label", anchor: "left" },
      geometry: { x: 100, y: 16, width: 60, height: 8 },
      style: { stroke: "#3FA266" },
      accessibility: { label: "Flow arrow" },
      fallback: "Arrow unavailable",
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
    {
      id: "draw-arrow",
      at: 200,
      duration: 300,
      target: "arrow",
      action: "draw",
      sourceMap,
    },
  ],
  narration: "",
  interactions: [],
  responsive: [],
  accessibility: { label: "Main scene", readingOrder: ["box", "label", "arrow"] },
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

  test("accepts text/path/points nodes, foundation capabilities, and enter/draw cues", () => {
    const parsed = deckPackageSchema.parse(validPackage());
    const scene = parsed.slides[0]?.render?.scene;
    expect(scene).toBeDefined();
    if (scene === undefined) {
      return;
    }

    const [rect, text, connector] = scene.roots;
    expect(rect).toMatchObject({
      kind: "rect",
      capability: "core.rect",
    });
    expect(text).toMatchObject({
      kind: "text",
      capability: "core.text",
      text: "Coordinator",
    });
    expect(connector).toMatchObject({
      kind: "connector",
      capability: "core.connector",
      path: "M100 20 H160",
      points: [
        { x: 100, y: 20 },
        { x: 160, y: 20 },
      ],
    });
    expect(scene.timeline.map((cue) => cue.action)).toEqual(["enter", "draw"]);
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
