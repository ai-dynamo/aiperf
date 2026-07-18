/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

import { createElement } from "react";
import { cleanup, render } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { packageToDeckDefinition } from "../core/package-adapter";

afterEach(() => {
  cleanup();
});

const fixturePackage = {
  schemaVersion: 1 as const,
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
        kind: "scene" as const,
        scene: {
          id: "main",
          title: "Main",
          summary: "A diagram slide",
          roots: [
            {
              id: "box",
              kind: "rect",
              geometry: { x: 80, y: 120, width: 160, height: 72 },
              style: { fill: "#3FA266" },
              accessibility: { label: "Coordinator" },
            },
          ],
          timeline: [
            {
              id: "enter-box",
              at: 0,
              duration: 400,
              action: "enter",
              target: "box",
            },
          ],
        },
      },
    },
  ],
  glossary: [{ word: "aiperf-cli", meaning: "Native CLI crate" }],
};

describe("packageToDeckDefinition", () => {
  it("maps id, route, slide count, and narration from the package", () => {
    const deck = packageToDeckDefinition(fixturePackage);

    expect(deck.id).toBe("rust-architecture");
    expect(deck.route).toBe("/rust-architecture");
    expect(deck.slides).toHaveLength(1);
    expect(deck.slides[0]?.narration).toBe(
      "AIPerf ships as one native aiperf binary.",
    );
  });

  it("renders MentalModel without throwing for a fixture with one scene", () => {
    const deck = packageToDeckDefinition(fixturePackage);
    const slide = deck.slides[0]!;

    expect(() => {
      render(createElement(deck.MentalModel, { slideIndex: 0, slide }));
    }).not.toThrow();
  });
});
