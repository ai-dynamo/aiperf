/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { DeckPackage } from "@aiperf/flow-schema";
import { describe, expect, test } from "vitest";

import { validateExplainerSet } from "../src/validate-explainer-set.js";

function deck(
  overrides: Partial<Pick<DeckPackage, "id" | "route">> & {
    id: string;
    route: string;
  },
): DeckPackage {
  return {
    schemaVersion: 1,
    id: overrides.id,
    route: overrides.route,
    topic: "architecture",
    storagePrefix: `${overrides.id}-explainer`,
    classPrefix: overrides.id,
    eyebrowLabel: overrides.id.toUpperCase(),
    startGateTitle: `${overrides.id} walkthrough`,
    hub: {
      title: "from scratch",
      highlight: overrides.id,
      description: `Narrated walkthrough for ${overrides.id}.`,
    },
    slides: [
      {
        id: "intro",
        eyebrow: "Intro",
        title: "Opening",
        lede: "Lede",
        narration: "A sufficiently detailed narration for uniqueness tests.",
        points: [],
        caption: "Caption",
      },
    ],
    glossary: [],
  };
}

describe("validateExplainerSet", () => {
  test("accepts an empty set", () => {
    const result = validateExplainerSet([]);

    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.value).toEqual([]);
    }
  });

  test("accepts packages with unique ids and routes", () => {
    const packages = [
      deck({ id: "rust-architecture", route: "/rust-architecture" }),
      deck({ id: "slurm-velo", route: "/slurm-velo" }),
    ];

    const result = validateExplainerSet(packages);

    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.value).toBe(packages);
      expect(result.diagnostics).toEqual([]);
    }
  });

  test("rejects duplicate ids with EXPLAINER_DUPLICATE_ID", () => {
    const result = validateExplainerSet([
      deck({ id: "rust-architecture", route: "/rust-architecture" }),
      deck({ id: "rust-architecture", route: "/other-route" }),
    ]);

    expect(result.ok).toBe(false);
    expect(result.diagnostics).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          code: "EXPLAINER_DUPLICATE_ID",
          severity: "error",
          message: expect.stringContaining('Duplicate explainer id "rust-architecture"'),
        }),
      ]),
    );
  });

  test("rejects duplicate routes with EXPLAINER_DUPLICATE_ROUTE", () => {
    const result = validateExplainerSet([
      deck({ id: "rust-architecture", route: "/shared" }),
      deck({ id: "slurm-velo", route: "/shared" }),
    ]);

    expect(result.ok).toBe(false);
    expect(result.diagnostics).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          code: "EXPLAINER_DUPLICATE_ROUTE",
          severity: "error",
          message: expect.stringContaining('Duplicate explainer route "/shared"'),
        }),
      ]),
    );
  });

  test("reports both id and route collisions when present", () => {
    const result = validateExplainerSet([
      deck({ id: "same", route: "/same" }),
      deck({ id: "same", route: "/same" }),
    ]);

    expect(result.ok).toBe(false);
    expect(result.diagnostics.map(({ code }) => code).sort()).toEqual([
      "EXPLAINER_DUPLICATE_ID",
      "EXPLAINER_DUPLICATE_ROUTE",
    ]);
  });
});
