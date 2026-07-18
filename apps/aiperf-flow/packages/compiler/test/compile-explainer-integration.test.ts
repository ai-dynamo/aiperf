/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { FOUNDATION_CAPABILITIES } from "@aiperf/flow-schema";
import { describe, expect, test } from "vitest";

import { compileExplainerSource } from "../src/compile-explainer.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const fixturePath = path.join(here, "fixtures", "minimal-explainer.flow");
const fixtureSource = readFileSync(fixturePath, "utf8");

describe("compileExplainerSource integration", () => {
  test("minimal-explainer.flow yields a DeckPackage with narrated slides and scene timelines", () => {
    const result = compileExplainerSource({
      source: fixtureSource,
      sourceName: "minimal-explainer.flow",
      capabilities: FOUNDATION_CAPABILITIES,
      strict: true,
    });

    expect(result.ok, JSON.stringify(result.diagnostics)).toBe(true);
    if (!result.ok) {
      return;
    }

    const pkg = result.value;
    expect(pkg.schemaVersion).toBe(1);
    expect(pkg.slides.length).toBeGreaterThan(0);

    for (const slide of pkg.slides) {
      expect(slide.narration.trim().length).toBeGreaterThan(0);

      if (slide.render !== undefined) {
        expect(slide.render.kind).toBe("scene");
        expect(slide.render.scene.roots.length).toBeGreaterThan(0);
        expect(slide.render.scene.timeline.length).toBeGreaterThan(0);
      }
    }
  });
});
