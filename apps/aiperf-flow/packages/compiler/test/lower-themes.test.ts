/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { FOUNDATION_CAPABILITIES } from "@aiperf/flow-schema";
import { describe, expect, test } from "vitest";

import { compileSource } from "../src/index.js";
import { FOUNDATION_SOURCE } from "./fixture.js";

const THEMED_SOURCE = `flow "Lab" as lab {
  language 1
  require core.rect "^1.0.0"
  token accent = "#7aa2f7"

  theme lab_chalk extends systems_chalk {
    color accent.control = "#78dce8"
    number stroke.standard = 2
    duration motion.draw = 420ms
    font font.body = ["Nunito Sans", "sans-serif"]
    enum stroke.cap = "round"
  }

  use theme lab_chalk

  scene "Theme lowering" as main {
    summary "A themed rectangle verifies compiler lowering."
    rect router {
      x 0
      y 0
      width 10
      height 10
      fill theme(surface.raised)
      stroke theme(accent.control)
      label "Router"
      role "group"
      description "A router rendered with semantic theme roles."
      fallback "Router"
    }
    reading-order router
    narrate "The router uses semantic theme roles that remain unresolved for runtime selection."
    fallback "A single themed router."
  }
}
`;

describe("theme lowering", () => {
  test("preserves theme roles and emits authored themes without bundled values", () => {
    const result = compileSource({
      source: THEMED_SOURCE,
      sourceName: "lab.flow",
      capabilities: FOUNDATION_CAPABILITIES,
      strict: true,
    });

    expect(result.ok, JSON.stringify(result.diagnostics)).toBe(true);
    if (!result.ok) {
      return;
    }

    expect(result.value.irVersion).toBe(2);
    expect(result.value.defaultTheme).toBe("lab_chalk");
    expect(result.value.themes).toEqual([
      expect.objectContaining({
        id: "lab_chalk",
        extends: "systems_chalk",
        values: {
          "accent.control": { kind: "color", value: "#78dce8" },
          "font.body": { kind: "font", value: ["Nunito Sans", "sans-serif"] },
          "motion.draw": { kind: "duration", valueMs: 420 },
          "stroke.cap": { kind: "enum", value: "round" },
          "stroke.standard": { kind: "number", value: 2 },
        },
      }),
    ]);

    const router = result.value.scenes[0]?.roots.find((node) => node.id === "router");
    expect(router?.style.fill).toEqual({
      kind: "theme-role",
      role: "surface.raised",
    });
    expect(router?.style.stroke).toEqual({
      kind: "theme-role",
      role: "accent.control",
    });
  });

  test("still lowers token references to scalar literals", () => {
    const result = compileSource({
      source: FOUNDATION_SOURCE,
      sourceName: "request-flow.flow",
      capabilities: FOUNDATION_CAPABILITIES,
      strict: false,
    });

    expect(result.ok, JSON.stringify(result.diagnostics)).toBe(true);
    if (!result.ok) {
      return;
    }

    const cli = result.value.scenes[0]?.roots.find((node) => node.id === "cli");
    expect(cli?.style.fill).toBe("#7aa2f7");
  });
});
