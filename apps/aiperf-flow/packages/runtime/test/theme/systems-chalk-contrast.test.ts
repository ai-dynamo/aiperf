/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, test } from "vitest";

import {
  BUNDLED_ROOT_BASE,
  SYSTEMS_CHALK,
  SYSTEMS_CHALK_SHAPE,
  createBootstrapThemeRegistry,
} from "../../src/theme/index.js";

describe("Systems Chalk", () => {
  test("matches the approved typography, shape, stroke, and motion values", () => {
    expect(SYSTEMS_CHALK.extends).toBe(BUNDLED_ROOT_BASE);
    expect(SYSTEMS_CHALK_SHAPE).toEqual({ cornerRadiusPx: 12 });
    expect(SYSTEMS_CHALK.values).toMatchObject({
      "font.display": {
        kind: "font",
        value: ["Nunito Sans", "Segoe UI", "sans-serif"],
      },
      "font.body": {
        kind: "font",
        value: ["Nunito Sans", "Segoe UI", "sans-serif"],
      },
      "font.data": {
        kind: "font",
        value: ["IBM Plex Mono", "Cascadia Code", "monospace"],
      },
      "weight.regular": { kind: "number", value: 400 },
      "weight.label": { kind: "number", value: 500 },
      "weight.emphasis": { kind: "number", value: 600 },
      "size.caption": { kind: "number", value: 11 },
      "size.body": { kind: "number", value: 13 },
      "size.label": { kind: "number", value: 12 },
      "size.title": { kind: "number", value: 18 },
      "stroke.hairline": { kind: "number", value: 1 },
      "stroke.standard": { kind: "number", value: 2 },
      "stroke.emphasis": { kind: "number", value: 3 },
      "stroke.cap": { kind: "enum", value: "round" },
      "stroke.join": { kind: "enum", value: "round" },
      "motion.draw": { kind: "duration", valueMs: 420 },
      "motion.enter": { kind: "duration", valueMs: 240 },
      "motion.emphasis": { kind: "duration", valueMs: 180 },
      "motion.stagger": { kind: "duration", valueMs: 60 },
      "motion.easing": { kind: "enum", value: "ease_out" },
    });
  });

  test("passes every required WCAG contrast pair", () => {
    expect(() =>
      createBootstrapThemeRegistry().freeze().resolve("systems_chalk"),
    ).not.toThrow();
  });
});
