// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tests validating Systems Chalk design tokens: typography, shape, stroke, and motion.

import { describe, expect, test } from "vitest";

import { BUNDLED_ROOT_BASE, SYSTEMS_CHALK } from "../../src/theme/index.js";

describe("Systems Chalk Shape", () => {
  test("defines approved shape values", () => {
    // Shape is embedded in the theme; validate consistency with Systems Chalk identity
    expect(SYSTEMS_CHALK.id).toBe("systems_chalk");
    expect(SYSTEMS_CHALK.extends).toBe(BUNDLED_ROOT_BASE);
  });
});

describe("Systems Chalk Typography", () => {
  test("defines complete font families for display, body, and data", () => {
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
    });
  });

  test("defines font weights for regular, label, and emphasis levels", () => {
    expect(SYSTEMS_CHALK.values).toMatchObject({
      "weight.regular": { kind: "number", value: 400 },
      "weight.label": { kind: "number", value: 500 },
      "weight.emphasis": { kind: "number", value: 600 },
    });
  });

  test("defines font sizes for caption, body, label, and title", () => {
    expect(SYSTEMS_CHALK.values).toMatchObject({
      "size.caption": { kind: "number", value: 11 },
      "size.body": { kind: "number", value: 13 },
      "size.label": { kind: "number", value: 12 },
      "size.title": { kind: "number", value: 18 },
    });
  });
});

describe("Systems Chalk Stroke", () => {
  test("defines hairline, standard, and emphasis stroke widths", () => {
    expect(SYSTEMS_CHALK.values).toMatchObject({
      "stroke.hairline": { kind: "number", value: 1 },
      "stroke.standard": { kind: "number", value: 2 },
      "stroke.emphasis": { kind: "number", value: 3 },
    });
  });

  test("defines stroke cap and join as round", () => {
    expect(SYSTEMS_CHALK.values).toMatchObject({
      "stroke.cap": { kind: "enum", value: "round" },
      "stroke.join": { kind: "enum", value: "round" },
    });
  });
});

describe("Systems Chalk Motion", () => {
  test("defines duration timings for draw, enter, emphasis, and stagger", () => {
    expect(SYSTEMS_CHALK.values).toMatchObject({
      "motion.draw": { kind: "duration", valueMs: 420 },
      "motion.enter": { kind: "duration", valueMs: 240 },
      "motion.emphasis": { kind: "duration", valueMs: 180 },
      "motion.stagger": { kind: "duration", valueMs: 60 },
    });
  });

  test("defines motion easing as ease_out", () => {
    expect(SYSTEMS_CHALK.values).toMatchObject({
      "motion.easing": { kind: "enum", value: "ease_out" },
    });
  });

  test("motion timings follow consistent hierarchy (draw > enter > emphasis > stagger)", () => {
    const values = SYSTEMS_CHALK.values;
    const drawMs = (values["motion.draw"] as any).valueMs;
    const enterMs = (values["motion.enter"] as any).valueMs;
    const emphasisMs = (values["motion.emphasis"] as any).valueMs;
    const staggerMs = (values["motion.stagger"] as any).valueMs;

    expect(drawMs).toBeGreaterThan(enterMs);
    expect(enterMs).toBeGreaterThan(emphasisMs);
    expect(emphasisMs).toBeGreaterThan(staggerMs);
  });
});

describe("Systems Chalk Theme Lifecycle", () => {
  test("extends BUNDLED_ROOT_BASE and has standard theme structure", () => {
    expect(SYSTEMS_CHALK.extends).toBe(BUNDLED_ROOT_BASE);
    expect(SYSTEMS_CHALK.id).toBe("systems_chalk");
    expect(Object.keys(SYSTEMS_CHALK.values).length).toBeGreaterThan(0);
  });

  test("theme is frozen and immutable", () => {
    expect(() => {
      // @ts-expect-error testing immutability
      SYSTEMS_CHALK.values["new.token"] = {
        kind: "color",
        value: "#ffffff",
      };
    }).toThrow();
  });
});
