// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, test } from "vitest";

import { measureGlyphRun } from "../../src/leaves/glyph-measure.js";

describe("leaf.glyph-measure", () => {
  test("segments café 🚀 into grapheme units with byte ranges", () => {
    const text = "café 🚀";
    const measured = measureGlyphRun("glyph-run-cafe-rocket", text);

    expect(measured.graphemes.map(({ id, text: grapheme }) => ({ id, text: grapheme }))).toEqual([
      { id: "g0", text: "c" },
      { id: "g1", text: "a" },
      { id: "g2", text: "f" },
      { id: "g3", text: "é" },
      { id: "g4", text: " " },
      { id: "g5", text: "🚀" },
    ]);

    const accent = measured.graphemes[3];
    expect(accent?.byteStart).toBe(3);
    expect(accent?.byteEnd).toBe(5);

    const rocket = measured.graphemes[5];
    expect(rocket?.byteStart).toBe(6);
    expect(rocket?.byteEnd).toBe(10);
  });
});
