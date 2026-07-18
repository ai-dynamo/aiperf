// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, test, vi } from "vitest";

import { contributeGlyphRun } from "../../src/evaluate/contributions/glyph-run.js";

describe("contributeGlyphRun", () => {
  test("emits one backend-neutral text command without semantic stubs by default", () => {
    const dateNow = vi.spyOn(Date, "now").mockImplementation(() => {
      throw new Error("wall time must not be read");
    });

    const contribution = contributeGlyphRun({
      id: "prompt",
      text: "Hello",
      bounds: { x: 12, y: 20, width: 90, height: 24 },
      origin: { x: 12, y: 39 },
      font: { family: "Inter", sizePx: 16, weight: 600 },
      fill: "#f8fafc",
      order: 7,
    });

    expect(contribution).toEqual({
      commands: [
        {
          kind: "text",
          id: "prompt:text",
          order: 7,
          paintBounds: { x: 12, y: 20, width: 90, height: 24 },
          damageBounds: { x: 12, y: 20, width: 90, height: 24 },
          text: "Hello",
          origin: { x: 12, y: 39 },
          font: { family: "Inter", sizePx: 16, weight: 600 },
          fill: "#f8fafc",
        },
      ],
    });
    expect(dateNow).not.toHaveBeenCalled();
  });

  test("uses measured grapheme boundaries for optional semantic entity stubs", () => {
    const contribution = contributeGlyphRun({
      id: "tokens",
      text: "A👩‍💻",
      locale: "en",
      bounds: { x: 0, y: 0, width: 100, height: 20 },
      origin: { x: 0, y: 16 },
      font: { family: "sans-serif", sizePx: 16 },
      semantics: { role: "text", description: "Prompt grapheme" },
    });

    expect(contribution.semanticEntities).toEqual([
      {
        id: "tokens:g0",
        label: "A",
        role: "text",
        description: "Prompt grapheme",
      },
      {
        id: "tokens:g1",
        label: "👩‍💻",
        role: "text",
        description: "Prompt grapheme",
      },
    ]);
  });
});
