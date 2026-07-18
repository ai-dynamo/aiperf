// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, test, vi } from "vitest";

import {
  CanvasTextAtlas,
  canvasFont,
  type TextMeasurementContext,
} from "../../src/backends/canvas/text-atlas.js";

function measurementContext(
  measureText: TextMeasurementContext["measureText"],
): TextMeasurementContext {
  return {
    font: "",
    measureText,
  };
}

describe("Canvas text atlas", () => {
  test("uses a canonical Canvas font string", () => {
    expect(canvasFont({ family: "Inter", sizePx: 16 })).toBe("16px Inter");
    expect(canvasFont({ family: "NVIDIA Sans", sizePx: 14, weight: 600 })).toBe(
      '600 14px "NVIDIA Sans"',
    );
  });

  test("measures each unique text and font tuple once", () => {
    const measureText = vi.fn((text: string) => ({
      width: text.length * 8,
      actualBoundingBoxAscent: 11,
      actualBoundingBoxDescent: 3,
    })) as TextMeasurementContext["measureText"];
    const context = measurementContext(measureText);
    const atlas = new CanvasTextAtlas(context);
    const font = { family: "Inter", sizePx: 16, weight: 600 } as const;

    const first = atlas.measure("Runtime", font);
    const second = atlas.measure("Runtime", { ...font });

    expect(first).toBe(second);
    expect(first).toEqual({
      width: 56,
      actualBoundingBoxAscent: 11,
      actualBoundingBoxDescent: 3,
    });
    expect(Object.isFrozen(first)).toBe(true);
    expect(measureText).toHaveBeenCalledOnce();
    expect(context.font).toBe("600 16px Inter");
    expect(atlas.size).toBe(1);
  });

  test("keeps different text and font tuples separate", () => {
    const measureText = vi.fn((text: string) => ({
      width: text.length,
      actualBoundingBoxAscent: 4,
      actualBoundingBoxDescent: 1,
    })) as TextMeasurementContext["measureText"];
    const atlas = new CanvasTextAtlas(measurementContext(measureText));

    atlas.measure("A", { family: "Inter", sizePx: 12 });
    atlas.measure("B", { family: "Inter", sizePx: 12 });
    atlas.measure("A", { family: "Inter", sizePx: 14 });

    expect(measureText).toHaveBeenCalledTimes(3);
    expect(atlas.size).toBe(3);
  });

  test("rejects invalid fonts and non-finite measurements", () => {
    const atlas = new CanvasTextAtlas(
      measurementContext(
        () =>
          ({
            width: Number.NaN,
            actualBoundingBoxAscent: 4,
            actualBoundingBoxDescent: 1,
          }) as TextMetrics,
      ),
    );

    expect(() =>
      canvasFont({ family: "", sizePx: 16 }),
    ).toThrowError("font family must not be empty");
    expect(() =>
      canvasFont({ family: "Inter", sizePx: 0 }),
    ).toThrowError("font size must be a positive finite number");
    expect(() =>
      atlas.measure("A", { family: "Inter", sizePx: 16 }),
    ).toThrowError("Canvas text measurement must be finite");
    expect(atlas.size).toBe(0);
  });

  test("clears cached measurements deterministically", () => {
    const measureText = vi.fn(
      () =>
        ({
          width: 8,
          actualBoundingBoxAscent: 6,
          actualBoundingBoxDescent: 2,
        }) as TextMetrics,
    );
    const atlas = new CanvasTextAtlas(measurementContext(measureText));
    const font = { family: "Inter", sizePx: 12 } as const;

    atlas.measure("A", font);
    atlas.clear();
    atlas.measure("A", font);

    expect(atlas.size).toBe(1);
    expect(measureText).toHaveBeenCalledTimes(2);
  });
});
