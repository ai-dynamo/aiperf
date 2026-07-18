// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, test, vi } from "vitest";

import {
  renderDisplayList,
  type CanvasRenderContext,
} from "../../src/backends/canvas/canvas-renderer.js";
import type { DisplayList } from "../../src/display-list.js";

type RecordedCall = readonly [name: string, ...args: unknown[]];

function recordingContext(): {
  readonly context: CanvasRenderContext;
  readonly calls: RecordedCall[];
} {
  const calls: RecordedCall[] = [];
  const context = {
    beginPath: () => calls.push(["beginPath"]),
    closePath: () => calls.push(["closePath"]),
    fill: () => calls.push(["fill"]),
    fillRect: (...args: unknown[]) => calls.push(["fillRect", ...args]),
    fillText: (...args: unknown[]) => calls.push(["fillText", ...args]),
    lineTo: (...args: unknown[]) => calls.push(["lineTo", ...args]),
    moveTo: (...args: unknown[]) => calls.push(["moveTo", ...args]),
    restore: () => calls.push(["restore"]),
    save: () => calls.push(["save"]),
    scale: (...args: unknown[]) => calls.push(["scale", ...args]),
    stroke: () => calls.push(["stroke"]),
    measureText: () =>
      ({
        width: 0,
        actualBoundingBoxAscent: 0,
        actualBoundingBoxDescent: 0,
      }) as TextMetrics,
  } as unknown as CanvasRenderContext;
  return { context, calls };
}

function displayList(commands: readonly unknown[]): DisplayList {
  return {
    commands,
    hitRegions: [],
    paintBounds: { x: 0, y: 0, width: 100, height: 50 },
    damageBounds: { x: 0, y: 0, width: 100, height: 50 },
  } as unknown as DisplayList;
}

describe("Canvas display-list renderer", () => {
  test("dispatches rect, text, line, and path commands in list order", () => {
    const { context, calls } = recordingContext();

    renderDisplayList(
      context,
      displayList([
        {
          kind: "rect",
          id: "background",
          order: 0,
          bounds: { x: 1, y: 2, width: 30, height: 12 },
        },
        {
          kind: "text",
          id: "label",
          order: 1,
          text: "Runtime",
          origin: { x: 4, y: 10 },
          font: { family: "Inter", sizePx: 16 },
        },
        {
          kind: "line",
          id: "connector",
          order: 2,
          from: { x: 2, y: 3 },
          to: { x: 20, y: 21 },
        },
        {
          kind: "path",
          id: "route",
          order: 3,
          path: "M 0 0 L 10 5 H 20 V 15 Z",
        },
      ]),
    );

    expect(calls).toEqual([
      ["fillRect", 1, 2, 30, 12],
      ["fillText", "Runtime", 4, 10],
      ["beginPath"],
      ["moveTo", 2, 3],
      ["lineTo", 20, 21],
      ["stroke"],
      ["beginPath"],
      ["moveTo", 0, 0],
      ["lineTo", 10, 5],
      ["lineTo", 20, 5],
      ["lineTo", 20, 15],
      ["closePath"],
      ["stroke"],
    ]);
  });

  test("applies device-pixel scaling around one deterministic frame", () => {
    const { context, calls } = recordingContext();

    const metrics = renderDisplayList(
      context,
      displayList([
        {
          kind: "rect",
          id: "box",
          order: 0,
          bounds: { x: 0, y: 0, width: 10, height: 10 },
        },
      ]),
      { devicePixelRatio: 2 },
    );

    expect(calls).toEqual([
      ["save"],
      ["scale", 2, 2],
      ["fillRect", 0, 0, 10, 10],
      ["restore"],
    ]);
    expect(metrics).toEqual({ commandCount: 1 });
  });

  test("measures repeated text through one cached Canvas atlas entry", () => {
    const recorded = recordingContext();
    const measureText = vi.fn(
      () =>
        ({
          width: 56,
          actualBoundingBoxAscent: 11,
          actualBoundingBoxDescent: 3,
        }) as TextMetrics,
    );
    const context = {
      ...recorded.context,
      measureText,
    };
    const text = {
      kind: "text",
      text: "Runtime",
      font: { family: "NVIDIA Sans", sizePx: 16, weight: 600 },
      origin: { x: 4, y: 10 },
    };

    renderDisplayList(
      context,
      displayList([
        { ...text, id: "first", order: 0 },
        { ...text, id: "second", order: 1 },
      ]),
    );

    expect(measureText).toHaveBeenCalledOnce();
    expect(measureText).toHaveBeenCalledWith("Runtime");
    expect(context.font).toBe('600 16px "NVIDIA Sans"');
  });
});
