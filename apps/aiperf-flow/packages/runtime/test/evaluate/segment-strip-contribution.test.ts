// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, test, vi } from "vitest";

import { contributeSegmentStrip } from "../../src/evaluate/contributions/segment-strip.js";

const layout = {
  originX: 10,
  originY: 20,
  rowHeight: 24,
  gap: 2,
  unitWidth: 3,
  seed: 42,
} as const;

describe("contributeSegmentStrip", () => {
  test("projects leaf geometry into ordered commands, semantics, and hit regions", () => {
    const contribution = contributeSegmentStrip({
      id: "prompt",
      segments: [
        { id: "system", tokens: 2, role: "system" },
        { id: "user", tokens: 3, role: "user" },
      ],
      layout,
      order: 7,
    });

    expect(contribution.commands).toEqual([
      {
        kind: "group",
        id: "prompt:system",
        order: 7,
        paintBounds: { x: 10, y: 20, width: 6, height: 24 },
        damageBounds: { x: 10, y: 20, width: 6, height: 24 },
        children: [
          {
            kind: "path",
            id: "prompt:system:rect",
            order: 0,
            paintBounds: { x: 10, y: 20, width: 6, height: 24 },
            damageBounds: { x: 10, y: 20, width: 6, height: 24 },
            path: "M 10 20 H 16 V 44 H 10 Z",
            fill: "#334155",
          },
          {
            kind: "text",
            id: "prompt:system:label",
            order: 1,
            paintBounds: { x: 10, y: 20, width: 6, height: 24 },
            damageBounds: { x: 10, y: 20, width: 6, height: 24 },
            text: "system",
            origin: { x: 14, y: 36 },
            font: { family: "sans-serif", sizePx: 12 },
            fill: "#f8fafc",
          },
        ],
      },
      {
        kind: "group",
        id: "prompt:user",
        order: 8,
        paintBounds: { x: 18, y: 20, width: 9, height: 24 },
        damageBounds: { x: 18, y: 20, width: 9, height: 24 },
        children: [
          {
            kind: "path",
            id: "prompt:user:rect",
            order: 0,
            paintBounds: { x: 18, y: 20, width: 9, height: 24 },
            damageBounds: { x: 18, y: 20, width: 9, height: 24 },
            path: "M 18 20 H 27 V 44 H 18 Z",
            fill: "#334155",
          },
          {
            kind: "text",
            id: "prompt:user:label",
            order: 1,
            paintBounds: { x: 18, y: 20, width: 9, height: 24 },
            damageBounds: { x: 18, y: 20, width: 9, height: 24 },
            text: "user",
            origin: { x: 22, y: 36 },
            font: { family: "sans-serif", sizePx: 12 },
            fill: "#f8fafc",
          },
        ],
      },
    ]);
    expect(contribution.semanticEntities).toEqual([
      {
        id: "system",
        label: "system",
        role: "system",
        kind: "segment",
        description: "2 tokens",
      },
      {
        id: "user",
        label: "user",
        role: "user",
        kind: "segment",
        description: "3 tokens",
      },
    ]);
    expect(contribution.hitRegions).toEqual([
      {
        id: "prompt:system:hit",
        semanticId: "system",
        order: 7,
        bounds: { x: 10, y: 20, width: 6, height: 24 },
      },
      {
        id: "prompt:user:hit",
        semanticId: "user",
        order: 8,
        bounds: { x: 18, y: 20, width: 9, height: 24 },
      },
    ]);
  });

  test("represents truncation and continuation in display and semantics", () => {
    const contribution = contributeSegmentStrip({
      id: "prompt",
      segments: [
        {
          id: "history",
          tokens: 4,
          role: "history",
          truncated: true,
          reused: true,
        },
      ],
      layout,
    });

    expect(contribution.commands).toEqual([
      {
        kind: "clip",
        id: "prompt:history",
        order: 0,
        paintBounds: { x: 10, y: 20, width: 12, height: 24 },
        damageBounds: { x: 10, y: 20, width: 12, height: 24 },
        path: "M 10 20 H 22 V 44 H 10 Z",
        children: [
          expect.objectContaining({ id: "prompt:history:rect", order: 0 }),
          expect.objectContaining({ id: "prompt:history:label", order: 1 }),
          {
            kind: "path",
            id: "prompt:history:continuation",
            order: 2,
            paintBounds: { x: 10, y: 20, width: 12, height: 24 },
            damageBounds: { x: 10, y: 20, width: 12, height: 24 },
            path: "M 10 20 L 14 32 L 10 44",
            stroke: "#38bdf8",
            strokeWidth: 2,
          },
        ],
      },
    ]);
    expect(contribution.semanticEntities).toEqual([
      {
        id: "history",
        label: "history",
        role: "history",
        kind: "segment",
        description: "4 tokens; truncated; continuation",
      },
    ]);
  });

  test("returns deeply immutable output without reading wall time", () => {
    const dateNow = vi.spyOn(Date, "now").mockImplementation(() => {
      throw new Error("wall time must not be read");
    });

    const contribution = contributeSegmentStrip({
      id: "prompt",
      segments: [{ id: "user", tokens: 1, role: "user" }],
      layout,
    });

    expect(dateNow).not.toHaveBeenCalled();
    expect(Object.isFrozen(contribution)).toBe(true);
    expect(Object.isFrozen(contribution.commands)).toBe(true);
    expect(Object.isFrozen(contribution.commands[0])).toBe(true);
    expect(Object.isFrozen(contribution.semanticEntities[0])).toBe(true);
    expect(Object.isFrozen(contribution.hitRegions[0]?.bounds)).toBe(true);
  });
});
