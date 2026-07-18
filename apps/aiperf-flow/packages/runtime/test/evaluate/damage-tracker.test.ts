// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, test } from "vitest";

import {
  buildDisplayList,
  type Bounds,
  type DisplayList,
  type DrawCommand,
} from "../../src/display-list.js";
import {
  computeDamageBetween,
  mergeDamageRegions,
} from "../../src/evaluate/damage-tracker.js";

const emptyBounds: Bounds = { x: 0, y: 0, width: 0, height: 0 };

function pathCommand(
  id: string,
  damageBounds: Bounds,
  order = 0,
  fill = "#fff",
): DrawCommand {
  return {
    kind: "path",
    id,
    order,
    path: `M ${damageBounds.x} ${damageBounds.y}`,
    fill,
    paintBounds: damageBounds,
    damageBounds,
  };
}

function displayList(commands: readonly DrawCommand[]): DisplayList {
  return buildDisplayList({
    commands,
    hitRegions: [],
    paintBounds: emptyBounds,
    damageBounds: emptyBounds,
  });
}

describe("mergeDamageRegions", () => {
  test("returns deterministic minimal supersets for overlapping regions", () => {
    const regions: readonly Bounds[] = [
      { x: 10, y: 0, width: 10, height: 10 },
      { x: 0, y: 0, width: 12, height: 10 },
      { x: 2, y: 2, width: 2, height: 2 },
      { x: 40, y: 5, width: 4, height: 4 },
    ];

    const expected = [
      { x: 0, y: 0, width: 20, height: 10 },
      { x: 40, y: 5, width: 4, height: 4 },
    ];

    expect(mergeDamageRegions(regions)).toEqual(expected);
    expect(mergeDamageRegions([...regions].reverse())).toEqual(expected);
  });

  test("transitively merges touching regions without spanning gaps", () => {
    expect(
      mergeDamageRegions([
        { x: 4, y: 0, width: 4, height: 4 },
        { x: 12, y: 0, width: 4, height: 4 },
        { x: 0, y: 0, width: 4, height: 4 },
        { x: 8, y: 0, width: 4, height: 4 },
        { x: 30, y: 0, width: 4, height: 4 },
      ]),
    ).toEqual([
      { x: 0, y: 0, width: 16, height: 4 },
      { x: 30, y: 0, width: 4, height: 4 },
    ]);
  });
});

describe("computeDamageBetween", () => {
  test("returns no damage for equivalent display lists", () => {
    const previous = displayList([
      pathCommand("semantic", { x: 0, y: 0, width: 20, height: 10 }),
    ]);
    const current = displayList([
      pathCommand("semantic", { x: 0, y: 0, width: 20, height: 10 }),
    ]);

    expect(computeDamageBetween(previous, current)).toEqual([]);
  });

  test("covers both old and new bounds when a command changes", () => {
    const previous = displayList([
      pathCommand("semantic", { x: 0, y: 0, width: 10, height: 10 }),
    ]);
    const current = displayList([
      pathCommand("semantic", { x: 6, y: 0, width: 10, height: 10 }),
    ]);

    expect(computeDamageBetween(previous, current)).toEqual([
      { x: 0, y: 0, width: 16, height: 10 },
    ]);
  });

  test("limits damage to a nested decorative command removed by quality policy", () => {
    const semanticBounds = { x: 0, y: 0, width: 100, height: 40 };
    const decorativeBounds = { x: 8, y: 6, width: 12, height: 12 };
    const previous = displayList([
      {
        kind: "group",
        id: "semantic-group",
        order: 0,
        paintBounds: semanticBounds,
        damageBounds: semanticBounds,
        children: [
          pathCommand("semantic", semanticBounds),
          pathCommand("decorative-glow", decorativeBounds, 1),
        ],
      },
    ]);
    const current = displayList([
      {
        kind: "group",
        id: "semantic-group",
        order: 0,
        paintBounds: semanticBounds,
        damageBounds: semanticBounds,
        children: [pathCommand("semantic", semanticBounds)],
      },
    ]);

    expect(computeDamageBetween(previous, current)).toEqual([
      decorativeBounds,
    ]);
  });

  test("tracks additions and removals independently of command ordering", () => {
    const removedBounds = { x: 0, y: 0, width: 5, height: 5 };
    const addedBounds = { x: 20, y: 0, width: 5, height: 5 };
    const stableBounds = { x: 10, y: 0, width: 5, height: 5 };
    const previous = displayList([
      pathCommand("removed", removedBounds, 0),
      pathCommand("stable", stableBounds, 1),
    ]);
    const current = displayList([
      pathCommand("stable", stableBounds, 0),
      pathCommand("added", addedBounds, 1),
    ]);

    expect(computeDamageBetween(previous, current)).toEqual([
      removedBounds,
      stableBounds,
      addedBounds,
    ]);
  });
});
