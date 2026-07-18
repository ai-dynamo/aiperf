// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, test } from "vitest";

import {
  createHitRegionIndex,
  pickHitRegions,
} from "../../src/evaluate/hit-region-index.js";
import type {
  DisplayList,
  HitRegion,
  Point,
} from "../../src/display-list.js";

const background: HitRegion = {
  id: "background-hit",
  semanticId: "background",
  order: 0,
  bounds: { x: 0, y: 0, width: 100, height: 100 },
};

const foreground: HitRegion = {
  id: "foreground-hit",
  semanticId: "request-7",
  order: 2,
  bounds: { x: 20, y: 20, width: 40, height: 40 },
};

function displayList(hitRegions: readonly HitRegion[]): DisplayList {
  return {
    commands: [],
    hitRegions,
    paintBounds: { x: 0, y: 0, width: 100, height: 100 },
    damageBounds: { x: 0, y: 0, width: 100, height: 100 },
  };
}

function pickedIds(list: DisplayList, point: Point): readonly string[] {
  return pickHitRegions(createHitRegionIndex(list), point).map(
    (region) => region.id,
  );
}

describe("hit-region index", () => {
  test("returns overlapping regions from top-most to bottom-most", () => {
    const index = createHitRegionIndex(displayList([background, foreground]));

    expect(pickHitRegions(index, { x: 30, y: 30 })).toEqual([
      foreground,
      background,
    ]);
  });

  test("breaks equal visual-order ties deterministically by id", () => {
    const alpha = {
      ...background,
      id: "alpha",
      semanticId: "alpha",
      order: 1,
    };
    const zebra = {
      ...background,
      id: "zebra",
      semanticId: "zebra",
      order: 1,
    };
    const point = { x: 50, y: 50 };

    expect(pickedIds(displayList([alpha, zebra]), point)).toEqual([
      "zebra",
      "alpha",
    ]);
    expect(pickedIds(displayList([zebra, alpha]), point)).toEqual([
      "zebra",
      "alpha",
    ]);
  });

  test("preserves authored keyboard traversal independently of visual ties", () => {
    const first = {
      ...background,
      id: "zebra",
      semanticId: "first",
      order: 1,
    };
    const second = {
      ...background,
      id: "alpha",
      semanticId: "second",
      order: 1,
    };
    const index = createHitRegionIndex(displayList([first, second]));

    expect(index.keyboardTraversal.map((region) => region.semanticId)).toEqual([
      "first",
      "second",
    ]);
    expect(pickHitRegions(index, { x: 50, y: 50 }).map(({ id }) => id)).toEqual([
      "zebra",
      "alpha",
    ]);
  });

  test("includes bounds edges and returns no matches outside them", () => {
    const index = createHitRegionIndex(displayList([foreground]));

    expect(pickHitRegions(index, { x: 60, y: 60 })).toEqual([foreground]);
    expect(pickHitRegions(index, { x: 61, y: 60 })).toEqual([]);
  });
});
