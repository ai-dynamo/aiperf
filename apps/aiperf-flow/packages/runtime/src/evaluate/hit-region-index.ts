// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type {
  DisplayList,
  HitRegion,
  Point,
} from "../display-list.js";

/** Immutable point-picking index with authored keyboard traversal order. */
export type HitRegionIndex = Readonly<{
  keyboardTraversal: readonly HitRegion[];
  regionsByMinX: readonly HitRegion[];
}>;

function compareByMinX(left: HitRegion, right: HitRegion): number {
  return (
    left.bounds.x - right.bounds.x ||
    left.order - right.order ||
    left.id.localeCompare(right.id)
  );
}

function compareTopMost(left: HitRegion, right: HitRegion): number {
  return right.order - left.order || right.id.localeCompare(left.id);
}

function contains(region: HitRegion, point: Point): boolean {
  const { bounds } = region;
  return (
    point.x >= bounds.x &&
    point.x <= bounds.x + bounds.width &&
    point.y >= bounds.y &&
    point.y <= bounds.y + bounds.height
  );
}

/**
 * Creates an immutable index without deriving keyboard traversal from visual
 * z-order. The display list's authored hit-region sequence is traversal order.
 */
export function createHitRegionIndex(list: DisplayList): HitRegionIndex {
  return Object.freeze({
    keyboardTraversal: Object.freeze([...list.hitRegions]),
    regionsByMinX: Object.freeze([...list.hitRegions].sort(compareByMinX)),
  });
}

/** Returns all regions containing `point`, ordered from top-most to bottom-most. */
export function pickHitRegions(
  index: HitRegionIndex,
  point: Point,
): readonly HitRegion[] {
  const matches: HitRegion[] = [];

  for (const region of index.regionsByMinX) {
    if (region.bounds.x > point.x) {
      break;
    }
    if (contains(region, point)) {
      matches.push(region);
    }
  }

  return matches.sort(compareTopMost);
}
