// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Bounds, HitRegion } from "../../display-list.js";

export type Point = Readonly<{
  x: number;
  y: number;
}>;

function contains(bounds: Bounds, point: Point): boolean {
  return (
    point.x >= bounds.x &&
    point.x <= bounds.x + bounds.width &&
    point.y >= bounds.y &&
    point.y <= bounds.y + bounds.height
  );
}

function compareRegions(left: HitRegion, right: HitRegion): number {
  return left.order - right.order || left.id.localeCompare(right.id);
}

/** Returns the topmost semantic hit region containing `point`, if any. */
export function hitTest(
  regions: readonly HitRegion[],
  point: Point,
): HitRegion | undefined {
  const matches = regions.filter((region) => contains(region.bounds, point));
  if (matches.length === 0) {
    return undefined;
  }
  return [...matches].sort(compareRegions)[matches.length - 1];
}
