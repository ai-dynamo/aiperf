/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Geometry and rank derivation for the `intervals` node type.
//!
//! Split from the component for the same reason as `timelineLayout`: the authoring helper
//! must know the node's box before React Flow measures it, or `Slide`'s reveal-driven
//! `fitView` reframes the diagram mid-cascade.

import type { CategoryRole } from "../theme/tokens.js";

/** One recorded request as an interval on a single clock. */
export type IntervalRow = {
  id: string;
  /** Text drawn inside the bar; the id is drawn in the left gutter. */
  label: string;
  start: number;
  end: number;
  role: CategoryRole;
  /** Dashed outline — an async-launched interval that never serializes a successor. */
  dashed?: boolean;
  /** Overrides the derived global rank. */
  rank?: number;
};

/**
 * Global rank: the position of each interval in `sort(start, end, id)`.
 *
 * This is the total order interval-order edge derivation breaks ties with, so the badge is
 * showing a real quantity rather than a row number — the rows may be authored in any order.
 */
export function intervalRanks(rows: readonly IntervalRow[]): Map<string, number> {
  const sorted = [...rows].sort(
    (a, b) => a.start - b.start || a.end - b.end || (a.id < b.id ? -1 : 1),
  );
  return new Map(sorted.map((row, i) => [row.id, i]));
}

/** Resolve each row's badge number, honouring an explicit `rank` override. */
export function resolveRanks(rows: readonly IntervalRow[]): Map<string, number> {
  const derived = intervalRanks(rows);
  return new Map(rows.map((row) => [row.id, row.rank ?? derived.get(row.id)!]));
}

/** Gutter reserved for row ids. */
const LEFT = 78;
const ROW_H = 26;
const ROW_GAP = 8;
const TOP = 12;
/** Matches the `px-4 py-3.5` chrome `Card`/`Panel` use. */
const PAD_X = 16;
const PAD_Y = 14;
const TITLE_H = 26;
const BORDER = 2;

export const DEFAULT_INTERVALS_WIDTH = 720;

export type IntervalsLayoutInput = {
  rows: readonly IntervalRow[];
  hasTitle: boolean;
  width?: number;
};

export type IntervalsLayout = {
  px: number;
  x: (t: number) => number;
  /** Half a second of headroom past the last end, so a badge at the right edge is not clipped. */
  maxEnd: number;
  rowHeight: number;
  rowY: (i: number) => number;
  gridTicks: number[];
  gridTop: number;
  gridBottom: number;
  svgWidth: number;
  svgHeight: number;
  nodeWidth: number;
  nodeHeight: number;
};

export function layoutIntervals({
  rows,
  hasTitle,
  width = DEFAULT_INTERVALS_WIDTH,
}: IntervalsLayoutInput): IntervalsLayout {
  const maxEnd = Math.max(1, ...rows.map((r) => r.end)) + 0.5;
  const px = Math.max(40, Math.min(78, Math.floor((width - LEFT) / maxEnd)));
  const x = (t: number) => LEFT + t * px;

  const gridBottom = TOP + rows.length * (ROW_H + ROW_GAP);
  // Ceil: `maxEnd` carries a half-second of headroom, so an integer `px` still lands the raw
  // width on a half pixel, and that fraction would reach `style.width` through the helper.
  const svgWidth = Math.ceil(x(maxEnd) + 16);
  const svgHeight = gridBottom + 26;

  const gridMax = Math.ceil(maxEnd);
  const gridTicks = Array.from({ length: gridMax + 1 }, (_, t) => t);

  return {
    px,
    x,
    maxEnd,
    rowHeight: ROW_H,
    rowY: (i: number) => TOP + i * (ROW_H + ROW_GAP),
    gridTicks,
    gridTop: TOP - 4,
    gridBottom,
    svgWidth,
    svgHeight,
    nodeWidth: svgWidth + 2 * PAD_X + BORDER,
    nodeHeight: svgHeight + 2 * PAD_Y + BORDER + (hasTitle ? TITLE_H : 0),
  };
}

/** The box an `intervals` node will occupy, for the authoring helper's `style`. */
export function intervalsNodeSize(input: IntervalsLayoutInput): { width: number; height: number } {
  const layout = layoutIntervals(input);
  return { width: layout.nodeWidth, height: layout.nodeHeight };
}
