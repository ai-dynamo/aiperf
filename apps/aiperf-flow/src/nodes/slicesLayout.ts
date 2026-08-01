/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Geometry and binning for the `slices` node type: a uniform grid laid over a Gantt, showing
//! which bucket each interval falls into and where the grid overruns real activity.

const MARGIN_L = 40;
const MARGIN_R = 20;
const TOP = 10;
const ROW_H = 22;
const AXIS_GAP = 6;
const AXIS_LABEL_H = 40;
const PAD_X = 16;
const PAD_Y = 14;
/** Title `<div>`: its pinned `leading-[24px]` line box plus the `mb-1.5 (6px)` gap below it.
 * The component pins that leading so this stays a contract rather than a font-metric guess. */
const TITLE_H = 30;
/** The 1px `border` on each side of the node chrome. */
const BORDER = 2;

export const DEFAULT_SLICES_WIDTH = 760;

/** One interval placed on the grid. */
export type SliceRequest = {
  id: string;
  start: number;
  end: number;
};

/** A bucket of the uniform grid. `isComplete` is false when it runs past real activity. */
export type Slice = {
  index: number;
  start: number;
  /** Grid-defined end, which may exceed `spanEnd`. */
  end: number;
  /** `end` clipped to real activity — what a rate metric should divide by. */
  clippedEnd: number;
  isComplete: boolean;
};

/**
 * Cut `[spanStart, spanEnd]` into buckets of `duration`.
 *
 * The trailing bucket is flagged rather than dropped: a rate computed over the grid-defined width
 * would be diluted by idle padding, so a consumer needs to see that it is short.
 */
export function buildSlices(spanStart: number, spanEnd: number, duration: number): Slice[] {
  if (duration <= 0 || spanEnd <= spanStart) return [];
  const count = Math.ceil((spanEnd - spanStart) / duration);
  return Array.from({ length: count }, (_, index) => {
    const start = spanStart + index * duration;
    const end = start + duration;
    return { index, start, end, clippedEnd: Math.min(end, spanEnd), isComplete: end <= spanEnd };
  });
}

/** Which bucket an interval is binned into — by its start, the binning key. */
export function binOf(start: number, spanStart: number, duration: number, count: number): number {
  if (count <= 0) return 0;
  return Math.min(count - 1, Math.max(0, Math.floor((start - spanStart) / duration)));
}

export type SlicesLayoutInput = {
  requests: readonly SliceRequest[];
  duration: number;
  hasTitle: boolean;
  width?: number;
};

export type SlicesLayout = {
  x: (t: number) => number;
  spanStart: number;
  spanEnd: number;
  tMax: number;
  slices: Slice[];
  xLeft: number;
  xRight: number;
  top: number;
  rowHeight: number;
  ganttHeight: number;
  axisY: number;
  svgWidth: number;
  svgHeight: number;
  nodeWidth: number;
  nodeHeight: number;
};

export function layoutSlices({
  requests,
  duration,
  hasTitle,
  width = DEFAULT_SLICES_WIDTH,
}: SlicesLayoutInput): SlicesLayout {
  const spanStart = requests.length === 0 ? 0 : Math.min(...requests.map((r) => r.start));
  const spanEnd = Math.max(spanStart + 1, ...requests.map((r) => r.end));
  const slices = buildSlices(spanStart, spanEnd, duration);
  // The axis must reach the grid's end, not just activity's, or an incomplete trailing slice
  // would be drawn off the right edge — which is the one thing this chart exists to show.
  const tMax = Math.max(spanEnd, slices.length > 0 ? slices[slices.length - 1]!.end : spanEnd);

  const ganttHeight = requests.length * ROW_H;
  const axisY = TOP + ganttHeight + AXIS_GAP;
  const svgHeight = axisY + AXIS_LABEL_H;
  const xLeft = MARGIN_L;
  const xRight = width - MARGIN_R;

  return {
    x: (t: number) => xLeft + ((t - spanStart) / (tMax - spanStart)) * (xRight - xLeft),
    spanStart,
    spanEnd,
    tMax,
    slices,
    xLeft,
    xRight,
    top: TOP,
    rowHeight: ROW_H,
    ganttHeight,
    axisY,
    svgWidth: width,
    svgHeight,
    nodeWidth: width + 2 * PAD_X + BORDER,
    nodeHeight: svgHeight + 2 * PAD_Y + BORDER + (hasTitle ? TITLE_H : 0),
  };
}

/** The box a `slices` node will occupy, for the authoring helper's `style`. */
export function slicesNodeSize(input: SlicesLayoutInput): { width: number; height: number } {
  const layout = layoutSlices(input);
  return { width: layout.nodeWidth, height: layout.nodeHeight };
}
