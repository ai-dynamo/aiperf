/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Geometry for the `sweep` node type: a Gantt stacked over the step function its intervals
//! generate, both on one time axis.
//!
//! Split from the component so the authoring helper can size the node before React Flow measures
//! it — see `timelineLayout` for why that matters to `Slide`'s reveal-driven `fitView`.

import { buildEvents, niceMax, stepPoints, type SweepCurveId, type SweepRequest } from "./sweepMath.js";

const MARGIN_L = 54;
const MARGIN_R = 20;
const TOP = 12;
const ROW_H = 20;
/** Clearance between the Gantt and the step plot, holding the separator rule. */
const GAP_H = 30;
const STEP_H = 150;
const AXIS_LABEL_H = 34;
const PAD_X = 16;
const PAD_Y = 14;
/** Title `<div>`: its pinned `leading-[24px]` line box plus the `mb-1.5 (6px)` gap below it.
 * The component pins that leading so this stays a contract rather than a font-metric guess. */
const TITLE_H = 30;
/** The 1px `border` on each side of the node chrome. */
const BORDER = 2;

export const DEFAULT_SWEEP_WIDTH = 760;

export type SweepLayoutInput = {
  requests: readonly SweepRequest[];
  curve: SweepCurveId;
  hasTitle: boolean;
  /** Right edge of the time axis. Defaults to the latest request end. */
  tMax?: number;
  width?: number;
};

export type SweepLayout = {
  x: (t: number) => number;
  y: (v: number) => number;
  tMax: number;
  /** Axis maximum, rounded up to a 1/2/5 x 10^n value. */
  vMax: number;
  xLeft: number;
  xRight: number;
  top: number;
  rowHeight: number;
  ganttHeight: number;
  stepTop: number;
  stepHeight: number;
  axisY: number;
  svgWidth: number;
  svgHeight: number;
  nodeWidth: number;
  nodeHeight: number;
  tTicks: number[];
  vTicks: number[];
};

/** Six evenly spaced time ticks, snapped to whole units when the span allows. */
function timeTicks(tMax: number): number[] {
  const step = tMax / 5;
  const snapped = step >= 1 ? Math.round(step) : step;
  const ticks: number[] = [];
  for (let t = 0; t <= tMax + 1e-9; t += snapped) ticks.push(Number(t.toFixed(6)));
  return ticks;
}

export function layoutSweep({
  requests,
  curve,
  hasTitle,
  tMax,
  width = DEFAULT_SWEEP_WIDTH,
}: SweepLayoutInput): SweepLayout {
  const resolvedMax = tMax ?? Math.max(1, ...requests.map((r) => r.end));
  const points = stepPoints(buildEvents(requests, curve));
  const vMax = niceMax(points.reduce((m, p) => Math.max(m, p.v), 0));

  const ganttHeight = requests.length * ROW_H;
  const stepTop = TOP + ganttHeight + GAP_H;
  const axisY = stepTop + STEP_H;
  const svgHeight = axisY + AXIS_LABEL_H;

  const xLeft = MARGIN_L;
  const xRight = width - MARGIN_R;

  return {
    x: (t: number) => xLeft + (t / resolvedMax) * (xRight - xLeft),
    y: (v: number) => stepTop + STEP_H - (v / vMax) * STEP_H,
    tMax: resolvedMax,
    vMax,
    xLeft,
    xRight,
    top: TOP,
    rowHeight: ROW_H,
    ganttHeight,
    stepTop,
    stepHeight: STEP_H,
    axisY,
    svgWidth: width,
    svgHeight,
    nodeWidth: width + 2 * PAD_X + BORDER,
    nodeHeight: svgHeight + 2 * PAD_Y + BORDER + (hasTitle ? TITLE_H : 0),
    tTicks: timeTicks(resolvedMax),
    vTicks: [0, vMax / 2, vMax],
  };
}

/** The box a `sweep` node will occupy, for the authoring helper's `style`. */
export function sweepNodeSize(input: SweepLayoutInput): { width: number; height: number } {
  const layout = layoutSweep(input);
  return { width: layout.nodeWidth, height: layout.nodeHeight };
}
