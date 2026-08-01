/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Geometry for the `timeline` node type, kept apart from the component so both the renderer
//! and the deck helper that sizes the node read the same numbers.
//!
//! A React Flow node that measures itself late makes `Slide`'s reveal-driven `fitView` reframe
//! mid-cascade, so a timeline declares its box up front: `layoutTimeline` is pure, and
//! `timelineNodeSize` gives the authoring helper the exact `style.width`/`style.height` the
//! rendered node will occupy.

import type { CategoryRole } from "../theme/tokens.js";

/** One request drawn on both clocks. Raw is what was recorded; warped is what a runtime replays. */
export type TimelineBar = {
  id: string;
  /** Swimlane this bar belongs to; must appear in `TimelineNodeData.lanes`. */
  lane: string;
  rawStart: number;
  rawEnd: number;
  warpStart: number;
  warpEnd: number;
};

/** True dead air: running-max end to the next start, classified against the idle cap. */
export type TimelineGap = {
  start: number;
  end: number;
  idle: number;
  /** Drawn as a warning band — this gap exceeds the cap and is what the warp collapses. */
  capped: boolean;
};

/** Lane hues, cycled. Mirrors the palette the interactive weka deck uses for subagent lanes. */
export const TIMELINE_LANE_ROLES: readonly CategoryRole[] = [
  "blue",
  "green",
  "purple",
  "orange",
  "red",
  "cyan",
  "yellow",
  "gray",
];

export function laneRole(lane: string, lanes: readonly string[]): CategoryRole {
  const i = Math.max(0, lanes.indexOf(lane));
  return TIMELINE_LANE_ROLES[i % TIMELINE_LANE_ROLES.length]!;
}

/** Gutter reserved for lane labels. */
const LEFT = 92;
const LANE_H = 24;
const LANE_GAP = 6;
/** Matches the `px-4 py-3.5` chrome `Card`/`Panel` use, so a timeline sits flush beside them. */
const PAD_X = 16;
const PAD_Y = 14;
const TITLE_H = 26;
const BORDER = 2;

/** Default drawing width before the per-second scale is clamped. */
export const DEFAULT_TIMELINE_WIDTH = 720;

export type TimelineLayoutInput = {
  lanes: readonly string[];
  bars: readonly TimelineBar[];
  showWarp: boolean;
  hasTitle: boolean;
  width?: number;
};

export type TimelineLayout = {
  /** Pixels per second, clamped so a short trace does not stretch to absurd bar widths. */
  px: number;
  /** Time-to-x projection, including the lane-label gutter. */
  x: (t: number) => number;
  maxEnd: number;
  svgWidth: number;
  svgHeight: number;
  nodeWidth: number;
  nodeHeight: number;
  blockHeight: number;
  laneHeight: number;
  rawTitleY: number;
  rawTop: number;
  rawBottom: number;
  warpTitleY: number;
  warpTop: number;
  axisY: number;
  ticks: number[];
  laneY: (top: number, lane: string) => number;
};

/**
 * Resolve every coordinate the timeline draws at.
 *
 * `maxEnd` floors at 1 so an empty or zero-length trace still produces a finite scale rather
 * than dividing by zero.
 */
export function layoutTimeline({
  lanes,
  bars,
  showWarp,
  hasTitle,
  width = DEFAULT_TIMELINE_WIDTH,
}: TimelineLayoutInput): TimelineLayout {
  const maxEnd = Math.max(1, ...bars.map((b) => Math.max(b.rawEnd, b.warpEnd)));
  const px = Math.max(8, Math.min(24, Math.floor((width - LEFT) / maxEnd)));
  const x = (t: number) => LEFT + t * px;

  const blockHeight = Math.max(0, lanes.length * (LANE_H + LANE_GAP) - LANE_GAP);
  const laneIndex = (lane: string) => Math.max(0, lanes.indexOf(lane));
  const laneY = (top: number, lane: string) => top + laneIndex(lane) * (LANE_H + LANE_GAP);

  const rawTitleY = 16;
  const rawTop = 26;
  const rawBottom = rawTop + blockHeight;
  const warpTitleY = rawBottom + 26;
  const warpTop = warpTitleY + 10;
  const axisY = (showWarp ? warpTop + blockHeight : rawBottom) + 14;

  const svgWidth = x(maxEnd) + 24;
  const svgHeight = axisY + 22;

  const tickStep = maxEnd > 45 ? 15 : maxEnd > 20 ? 10 : 5;
  const ticks: number[] = [];
  for (let t = 0; t <= maxEnd; t += tickStep) ticks.push(t);

  return {
    px,
    x,
    maxEnd,
    svgWidth,
    svgHeight,
    nodeWidth: svgWidth + 2 * PAD_X + BORDER,
    nodeHeight: svgHeight + 2 * PAD_Y + BORDER + (hasTitle ? TITLE_H : 0),
    blockHeight,
    laneHeight: LANE_H,
    rawTitleY,
    rawTop,
    rawBottom,
    warpTitleY,
    warpTop,
    axisY,
    ticks,
    laneY,
  };
}

/** The box a `timeline` node will occupy, for the authoring helper's `style`. */
export function timelineNodeSize(input: TimelineLayoutInput): { width: number; height: number } {
  const layout = layoutTimeline(input);
  return { width: layout.nodeWidth, height: layout.nodeHeight };
}
