/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared diagram vocabulary for narrated decks.
//!
//! The engine's whole subject is fan-out and fan-in — successors spawning, AND-joins on
//! channel counts, traces spreading across workers. These helpers place nodes on an
//! explicit grid so a slide can draw that shape, rather than flattening every idea into
//! a left-to-right chain.

import type { Edge, Node } from "@xyflow/react";
import { timelineNodeSize, type TimelineBar, type TimelineGap } from "../../nodes/timelineLayout.js";
import { intervalsNodeSize, type IntervalRow } from "../../nodes/intervalsLayout.js";
import { blocksNodeSize, type BlockStrip } from "../../nodes/blocksLayout.js";
import { sweepNodeSize } from "../../nodes/sweepLayout.js";
import { slicesNodeSize, type SliceRequest } from "../../nodes/slicesLayout.js";
import { raggedNodeSize } from "../../nodes/raggedLayout.js";
import type { SweepCurveId, SweepRequest } from "../../nodes/sweepMath.js";

/** Grid step. One column clears a card's max width; one row clears its height. */
export const COL = 300;
export const ROW = 118;

/**
 * Semantic colour, shared by nodes and the edges between them so a hop and the boxes it
 * joins agree. Mirrors the categories in `theme/tokens.ts`.
 */
export type Tone = "control" | "data" | "time" | "failure" | "done" | "muted";

export const TONE_STROKE: Record<Tone, string> = {
  control: "var(--color-accent-primary)",
  data: "var(--color-category-purple)",
  time: "var(--color-category-yellow)",
  failure: "var(--color-category-red)",
  done: "var(--color-category-green)",
  muted: "var(--color-category-gray)",
};

// Tailwind only sees class strings it can read literally, so each tone's arbitrary
// property is spelled out rather than interpolated.
const TONE_TINT: Record<Tone, string> = {
  control: "[--accent:var(--color-accent-primary)]",
  data: "[--accent:var(--color-category-purple)]",
  time: "[--accent:var(--color-category-yellow)]",
  failure: "[--accent:var(--color-category-red)]",
  done: "[--accent:var(--color-category-green)]",
  muted: "[--accent:var(--color-category-gray)]",
};

interface Placed {
  /** Grid column; fractional values are allowed for centring a fan's apex. */
  col: number;
  /** Grid row. */
  row: number;
}

/** A titled card, the workhorse node. */
export function card(
  id: string,
  title: string,
  subtitle: string | undefined,
  tone: Tone,
  { col, row }: Placed,
): Node {
  return {
    id,
    type: "card",
    position: { x: col * COL, y: row * ROW },
    data: { title, subtitle, className: TONE_TINT[tone] },
  };
}

/** Small junction label — a gate, a combiner, a count. Reads as an operator, not a stage. */
export function chip(id: string, label: string, { col, row }: Placed): Node {
  return {
    id,
    type: "chip",
    // Nudged toward the row's vertical centre, since a chip is far shorter than a card.
    position: { x: col * COL, y: row * ROW + 26 },
    data: { label },
  };
}

/** Aside panel carrying a claim the diagram itself cannot state. */
export function note(id: string, title: string, detail: string, { col, row }: Placed): Node {
  return {
    id,
    type: "panel",
    position: { x: col * COL, y: row * ROW },
    data: { title, detail },
  };
}

/** Band caption above a group of nodes. */
export function band(id: string, title: string, { col, row }: Placed): Node {
  return {
    id,
    type: "header",
    position: { x: col * COL, y: row * ROW - 34 },
    data: { title },
  };
}

/**
 * Time-scaled swimlane Gantt — request overlap and clock warp on a shared axis.
 *
 * Far wider and taller than a card, so it is sized explicitly rather than left for React Flow to
 * measure: `Slide` re-fits the view on every reveal tick, and a node whose box arrives late makes
 * the whole diagram reframe mid-cascade.
 */
export function timeline(
  id: string,
  opts: {
    title?: string;
    lanes: string[];
    bars: TimelineBar[];
    gaps?: TimelineGap[];
    showWarp?: boolean;
    rawLabel?: string;
    warpLabel?: string;
    width?: number;
  },
  { col, row }: Placed,
): Node {
  const showWarp = opts.showWarp ?? true;
  const size = timelineNodeSize({
    lanes: opts.lanes,
    bars: opts.bars,
    showWarp,
    hasTitle: opts.title !== undefined,
    width: opts.width,
  });
  return {
    id,
    type: "timeline",
    position: { x: col * COL, y: row * ROW },
    style: size,
    data: { ...opts, showWarp },
  };
}

/** Columns a `timeline` of this size spans, for placing neighbours clear of it. */
export function timelineCols(opts: Parameters<typeof timeline>[1]): number {
  const { width } = timelineNodeSize({
    lanes: opts.lanes,
    bars: opts.bars,
    showWarp: opts.showWarp ?? true,
    hasTitle: opts.title !== undefined,
    width: opts.width,
  });
  return Math.ceil(width / COL);
}

/**
 * Intervals on one clock with a rank badge on each end — what interval-order derivation reads.
 *
 * Sized explicitly, for the same reason as `timeline`.
 */
export function intervals(
  id: string,
  opts: { title?: string; rows: IntervalRow[]; width?: number; ariaLabel?: string },
  { col, row }: Placed,
): Node {
  return {
    id,
    type: "intervals",
    position: { x: col * COL, y: row * ROW },
    style: intervalsNodeSize({
      rows: opts.rows,
      hasTitle: opts.title !== undefined,
      width: opts.width,
    }),
    data: { ...opts },
  };
}

/**
 * Stacked per-block tag strips, for comparing two paths over a shared prefix.
 *
 * `detail` is allotted a fixed two-line block in the node's height, so keep it short.
 */
export function blocks(
  id: string,
  opts: { title?: string; strips: BlockStrip[]; highlight?: number; detail?: string },
  { col, row }: Placed,
): Node {
  return {
    id,
    type: "blocks",
    position: { x: col * COL, y: row * ROW },
    style: blocksNodeSize({
      strips: opts.strips,
      hasTitle: opts.title !== undefined,
      hasDetail: opts.detail !== undefined,
    }),
    data: { ...opts },
  };
}

/**
 * Intervals over the step function they generate — the sweep-line identity.
 *
 * Sized explicitly, for the same reason as `timeline`.
 */
export function sweep(
  id: string,
  opts: {
    title?: string;
    requests: SweepRequest[];
    curve?: SweepCurveId;
    tMax?: number;
    axisLabel?: string;
    valueLabel?: string;
    width?: number;
  },
  { col, row }: Placed,
): Node {
  const curve = opts.curve ?? "concurrency";
  return {
    id,
    type: "sweep",
    position: { x: col * COL, y: row * ROW },
    style: sweepNodeSize({
      requests: opts.requests,
      curve,
      hasTitle: opts.title !== undefined,
      tMax: opts.tMax,
      width: opts.width,
    }),
    data: { ...opts, curve },
  };
}

/** A uniform bucket grid over a Gantt, showing binning and the incomplete trailing slice. */
export function slices(
  id: string,
  opts: {
    title?: string;
    requests: SliceRequest[];
    duration: number;
    axisLabel?: string;
    width?: number;
  },
  { col, row }: Placed,
): Node {
  return {
    id,
    type: "slices",
    position: { x: col * COL, y: row * ROW },
    style: slicesNodeSize({
      requests: opts.requests,
      duration: opts.duration,
      hasTitle: opts.title !== undefined,
      width: opts.width,
    }),
    data: { ...opts },
  };
}

/** Ragged per-record lists over the flat arrays they pack into. */
export function ragged(
  id: string,
  opts: {
    title?: string;
    lists: number[][];
    highlight?: number;
    showFlat?: boolean;
    raggedLabel?: string;
    flatLabel?: string;
  },
  { col, row }: Placed,
): Node {
  const showFlat = opts.showFlat ?? true;
  return {
    id,
    type: "ragged",
    position: { x: col * COL, y: row * ROW },
    style: raggedNodeSize({
      lists: opts.lists,
      hasTitle: opts.title !== undefined,
      showFlat,
    }),
    data: { ...opts, showFlat },
  };
}

/** Directed connector. Ids derive from the pair; no slide connects a pair twice. */
export function link(
  source: string,
  target: string,
  tone: Tone = "control",
  speed: "slow" | "normal" | "fast" = "normal",
): Edge {
  return {
    id: `${source}->${target}`,
    source,
    target,
    type: "flow",
    data: { color: TONE_STROKE[tone], speed },
  };
}

/** One source branching to many targets. */
export function fanOut(
  source: string,
  targets: readonly string[],
  tone: Tone = "control",
  speed: "slow" | "normal" | "fast" = "normal",
): Edge[] {
  return targets.map((target) => link(source, target, tone, speed));
}

/** Many sources converging on one target — an AND-join in engine terms. */
export function fanIn(
  sources: readonly string[],
  target: string,
  tone: Tone = "control",
  speed: "slow" | "normal" | "fast" = "normal",
): Edge[] {
  return sources.map((source) => link(source, target, tone, speed));
}

/** Vertically centred row for a fan of `count` items starting at `firstRow`. */
export function apexRow(firstRow: number, count: number): number {
  return firstRow + (count - 1) / 2;
}
