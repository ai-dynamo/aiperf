/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Domain-agnostic data shapes + pure layout math for the `src/interactive/` swimlane-timeline
//! renderer. Nothing here is AIPerf-specific: `TimelineTrack` and its subcomponents are driven by
//! these generics (lanes / regions / events / seam frames / a request path), so any future deck can
//! render "one continuous track riding a time axis through subsystem swimlanes". The helpers here
//! own the x-mapping (time → fraction of the axis) for both the RealClock ("real", by wall-ms) and
//! SimClock ("virtual", evenly by event order) scales, kept pure so they can be unit-tested without
//! a DOM.

/** Stable id of one swimlane (a subsystem row). Generic string — decks pick their own vocabulary. */
export type LaneId = string;

/** One horizontal swimlane: a subsystem the request line threads through. */
export interface Lane {
  /** Stable lane id (referenced by `StageRegion.laneId` / `TimelineEvent.laneId`). */
  id: LaneId;
  /** Human-readable lane label, drawn in the left gutter. */
  label: string;
}

/**
 * A labeled block positioned along the time axis inside a lane (one per stage). `startOrder`/
 * `endOrder` are event-order coordinates (the same axis the events use); the renderer maps them to
 * x via the active scale. `id` is the stable stage id — clicking the region drills into that stage.
 */
export interface StageRegion {
  /** Stable region id (typically the stage id, so a click can drill into it). */
  id: string;
  /** Which lane this region sits in. */
  laneId: LaneId;
  /** Region label drawn on the block. */
  label: string;
  /** Left edge in event-order coordinates. */
  startOrder: number;
  /** Right edge in event-order coordinates. */
  endOrder: number;
}

/**
 * One point the request line passes through. `atOrder` is its position on the virtual (evenly
 * spaced) axis; `realOffsetMs` is its wall-clock offset for the "real" scale (falls back to
 * `atOrder` when absent). `laneId` is which swimlane the marker sits in — this is how the single
 * line weaves top↔bottom across lanes.
 */
export interface TimelineEvent {
  /** Stable event id (referenced from a `RequestPath`). */
  id: string;
  /** Marker label. */
  label: string;
  /** Which lane the marker sits in. */
  laneId: LaneId;
  /** Position on the virtual axis (evenly spaced by order). */
  atOrder: number;
  /** Wall-clock offset (ms) for the "real" scale; omitted → treated as `atOrder`. */
  realOffsetMs?: number;
}

/**
 * A translucent frame grouping a band of the timeline — the nested-composition (seam) view. Either
 * dimension is optional: `spanLaneIds` restricts the vertical band to those lanes (default: all
 * lanes); `spanOrder` restricts the horizontal band to `[startOrder, endOrder]` (default: the whole
 * axis). A frame with neither spans the entire track (e.g. the Clock seam = the whole time axis).
 */
export interface SeamFrame {
  /** Stable frame id. */
  id: string;
  /** Frame label drawn at its top-left. */
  label: string;
  /** Lanes the frame vertically covers; omitted → all lanes. */
  spanLaneIds?: readonly LaneId[];
  /** Horizontal `[startOrder, endOrder]` the frame covers; omitted → the whole axis. */
  spanOrder?: readonly [number, number];
}

/** The request's journey: an ordered list of `TimelineEvent` ids the single line threads through. */
export type RequestPath = readonly string[];

/** Which time scale the x-axis uses: real wall-ms offsets, or evenly-spaced virtual event ticks. */
export type TimelineScale = "real" | "virtual";

/** Min/max order + wall-ms offset across a set of events — the axis domain. */
export interface TimelineBounds {
  minOrder: number;
  maxOrder: number;
  minOffsetMs: number;
  maxOffsetMs: number;
}

function clamp01(value: number): number {
  if (value < 0) {
    return 0;
  }
  if (value > 1) {
    return 1;
  }
  return value;
}

/** The wall-ms offset of an event, falling back to its order when no `realOffsetMs` is given. */
export function eventOffsetMs(event: TimelineEvent): number {
  return event.realOffsetMs ?? event.atOrder;
}

/** Compute the axis domain (order + offset extents) over a set of events. Empty → all zeros. */
export function timelineBounds(events: readonly TimelineEvent[]): TimelineBounds {
  if (events.length === 0) {
    return { minOrder: 0, maxOrder: 0, minOffsetMs: 0, maxOffsetMs: 0 };
  }
  let minOrder = Infinity;
  let maxOrder = -Infinity;
  let minOffsetMs = Infinity;
  let maxOffsetMs = -Infinity;
  for (const event of events) {
    minOrder = Math.min(minOrder, event.atOrder);
    maxOrder = Math.max(maxOrder, event.atOrder);
    const offset = eventOffsetMs(event);
    minOffsetMs = Math.min(minOffsetMs, offset);
    maxOffsetMs = Math.max(maxOffsetMs, offset);
  }
  return { minOrder, maxOrder, minOffsetMs, maxOffsetMs };
}

/**
 * Build a monotone order→offset interpolator from the events (sorted by order, linearly
 * interpolated between the known points, clamped at the ends). Used to place region/seam bounds —
 * which are given in order coordinates — on the "real" (wall-ms) axis.
 */
export function buildOffsetForOrder(
  events: readonly TimelineEvent[],
): (order: number) => number {
  const pairs = events
    .map((event) => ({ order: event.atOrder, offset: eventOffsetMs(event) }))
    .sort((a, b) => a.order - b.order);
  return (order: number): number => {
    if (pairs.length === 0) {
      return order;
    }
    const first = pairs[0]!;
    const last = pairs[pairs.length - 1]!;
    if (order <= first.order) {
      return first.offset;
    }
    if (order >= last.order) {
      return last.offset;
    }
    for (let i = 0; i < pairs.length - 1; i++) {
      const lo = pairs[i]!;
      const hi = pairs[i + 1]!;
      if (order >= lo.order && order <= hi.order) {
        if (hi.order === lo.order) {
          return lo.offset;
        }
        const t = (order - lo.order) / (hi.order - lo.order);
        return lo.offset + t * (hi.offset - lo.offset);
      }
    }
    return last.offset;
  };
}

/**
 * Fraction (0..1) of the axis width at which a given event sits, under the active scale. "virtual"
 * spaces by `atOrder`; "real" spaces by wall-ms offset — so a large latency gap (e.g. TTFT) opens a
 * visibly wider gap on the real axis than on the evenly-spaced virtual one.
 */
export function fractionForEvent(
  event: TimelineEvent,
  scale: TimelineScale,
  bounds: TimelineBounds,
): number {
  if (scale === "virtual") {
    const span = bounds.maxOrder - bounds.minOrder;
    return span === 0 ? 0 : clamp01((event.atOrder - bounds.minOrder) / span);
  }
  const span = bounds.maxOffsetMs - bounds.minOffsetMs;
  return span === 0 ? 0 : clamp01((eventOffsetMs(event) - bounds.minOffsetMs) / span);
}

/**
 * Fraction (0..1) of the axis width for a bare order coordinate (a region/seam bound). "virtual"
 * uses the order directly; "real" runs it through `offsetForOrder` first.
 */
export function fractionForOrder(
  order: number,
  scale: TimelineScale,
  bounds: TimelineBounds,
  offsetForOrder: (order: number) => number,
): number {
  if (scale === "virtual") {
    const span = bounds.maxOrder - bounds.minOrder;
    return span === 0 ? 0 : clamp01((order - bounds.minOrder) / span);
  }
  const span = bounds.maxOffsetMs - bounds.minOffsetMs;
  return span === 0 ? 0 : clamp01((offsetForOrder(order) - bounds.minOffsetMs) / span);
}
