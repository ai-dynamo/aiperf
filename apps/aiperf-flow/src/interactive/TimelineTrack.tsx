/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! `TimelineTrack` — the top-level swimlane-timeline SVG renderer. Given `lanes`, per-lane stage
//! `regions`, point `events`, grouping `seamFrames`, and a `requestPath` (ordered event ids), it
//! lays out a horizontal `TimeAxis`, the stacked `Lane` bands, the labeled `StageRegion` blocks
//! positioned along x, and the translucent `SeamFrame`s. It is an EVEN-SPACED stage flow: block x
//! comes from event ORDER only (never wall-clock), so blocks stay uniform and readable. There is no
//! connecting request line; the play head is a single `EventMarker` on the active event's stage.
//! `activeEventId` (from `useFlowPlayer`) highlights the current region + shows that marker; `scale`
//! ("real" | "virtual", from the Clock toggle) only reformats the tick labels. Domain-agnostic: it knows
//! nothing about AIPerf; a deck supplies the data. Tones are derived from lane order so the caller's
//! data model stays minimal. Respects `prefers-reduced-motion`.

import { useMemo } from "react";
import type { CategoryRole } from "../theme/tokens.js";
import { inkClassName } from "../theme/tokens.js";
import { TimeAxis, type TimeAxisTick } from "./TimeAxis.js";
import { Lane as LaneBand } from "./Lane.js";
import { StageRegion as StageRegionBlock } from "./StageRegion.js";
import { SeamFrame as SeamFrameBox } from "./SeamFrame.js";
import { EventMarker } from "./EventMarker.js";
import {
  buildOffsetForOrder,
  fractionForOrder,
  timelineBounds,
  type Lane,
  type RequestPath,
  type SeamFrame,
  type StageRegion,
  type TimelineEvent,
  type TimelineScale,
} from "./timeline.js";

export interface TimelineTrackProps {
  /** Swimlanes, top→bottom. */
  lanes: readonly Lane[];
  /** Stage regions (one block per stage), each in a lane and spanning an order range. */
  regions: readonly StageRegion[];
  /** Event points the request line threads through. */
  events: readonly TimelineEvent[];
  /** Translucent frames grouping bands of the track (the nested-composition view). */
  seamFrames: readonly SeamFrame[];
  /** Ordered event ids the single request line weaves through. */
  requestPath: RequestPath;
  /** Id of the event the play head is currently on (highlights its marker + owning region). */
  activeEventId?: string;
  /** X-axis mapping: real wall-ms offsets, or evenly-spaced virtual event ticks. */
  scale: TimelineScale;
  /** Called with a region id when its block is clicked (drills into that stage). */
  onRegionClick?: (regionId: string) => void;
  className?: string;
}

// Layout constants (viewBox pixel space; the SVG scales to its container via width="100%").
const VIEW_W = 1600;
const GUTTER = 128; // left gutter for lane labels
const RIGHT_PAD = 34;
const AXIS_Y = 44; // baseline of the time axis
const LANES_TOP = AXIS_Y + 22;
const LANE_H = 58; // tall enough for a two-line wrapped stage label
const LANE_GAP = 8;
const REGION_INSET_Y = 6;
const REGION_PAD_X = 8;
const MIN_REGION_W = 96;

const AXIS_LEFT = GUTTER;
const AXIS_RIGHT = VIEW_W - RIGHT_PAD;
const AXIS_W = AXIS_RIGHT - AXIS_LEFT;

// Minimum pixel gap enforced between consecutive event orders. Without this, wall-ms spacing bunches
// the setup events (all within ~64ms) into the left edge, stacking their dots and labels illegibly.
// The spread preserves order and keeps big real gaps (e.g. TTFT) proportionally wider.
const MIN_EVENT_GAP = 90;
// A comfortable default width for a single-order stage block, capped by its neighbor.
const DEFAULT_REGION_W = 150;
const REGION_GAP = 10;

// Tones are derived from lane/seam order so the (domain-agnostic) data model needs no color field.
const LANE_TONES: readonly CategoryRole[] = ["green", "purple", "yellow", "blue", "cyan", "orange", "red", "gray"];
const SEAM_TONES: readonly CategoryRole[] = ["gray", "purple", "yellow", "cyan", "orange", "blue"];

function laneRowTop(index: number): number {
  return LANES_TOP + index * (LANE_H + LANE_GAP);
}

function clampX(x: number): number {
  return Math.max(AXIS_LEFT, Math.min(AXIS_RIGHT, x));
}

/**
 * The swimlane-timeline renderer. Lays out the axis, lanes, regions, seam frames, the request line,
 * and the event markers into one responsive SVG.
 */
export function TimelineTrack({
  lanes,
  regions,
  events,
  seamFrames,
  // `requestPath` is still part of the props (callers pass it) but the overview no longer draws a
  // connecting line, so it is intentionally not consumed here.
  activeEventId,
  scale,
  onRegionClick,
  className,
}: TimelineTrackProps): React.JSX.Element {
  const bounds = useMemo(() => timelineBounds(events), [events]);
  const offsetForOrder = useMemo(() => buildOffsetForOrder(events), [events]);
  const laneIndex = useMemo(() => {
    const map = new Map<string, number>();
    lanes.forEach((lane, i) => map.set(lane.id, i));
    return map;
  }, [lanes]);
  const eventById = useMemo(() => {
    const map = new Map<string, TimelineEvent>();
    for (const event of events) {
      map.set(event.id, event);
    }
    return map;
  }, [events]);

  // Spread the distinct event orders across the axis with a guaranteed minimum gap, then compress
  // uniformly back inside the axis if the enforced gaps overflowed. This de-collides clustered
  // (wall-ms) events while preserving order and the *relative* size of the big real latency gaps.
  const orderX = useMemo(() => {
    const distinct = Array.from(new Set(events.map((e) => e.atOrder))).sort((a, b) => a - b);
    // Even-spaced STAGE FLOW: positions come from event ORDER only, never wall-clock offsets, so the
    // blocks are uniform and readable (the Clock seam still recolors the line + reformats the tick
    // labels, but no longer crushes the setup stages into the left edge). `scale` intentionally
    // unused for positioning; `"virtual"` == evenly by order.
    const raw = distinct.map(
      (order) => AXIS_LEFT + fractionForOrder(order, "virtual", bounds, offsetForOrder) * AXIS_W,
    );
    const spread: number[] = [];
    for (let i = 0; i < raw.length; i++) {
      spread[i] = i === 0 ? raw[i]! : Math.max(raw[i]!, spread[i - 1]! + MIN_EVENT_GAP);
    }
    const first = spread[0] ?? AXIS_LEFT;
    const last = spread[spread.length - 1] ?? AXIS_RIGHT;
    if (last > AXIS_RIGHT && last > first) {
      const factor = AXIS_W / (last - first);
      for (let i = 0; i < spread.length; i++) {
        spread[i] = AXIS_LEFT + (spread[i]! - first) * factor;
      }
    }
    const map = new Map<number, number>();
    distinct.forEach((order, i) => map.set(order, spread[i]!));
    return { distinct, xs: spread, map };
  }, [events, scale, bounds, offsetForOrder]);

  // x for an order coordinate: exact for real event orders, linearly interpolated between the two
  // nearest event orders for in-between coordinates (seam-frame span bounds).
  const xForOrder = (order: number): number => {
    const { distinct, xs, map } = orderX;
    const exact = map.get(order);
    if (exact !== undefined) {
      return exact;
    }
    if (distinct.length === 0) {
      return AXIS_LEFT;
    }
    if (order <= distinct[0]!) {
      return xs[0]!;
    }
    if (order >= distinct[distinct.length - 1]!) {
      return xs[xs.length - 1]!;
    }
    for (let i = 0; i < distinct.length - 1; i++) {
      const lo = distinct[i]!;
      const hi = distinct[i + 1]!;
      if (order >= lo && order <= hi) {
        const t = hi === lo ? 0 : (order - lo) / (hi - lo);
        return xs[i]! + t * (xs[i + 1]! - xs[i]!);
      }
    }
    return xs[xs.length - 1]!;
  };
  const xForEvent = (event: TimelineEvent): number => xForOrder(event.atOrder);

  // Even-spaced stage flow: every stage gets one uniform column (AXIS_W / N), assigned by narrative
  // order across ALL stages, so blocks are generous and equal-width (labels fit on one line) and
  // staircase across the lanes by their order — no wall-clock crushing, no per-lane tiling.
  const regionRects = (() => {
    const sorted = [...regions].sort((a, b) => a.startOrder - b.startOrder);
    const n = sorted.length;
    const colW = n > 0 ? AXIS_W / n : AXIS_W;
    const rects = new Map<string, { x: number; width: number }>();
    sorted.forEach((region, i) => {
      rects.set(region.id, {
        x: AXIS_LEFT + i * colW + REGION_GAP / 2,
        width: Math.max(colW - REGION_GAP, MIN_REGION_W),
      });
    });
    return rects;
  })();
  const regionCenterX = (regionId: string): number => {
    const r = regionRects.get(regionId);
    return r ? r.x + r.width / 2 : AXIS_LEFT;
  };
  const laneCenterY = (laneId: string): number => {
    const idx = laneIndex.get(laneId) ?? 0;
    return laneRowTop(idx) + LANE_H / 2;
  };
  const laneTone = (laneId: string): CategoryRole => {
    const idx = laneIndex.get(laneId) ?? 0;
    return LANE_TONES[idx % LANE_TONES.length]!;
  };

  const height = LANES_TOP + lanes.length * (LANE_H + LANE_GAP) + 10;

  // The active event + the region that owns it (regions have disjoint order ranges, so the atOrder
  // uniquely identifies the owning stage regardless of lane).
  const activeEvent = activeEventId !== undefined ? eventById.get(activeEventId) : undefined;
  const activeRegionId =
    activeEvent !== undefined
      ? regions.find((r) => activeEvent.atOrder >= r.startOrder && activeEvent.atOrder <= r.endOrder)?.id
      : undefined;

  // The axis caption names the active clock as CONTEXT — it does not claim the blocks are placed by
  // wall time (they are evenly spaced by request order). No numeric ticks: labeling even positions
  // with wall-ms values would be misleading, so the axis is a plain "request order" rule.
  const unitLabel = `Request order · ${scale === "real" ? "RealClock" : "SimClock"}`;
  const ticks: TimeAxisTick[] = [];

  return (
    <div className={className}>
      <svg
        viewBox={`0 0 ${VIEW_W} ${height}`}
        width="100%"
        role="group"
        aria-label="Request timeline through subsystem swimlanes"
        style={{ display: "block" }}
      >
        {/* Lane bands (background). */}
        {lanes.map((lane, i) => (
          <LaneBand
            key={lane.id}
            x={AXIS_LEFT - 8}
            y={laneRowTop(i)}
            width={AXIS_RIGHT - (AXIS_LEFT - 8)}
            height={LANE_H}
            label={lane.label}
            labelX={12}
            tone={laneTone(lane.id)}
          />
        ))}

        {/* Seam frames (translucent, above bands, behind regions). */}
        {seamFrames.map((frame, i) => {
          const spanLanes = frame.spanLaneIds ?? lanes.map((l) => l.id);
          const idxs = spanLanes
            .map((id) => laneIndex.get(id))
            .filter((v): v is number => v !== undefined);
          const minIdx = idxs.length > 0 ? Math.min(...idxs) : 0;
          const maxIdx = idxs.length > 0 ? Math.max(...idxs) : lanes.length - 1;
          const yTop = laneRowTop(minIdx) - 4;
          const yBottom = laneRowTop(maxIdx) + LANE_H + 4;
          // Align the frame to the EVEN-COLUMN blocks it groups (not the order axis): union the
          // rects of the regions inside its lane set + order span, so the frame hugs those columns.
          const [startOrder, endOrder] = frame.spanOrder ?? [bounds.minOrder, bounds.maxOrder];
          const laneSet = frame.spanLaneIds ? new Set(frame.spanLaneIds) : undefined;
          const covered = regions
            .filter(
              (r) =>
                (laneSet === undefined || laneSet.has(r.laneId)) &&
                r.endOrder >= startOrder &&
                r.startOrder <= endOrder,
            )
            .map((r) => regionRects.get(r.id))
            .filter((r): r is { x: number; width: number } => r !== undefined);
          const fx1 =
            covered.length > 0
              ? Math.min(...covered.map((r) => r.x)) - REGION_PAD_X
              : clampX(xForOrder(startOrder) - REGION_PAD_X);
          const fx2 =
            covered.length > 0
              ? Math.max(...covered.map((r) => r.x + r.width)) + REGION_PAD_X
              : clampX(xForOrder(endOrder) + REGION_PAD_X);
          return (
            <SeamFrameBox
              key={frame.id}
              x={fx1}
              y={yTop}
              width={Math.max(fx2 - fx1, MIN_REGION_W)}
              height={yBottom - yTop}
              label={frame.label}
              tone={SEAM_TONES[i % SEAM_TONES.length]!}
            />
          );
        })}

        {/* Stage region blocks (tiled per lane so they never overlap). */}
        {regions.map((region) => {
          const idx = laneIndex.get(region.laneId) ?? 0;
          const rect = regionRects.get(region.id) ?? { x: AXIS_LEFT, width: DEFAULT_REGION_W };
          return (
            <StageRegionBlock
              key={region.id}
              x={rect.x}
              y={laneRowTop(idx) + REGION_INSET_Y}
              width={rect.width}
              height={LANE_H - 2 * REGION_INSET_Y}
              label={region.label}
              tone={laneTone(region.laneId)}
              active={region.id === activeRegionId}
              onClick={onRegionClick ? () => onRegionClick(region.id) : undefined}
            />
          );
        })}

        {/* Time axis (above the lanes). */}
        <TimeAxis x1={AXIS_LEFT} x2={AXIS_RIGHT} y={AXIS_Y} ticks={ticks} unitLabel={unitLabel} />

        {/* No connecting request line: the play head is shown only as a single marker on the active
            event's stage (paired with the region highlight + the caption above the track). */}
        {activeEvent !== undefined && (
          <EventMarker
            x={activeRegionId !== undefined ? regionCenterX(activeRegionId) : xForEvent(activeEvent)}
            y={laneCenterY(activeEvent.laneId)}
            tone={laneTone(activeEvent.laneId)}
            active
            label={activeEvent.label}
          />
        )}
      </svg>
      <p className={`mt-1 text-[11px] ${inkClassName("tertiary")}`}>
        {regions.length} stages in request order across {lanes.length} subsystem lanes — click a stage
        block to drill in.
      </p>
    </div>
  );
}
