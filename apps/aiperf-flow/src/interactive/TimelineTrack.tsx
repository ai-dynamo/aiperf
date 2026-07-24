/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! `TimelineTrack` — the top-level swimlane-timeline SVG renderer. Given `lanes`, per-lane stage
//! `regions`, point `events`, grouping `seamFrames`, and a `requestPath` (ordered event ids), it
//! lays out a horizontal `TimeAxis`, the stacked `Lane` bands, the labeled `StageRegion` blocks
//! positioned along x, the translucent `SeamFrame`s, and ONE weaving `RequestLine` polyline with an
//! `EventMarker` per event. `activeEventId` (from `useFlowPlayer`) highlights the current
//! region/marker; `scale` ("real" | "virtual", from the Clock toggle) controls the x-mapping —
//! "real" spaces by wall-ms offsets, "virtual" evenly by event order. Domain-agnostic: it knows
//! nothing about AIPerf; a deck supplies the data. Tones are derived from lane order so the caller's
//! data model stays minimal. Respects `prefers-reduced-motion`.

import { useMemo } from "react";
import { useReducedMotion } from "motion/react";
import type { CategoryRole } from "../theme/tokens.js";
import { inkClassName } from "../theme/tokens.js";
import { TimeAxis, type TimeAxisTick } from "./TimeAxis.js";
import { Lane as LaneBand } from "./Lane.js";
import { StageRegion as StageRegionBlock } from "./StageRegion.js";
import { SeamFrame as SeamFrameBox } from "./SeamFrame.js";
import { RequestLine } from "./RequestLine.js";
import { EventMarker } from "./EventMarker.js";
import {
  buildOffsetForOrder,
  eventOffsetMs,
  fractionForEvent,
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
const VIEW_W = 960;
const GUTTER = 122; // left gutter for lane labels
const RIGHT_PAD = 30;
const AXIS_Y = 44; // baseline of the time axis
const LANES_TOP = AXIS_Y + 20;
const LANE_H = 46;
const LANE_GAP = 6;
const REGION_INSET_Y = 6;
const REGION_PAD_X = 7;
const MIN_REGION_W = 74;

const AXIS_LEFT = GUTTER;
const AXIS_RIGHT = VIEW_W - RIGHT_PAD;

// Tones are derived from lane/seam order so the (domain-agnostic) data model needs no color field.
const LANE_TONES: readonly CategoryRole[] = ["green", "purple", "yellow", "blue", "cyan", "orange", "red", "gray"];
const SEAM_TONES: readonly CategoryRole[] = ["gray", "purple", "yellow", "cyan", "orange", "blue"];

function laneRowTop(index: number): number {
  return LANES_TOP + index * (LANE_H + LANE_GAP);
}

function clampX(x: number): number {
  return Math.max(AXIS_LEFT, Math.min(AXIS_RIGHT, x));
}

/** Pick ~5 evenly-spaced ticks from the request path so the axis stays legible. */
function pickTickIndices(count: number): number[] {
  if (count <= 5) {
    return Array.from({ length: count }, (_, i) => i);
  }
  return [0, Math.round(count / 4), Math.round(count / 2), Math.round((3 * count) / 4), count - 1];
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
  requestPath,
  activeEventId,
  scale,
  onRegionClick,
  className,
}: TimelineTrackProps): React.JSX.Element {
  const prefersReduced = useReducedMotion() ?? false;

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

  const xForFraction = (f: number): number => AXIS_LEFT + f * (AXIS_RIGHT - AXIS_LEFT);
  const xForEvent = (event: TimelineEvent): number => xForFraction(fractionForEvent(event, scale, bounds));
  const xForOrder = (order: number): number =>
    xForFraction(fractionForOrder(order, scale, bounds, offsetForOrder));
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

  // The request line points, in path order, weaving lane→lane.
  const linePoints = requestPath
    .map((id) => eventById.get(id))
    .filter((event): event is TimelineEvent => event !== undefined)
    .map((event) => ({ x: xForEvent(event), y: laneCenterY(event.laneId) }));

  // The line color reflects the active scale (a legible cue for the Clock seam).
  const lineTone: CategoryRole = scale === "virtual" ? "purple" : "green";

  const unitLabel = scale === "real" ? "RealClock · wall-ms" : "SimClock · virtual ticks";

  // Axis ticks: a downsampled set of request-path events, labeled by wall-ms (real) or tick (virtual).
  const pathEvents = requestPath
    .map((id) => eventById.get(id))
    .filter((event): event is TimelineEvent => event !== undefined);
  const tickIdx = new Set(pickTickIndices(pathEvents.length));
  const ticks: TimeAxisTick[] = pathEvents
    .map((event, i) => ({ event, i }))
    .filter(({ i }) => tickIdx.has(i))
    .map(({ event }) => ({
      x: xForEvent(event),
      label: scale === "real" ? `${Math.round(eventOffsetMs(event))}` : `t${event.atOrder}`,
    }));

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
          const [startOrder, endOrder] = frame.spanOrder ?? [bounds.minOrder, bounds.maxOrder];
          const fx1 = clampX(xForOrder(startOrder) - REGION_PAD_X);
          const fx2 = clampX(xForOrder(endOrder) + REGION_PAD_X);
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

        {/* Stage region blocks. */}
        {regions.map((region) => {
          const idx = laneIndex.get(region.laneId) ?? 0;
          const xStart = xForOrder(region.startOrder);
          const xEnd = xForOrder(region.endOrder);
          const mid = (xStart + xEnd) / 2;
          let rx = xStart - REGION_PAD_X;
          let rw = xEnd - xStart + 2 * REGION_PAD_X;
          if (rw < MIN_REGION_W) {
            rx = mid - MIN_REGION_W / 2;
            rw = MIN_REGION_W;
          }
          rx = Math.max(AXIS_LEFT - 6, Math.min(rx, AXIS_RIGHT - rw));
          return (
            <StageRegionBlock
              key={region.id}
              x={rx}
              y={laneRowTop(idx) + REGION_INSET_Y}
              width={rw}
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

        {/* The hero request line + event markers (top layer). */}
        {linePoints.length >= 2 && (
          <RequestLine points={linePoints} tone={lineTone} reducedMotion={prefersReduced} />
        )}
        {events.map((event) => (
          <EventMarker
            key={event.id}
            x={xForEvent(event)}
            y={laneCenterY(event.laneId)}
            tone={laneTone(event.laneId)}
            active={event.id === activeEventId}
            label={event.label}
          />
        ))}
      </svg>
      <p className={`mt-1 text-[11px] ${inkClassName("tertiary")}`}>
        One request riding the time axis through {lanes.length} subsystem lanes — click a stage block
        to drill in.
      </p>
    </div>
  );
}
