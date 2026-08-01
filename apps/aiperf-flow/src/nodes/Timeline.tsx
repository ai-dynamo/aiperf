/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { NodeProps, Node } from "@xyflow/react";
import clsx from "clsx";
import { surfaceClassName, inkClassName, categoryClassName, categoryFillClassName, categoryStrokeClassName } from "../theme/tokens.js";
import { NodeAnchorHandles } from "./anchors.js";
import { layoutTimeline, laneRole, type TimelineBar } from "./timelineLayout.js";
import type { TimelineNodeData } from "./types.js";

export type TimelineNodeType = Node<TimelineNodeData, "timeline">;

/** Shortest label for a duration: whole seconds stay whole. */
function fmt(n: number): string {
  return Number.isInteger(n) ? `${n}` : n.toFixed(1);
}

/**
 * A time-scaled swimlane Gantt as a first-class node.
 *
 * Box-and-arrow nodes cannot say "these two requests overlapped" or "this clock was compressed
 * here" — those are claims about a shared axis, so this node draws one. Two stacked blocks (raw
 * clock, warped clock) share a single time axis; dashed bands over the raw block mark true idle
 * gaps, and a capped gap is the dead air the warp collapses.
 *
 * Presentational only: authored data in, SVG out, no controls. The interactive weka deck keeps its
 * own controlled variant.
 */
export function TimelineNode({ data }: NodeProps<TimelineNodeType>): React.JSX.Element {
  const { lanes, bars, gaps = [], title } = data;
  const showWarp = data.showWarp ?? true;
  const layout = layoutTimeline({
    lanes,
    bars,
    showWarp,
    hasTitle: title !== undefined,
    width: data.width,
  });
  const { x, laneY, laneHeight, blockHeight } = layout;

  const laneLabels = (top: number) =>
    lanes.map((lane) => (
      <text
        key={`label-${top}-${lane}`}
        x={8}
        y={laneY(top, lane) + laneHeight / 2 + 4}
        fontSize={10.5}
        fontWeight={600}
        className={categoryClassName(laneRole(lane, lanes))}
        fill="currentColor"
      >
        {lane}
      </text>
    ));

  const bar = (b: TimelineBar, top: number, start: number, end: number) => {
    // Floors at 10px so a sub-second request stays legible enough to carry its own id.
    const w = Math.max((end - start) * layout.px, 10);
    const y = laneY(top, b.lane);
    const role = laneRole(b.lane, lanes);
    return (
      <g key={`${b.id}-${top}`}>
        <rect
          x={x(start)}
          y={y}
          width={w}
          height={laneHeight}
          className={`${categoryFillClassName(role)} ${categoryStrokeClassName(role)}`}
          fillOpacity={0.18}
          strokeWidth={1.5}
        />
        <text
          x={x(start) + w / 2}
          y={y + laneHeight / 2 + 4}
          textAnchor="middle"
          fontSize={10.5}
          fontWeight={600}
          className={inkClassName("primary")}
          fill="currentColor"
        >
          {b.id}
        </text>
      </g>
    );
  };

  const blockLabel = (y: number, text: string) => (
    <text x={8} y={y} fontSize={11} fontWeight={700} className={inkClassName("secondary")} fill="currentColor">
      {text}
    </text>
  );

  return (
    <div
      className={clsx(
        "rounded-[13px] border border-white/10 px-4 py-3.5",
        "shadow-[0_12px_28px_rgba(0,0,0,0.28)]",
        surfaceClassName(data.surfaceRole ?? "elevated"),
        data.className,
      )}
    >
      {title !== undefined && (
        <div className={`mb-1.5 text-sm font-semibold tracking-tight ${inkClassName("primary")}`}>
          {title}
        </div>
      )}
      <svg
        width={layout.svgWidth}
        height={layout.svgHeight}
        role="img"
        aria-label={data.ariaLabel ?? (showWarp ? "raw versus warped timeline" : "request timeline")}
      >
        {gaps.map((gap, i) => (
          <g key={`gap-${i}`}>
            <rect
              x={x(gap.start)}
              y={layout.rawTop - 4}
              width={Math.max(x(gap.end) - x(gap.start), 2)}
              height={blockHeight + 8}
              fill="none"
              className={gap.capped ? categoryClassName("orange") : inkClassName("tertiary")}
              stroke="currentColor"
              strokeWidth={1}
              strokeDasharray="4 4"
            />
            <text
              x={(x(gap.start) + x(gap.end)) / 2}
              y={layout.rawTop - 8}
              textAnchor="middle"
              fontSize={9.5}
              className={gap.capped ? categoryClassName("orange") : inkClassName("tertiary")}
              fill="currentColor"
            >
              idle {fmt(gap.idle)}s{gap.capped ? " > cap" : ""}
            </text>
          </g>
        ))}

        {blockLabel(layout.rawTitleY, data.rawLabel ?? "raw clock")}
        {laneLabels(layout.rawTop)}
        {bars.map((b) => bar(b, layout.rawTop, b.rawStart, b.rawEnd))}

        {showWarp && (
          <>
            <line
              x1={8}
              y1={layout.rawBottom + 12}
              x2={layout.svgWidth - 12}
              y2={layout.rawBottom + 12}
              className={inkClassName("tertiary")}
              stroke="currentColor"
              strokeWidth={1}
            />
            {blockLabel(layout.warpTitleY, data.warpLabel ?? "warped clock")}
            {laneLabels(layout.warpTop)}
            {bars.map((b) => bar(b, layout.warpTop, b.warpStart, b.warpEnd))}
          </>
        )}

        <line
          x1={x(0)}
          y1={layout.axisY}
          x2={layout.svgWidth - 12}
          y2={layout.axisY}
          className={inkClassName("secondary")}
          stroke="currentColor"
          strokeWidth={1}
        />
        {layout.ticks.map((t) => (
          <g key={`tick-${t}`}>
            <line
              x1={x(t)}
              y1={layout.axisY - 3}
              x2={x(t)}
              y2={layout.axisY + 3}
              className={inkClassName("secondary")}
              stroke="currentColor"
              strokeWidth={1}
            />
            <text
              x={x(t)}
              y={layout.axisY + 15}
              textAnchor="middle"
              fontSize={10}
              className={inkClassName("tertiary")}
              fill="currentColor"
            >
              {t}s
            </text>
          </g>
        ))}
      </svg>
      <NodeAnchorHandles />
    </div>
  );
}
