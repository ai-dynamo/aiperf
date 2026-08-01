/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { NodeProps, Node } from "@xyflow/react";
import clsx from "clsx";
import {
  surfaceClassName,
  inkClassName,
  strokeClassName,
  categoryClassName,
  categoryFillClassName,
  type CategoryRole,
} from "../theme/tokens.js";
import { NodeAnchorHandles } from "./anchors.js";
import { binOf, layoutSlices } from "./slicesLayout.js";
import type { SlicesNodeData } from "./types.js";

export type SlicesNodeType = Node<SlicesNodeData, "slices">;

/** Bin hues, cycled, so a bar's colour names the bucket it landed in. */
const BIN_ROLES: readonly CategoryRole[] = ["blue", "green", "purple", "orange", "cyan", "yellow"];

/**
 * A uniform grid laid over a Gantt — which bucket each interval is binned into, and where the
 * grid overruns real activity.
 *
 * Two things are easy to get wrong about time-slicing and both are visible here. Binning is by
 * *start*, not by overlap, so an interval spanning three buckets still counts once, in the bucket
 * it began in (the dot marks that key). And the trailing bucket usually runs past the last
 * activity: it is drawn clipped, with the overrun tinted, because dividing a rate by the
 * grid-defined width instead of the clipped width silently dilutes it with idle padding.
 */
export function SlicesNode({ data }: NodeProps<SlicesNodeType>): React.JSX.Element {
  const { requests, title } = data;
  const layout = layoutSlices({
    requests,
    duration: data.duration,
    hasTitle: title !== undefined,
    width: data.width,
  });
  const { x, top, rowHeight, ganttHeight, slices } = layout;

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
        aria-label={data.ariaLabel ?? "time slice chart"}
      >
        {slices.map((slice) => (
          <g key={`slice-${slice.index}`}>
            {slice.index % 2 === 0 && (
              <rect
                x={x(slice.start)}
                y={top}
                width={x(slice.clippedEnd) - x(slice.start)}
                height={ganttHeight}
                className={inkClassName("quaternary")}
                fill="currentColor"
                opacity={0.4}
              />
            )}
            {!slice.isComplete && (
              <rect
                x={x(slice.clippedEnd)}
                y={top}
                width={x(slice.end) - x(slice.clippedEnd)}
                height={ganttHeight}
                className={categoryFillClassName("orange")}
                opacity={0.15}
              />
            )}
            <line
              x1={x(slice.start)}
              y1={top}
              x2={x(slice.start)}
              y2={layout.axisY}
              className={strokeClassName("secondary")}
              stroke="currentColor"
              strokeDasharray="2 2"
            />
            <text
              x={(x(slice.start) + x(slice.clippedEnd)) / 2}
              y={layout.axisY + 16}
              textAnchor="middle"
              fontSize={10}
              className={slice.isComplete ? inkClassName("tertiary") : categoryClassName("orange")}
              fill="currentColor"
            >
              {`slice ${slice.index}${slice.isComplete ? "" : " *"}`}
            </text>
          </g>
        ))}

        {requests.map((r, i) => {
          const yTop = top + i * rowHeight + 4;
          const barH = rowHeight - 9;
          const bin = binOf(r.start, layout.spanStart, data.duration, slices.length);
          const role = BIN_ROLES[bin % BIN_ROLES.length]!;
          return (
            <g key={r.id}>
              <rect
                x={x(r.start)}
                y={yTop}
                width={Math.max(2, x(r.end) - x(r.start))}
                height={barH}
                className={categoryFillClassName(role)}
                opacity={0.85}
              />
              {/* The binning key is the start, not the span — marked so the rule is readable. */}
              <circle
                cx={x(r.start)}
                cy={yTop + barH / 2}
                r={3}
                className={inkClassName("primary")}
                fill="currentColor"
              />
              <text
                x={x(r.start) + 6}
                y={yTop + barH - 1}
                fontSize={10}
                fontWeight={600}
                className={inkClassName("primary")}
                fill="currentColor"
              >
                {r.id}
              </text>
            </g>
          );
        })}

        <line
          x1={layout.xLeft}
          y1={layout.axisY}
          x2={layout.xRight}
          y2={layout.axisY}
          className={strokeClassName("primary")}
          stroke="currentColor"
        />
        {data.axisLabel !== undefined && (
          <text
            x={(layout.xLeft + layout.xRight) / 2}
            y={layout.svgHeight - 4}
            textAnchor="middle"
            fontSize={10}
            className={inkClassName("tertiary")}
            fill="currentColor"
          >
            {data.axisLabel}
          </text>
        )}
      </svg>
      <NodeAnchorHandles />
    </div>
  );
}
