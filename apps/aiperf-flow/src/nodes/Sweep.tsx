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
} from "../theme/tokens.js";
import { NodeAnchorHandles } from "./anchors.js";
import { layoutSweep } from "./sweepLayout.js";
import { buildEvents, stepPathD, stepPoints } from "./sweepMath.js";
import type { SweepNodeData } from "./types.js";

export type SweepNodeType = Node<SweepNodeData, "sweep">;

/**
 * Intervals over the step function they generate, on one shared time axis.
 *
 * The claim this makes is the sweep-line identity: every interval contributes `+weight` at its
 * start and `-weight` at its end, and a running cumsum over those events is *exactly* the curve —
 * no point-by-point scan of the timeline. Drawing the Gantt directly above the step plot lets a
 * reader check that identity by eye, following a bar's edges down to the tick that moves the curve.
 *
 * Which curve is chosen changes only the weight: concurrency weighs 1, tokens weighs output
 * tokens, throughput weighs tokens per decode second.
 */
export function SweepNode({ data }: NodeProps<SweepNodeType>): React.JSX.Element {
  const { requests, title } = data;
  const curve = data.curve ?? "concurrency";
  const layout = layoutSweep({
    requests,
    curve,
    hasTitle: title !== undefined,
    tMax: data.tMax,
    width: data.width,
  });
  const { x, y, xLeft, xRight, top } = layout;

  const events = buildEvents(requests, curve);
  const points = stepPoints(events);
  const path = stepPathD(points, x, y, 0, layout.tMax);

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
        <div className={`mb-1.5 text-sm font-semibold leading-[24px] tracking-tight ${inkClassName("primary")}`}>
          {title}
        </div>
      )}
      {/* `block` drops the inline baseline strut: an inline <svg> reserves descender space
          below it, making the rendered box ~3px taller than the declared *NodeSize() height. */}
      <svg
        className="block"
        width={layout.svgWidth}
        height={layout.svgHeight}
        role="img"
        aria-label={data.ariaLabel ?? `sweep-line ${curve} chart`}
      >
        {layout.tTicks.map((t) => (
          <line
            key={`grid-${t}`}
            x1={x(t)}
            y1={top}
            x2={x(t)}
            y2={layout.axisY}
            className={strokeClassName("tertiary")}
            stroke="currentColor"
          />
        ))}

        {requests.map((r, i) => {
          const yTop = top + i * layout.rowHeight + 3;
          const barH = layout.rowHeight - 8;
          return (
            <g key={r.id}>
              <text
                x={xLeft - 8}
                y={yTop + barH - 2}
                textAnchor="end"
                fontSize={11}
                fontWeight={600}
                className={inkClassName("secondary")}
                fill="currentColor"
              >
                {r.id}
              </text>
              {/* Prefill and decode are drawn as separate spans: the throughput curve weighs only
                  the decode window, so the split is what makes that curve's shape legible. */}
              <rect
                x={x(r.start)}
                y={yTop}
                width={Math.max(1, x(r.gen) - x(r.start))}
                height={barH}
                className={categoryFillClassName("gray")}
              />
              <rect
                x={x(r.gen)}
                y={yTop}
                width={Math.max(1, x(r.end) - x(r.gen))}
                height={barH}
                className={categoryFillClassName("blue")}
                opacity={0.85}
              />
            </g>
          );
        })}

        <line
          x1={xLeft}
          y1={top + layout.ganttHeight + 12}
          x2={xRight}
          y2={top + layout.ganttHeight + 12}
          className={strokeClassName("secondary")}
          stroke="currentColor"
          strokeDasharray="3 3"
        />

        {layout.vTicks.map((v, i) => (
          <g key={`v-${i}`}>
            <line
              x1={xLeft}
              y1={y(v)}
              x2={xRight}
              y2={y(v)}
              className={strokeClassName("tertiary")}
              stroke="currentColor"
            />
            <text
              x={xLeft - 8}
              y={y(v) + 3}
              textAnchor="end"
              fontSize={10}
              className={inkClassName("tertiary")}
              fill="currentColor"
            >
              {Number.isInteger(v) ? v : v.toFixed(1)}
            </text>
          </g>
        ))}

        <path
          d={`${path} L ${x(layout.tMax)} ${y(0)} L ${x(0)} ${y(0)} Z`}
          className={categoryFillClassName("blue")}
          opacity={0.12}
        />
        <path d={path} fill="none" className={categoryClassName("blue")} stroke="currentColor" strokeWidth={2} />

        {events.map((e, i) => (
          <line
            key={`evt-${i}`}
            x1={x(e.t)}
            y1={layout.axisY - 4}
            x2={x(e.t)}
            y2={layout.axisY + 4}
            className={e.d > 0 ? categoryClassName("green") : categoryClassName("orange")}
            stroke="currentColor"
            strokeWidth={1.5}
          />
        ))}

        <line x1={xLeft} y1={layout.axisY} x2={xRight} y2={layout.axisY} className={strokeClassName("primary")} stroke="currentColor" />
        {layout.tTicks.map((t) => (
          <text
            key={`tick-${t}`}
            x={x(t)}
            y={layout.axisY + 18}
            textAnchor="middle"
            fontSize={10}
            className={inkClassName("tertiary")}
            fill="currentColor"
          >
            {t}
          </text>
        ))}
        {data.axisLabel !== undefined && (
          <text
            x={(xLeft + xRight) / 2}
            y={layout.svgHeight - 2}
            textAnchor="middle"
            fontSize={10}
            className={inkClassName("tertiary")}
            fill="currentColor"
          >
            {data.axisLabel}
          </text>
        )}
        {data.valueLabel !== undefined && (
          <text
            x={14}
            y={layout.stepTop + layout.stepHeight / 2}
            textAnchor="middle"
            fontSize={10}
            className={inkClassName("tertiary")}
            fill="currentColor"
            transform={`rotate(-90 14 ${layout.stepTop + layout.stepHeight / 2})`}
          >
            {data.valueLabel}
          </text>
        )}
      </svg>
      <NodeAnchorHandles />
    </div>
  );
}
