/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Lightweight SVG chart primitives: `BarChart` and `LineChart`. Both compute their own layout
//! by hand from a small, fixed-shape data array — no charting library dependency.

import clsx from "clsx";
import type { CategoryRole } from "../theme/tokens.js";
import { categoryBgClassName, inkClassName, strokeClassName } from "../theme/tokens.js";

export type ChartDatum = {
  label: string;
  value: number;
};

type ChartLayoutProps = {
  /** Data points, one per bar or line vertex. */
  data: ChartDatum[];
  /** Total SVG height in pixels, including the x-axis label row. Defaults to `160`. */
  height?: number;
  /** Category color used for bar fill or line marker fill. Defaults to `"blue"`. */
  color?: CategoryRole;
  /** Extra classes merged onto the svg root's own classes, appended last. */
  className?: string;
};

const CHART_WIDTH = 320;
const LABEL_ROW_HEIGHT = 20;
const PLOT_PADDING = 8;

/**
 * Vertical bar chart scaled to the max value in `data`, with an x-axis label under each bar.
 *
 * @example
 * ```tsx
 * <BarChart data={[{ label: "Mon", value: 10 }, { label: "Tue", value: 40 }]} color="purple" />
 * ```
 */
export function BarChart({
  data,
  height = 160,
  color = "blue",
  className,
}: ChartLayoutProps): React.JSX.Element {
  const plotHeight = height - LABEL_ROW_HEIGHT;
  const maxValue = Math.max(0, ...data.map((d) => d.value));
  const slotWidth = data.length > 0 ? CHART_WIDTH / data.length : CHART_WIDTH;
  const barWidth = Math.max(0, slotWidth - PLOT_PADDING);
  const fillClassName = categoryBgClassName(color);

  return (
    <svg
      className={clsx("w-full", className)}
      viewBox={`0 0 ${CHART_WIDTH} ${height}`}
      role="img"
    >
      <line
        x1={0}
        y1={plotHeight}
        x2={CHART_WIDTH}
        y2={plotHeight}
        className={strokeClassName("secondary")}
        stroke="currentColor"
      />
      {data.map((d, i) => {
        const barHeight = maxValue > 0 ? (d.value / maxValue) * (plotHeight - PLOT_PADDING) : 0;
        const x = i * slotWidth + PLOT_PADDING / 2;
        const y = plotHeight - barHeight;
        return (
          <g key={`${d.label}-${i}`}>
            <rect
              x={x}
              y={y}
              width={barWidth}
              height={barHeight}
              className={fillClassName}
            />
            <text
              x={x + barWidth / 2}
              y={height - 4}
              textAnchor="middle"
              className={`text-[9px] ${inkClassName("secondary")}`}
              fill="currentColor"
            >
              {d.label}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

/**
 * Single-series line chart scaled to the value range of `data`, with small circle markers at
 * each point and an x-axis label under each point.
 *
 * @example
 * ```tsx
 * <LineChart data={[{ label: "Mon", value: 10 }, { label: "Tue", value: 40 }]} color="green" />
 * ```
 */
export function LineChart({
  data,
  height = 160,
  color = "blue",
  className,
}: ChartLayoutProps): React.JSX.Element {
  const plotHeight = height - LABEL_ROW_HEIGHT;
  const values = data.map((d) => d.value);
  const maxValue = values.length > 0 ? Math.max(...values) : 0;
  const minValue = values.length > 0 ? Math.min(...values) : 0;
  const valueRange = maxValue - minValue;
  const fillClassName = categoryBgClassName(color);

  const points = data.map((d, i) => {
    const x =
      data.length > 1 ? (i / (data.length - 1)) * CHART_WIDTH : CHART_WIDTH / 2;
    const normalized = valueRange > 0 ? (d.value - minValue) / valueRange : 0.5;
    const y = plotHeight - PLOT_PADDING - normalized * (plotHeight - 2 * PLOT_PADDING);
    return { ...d, x, y };
  });

  return (
    <svg
      className={clsx("w-full", className)}
      viewBox={`0 0 ${CHART_WIDTH} ${height}`}
      role="img"
    >
      <line
        x1={0}
        y1={plotHeight}
        x2={CHART_WIDTH}
        y2={plotHeight}
        className={strokeClassName("secondary")}
        stroke="currentColor"
      />
      {points.length > 0 && (
        <polyline
          points={points.map((p) => `${p.x},${p.y}`).join(" ")}
          fill="none"
          stroke="currentColor"
          strokeWidth={2}
          className={inkClassName("primary")}
        />
      )}
      {points.map((p, i) => (
        <g key={`${p.label}-${i}`}>
          <circle cx={p.x} cy={p.y} r={3} className={fillClassName} />
          <text
            x={p.x}
            y={height - 4}
            textAnchor="middle"
            className={`text-[9px] ${inkClassName("secondary")}`}
            fill="currentColor"
          >
            {p.label}
          </text>
        </g>
      ))}
    </svg>
  );
}
