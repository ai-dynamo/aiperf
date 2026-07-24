/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! The horizontal time axis of a `TimelineTrack`: a baseline rule, per-event tick marks + labels,
//! and a unit caption that names the active clock scale (wall-ms vs virtual ticks). Pure SVG —
//! coordinates are precomputed by `TimelineTrack`; colors come from `inkClassName` with an explicit
//! `stroke`/`fill="currentColor"` (role colors have no `fill-*`/`stroke-*` helper). See SKILL.md
//! "SVG shapes need fill-/stroke- classes".

import { inkClassName } from "../theme/tokens.js";

/** One tick on the axis: an x pixel position and its label. */
export interface TimeAxisTick {
  x: number;
  label: string;
}

export interface TimeAxisProps {
  /** Left pixel edge of the axis line. */
  x1: number;
  /** Right pixel edge of the axis line. */
  x2: number;
  /** Baseline y of the axis line. */
  y: number;
  /** Tick marks (one per request event, typically). */
  ticks: readonly TimeAxisTick[];
  /** Unit caption naming the active scale, e.g. "RealClock · wall-ms". */
  unitLabel: string;
}

/** The axis rule + tick marks/labels + a unit caption for the active clock scale. */
export function TimeAxis({ x1, x2, y, ticks, unitLabel }: TimeAxisProps): React.JSX.Element {
  return (
    <g data-testid="time-axis">
      <text
        x={x1}
        y={y - 14}
        className={`text-[10px] font-semibold uppercase tracking-wide ${inkClassName("tertiary")}`}
        fill="currentColor"
      >
        {unitLabel}
      </text>
      <line
        x1={x1}
        x2={x2}
        y1={y}
        y2={y}
        strokeWidth={1}
        className={inkClassName("quaternary")}
        stroke="currentColor"
      />
      {ticks.map((tick) => (
        <g key={`${tick.label}-${tick.x}`}>
          <line
            x1={tick.x}
            x2={tick.x}
            y1={y - 4}
            y2={y + 4}
            strokeWidth={1}
            className={inkClassName("quaternary")}
            stroke="currentColor"
          />
          <text
            x={tick.x}
            y={y - 8}
            textAnchor="middle"
            className={`text-[9px] ${inkClassName("tertiary")}`}
            fill="currentColor"
          >
            {tick.label}
          </text>
        </g>
      ))}
    </g>
  );
}
