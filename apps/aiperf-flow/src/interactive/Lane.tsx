/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! One swimlane band of a `TimelineTrack`: a full-width subsystem row tinted by its category tone,
//! plus its label in the left gutter. The band `<rect>` is painted with `categoryFillClassName`
//! (real `fill-category-*`) at low opacity — never `categoryBgClassName` (a `bg-*` class is inert on
//! SVG). See SKILL.md "SVG shapes need fill-/stroke- classes".

import { categoryFillClassName, inkClassName } from "../theme/tokens.js";
import type { CategoryRole } from "../theme/tokens.js";

export interface LaneProps {
  /** Left pixel edge of the lane band (usually the axis left). */
  x: number;
  /** Top pixel edge of the band. */
  y: number;
  /** Band width in pixels. */
  width: number;
  /** Band height in pixels. */
  height: number;
  /** Lane label drawn in the left gutter. */
  label: string;
  /** x pixel position of the left-gutter label. */
  labelX: number;
  /** Category tone tinting the band. */
  tone: CategoryRole;
}

/** A single tinted swimlane band + its gutter label. */
export function Lane({ x, y, width, height, label, labelX, tone }: LaneProps): React.JSX.Element {
  return (
    <g data-testid="lane">
      <rect
        x={x}
        y={y}
        width={width}
        height={height}
        rx={6}
        className={categoryFillClassName(tone)}
        fillOpacity={0.06}
      />
      <text
        x={labelX}
        y={y + height / 2}
        dominantBaseline="middle"
        className={`text-[11px] font-semibold ${inkClassName("secondary")}`}
        fill="currentColor"
      >
        {label}
      </text>
    </g>
  );
}
