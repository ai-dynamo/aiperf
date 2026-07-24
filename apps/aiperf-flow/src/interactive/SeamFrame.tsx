/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! A translucent seam frame grouping a band of the timeline (the nested-composition view). A dashed
//! `<rect>` outline tinted by tone + a small top-left label. Uses `categoryStrokeClassName` (stroke)
//! and `categoryFillClassName` (a barely-there fill) — the SVG-safe helpers. See SKILL.md "SVG
//! shapes need fill-/stroke- classes".

import { categoryClassName, categoryFillClassName, categoryStrokeClassName } from "../theme/tokens.js";
import type { CategoryRole } from "../theme/tokens.js";

export interface SeamFrameProps {
  /** Left pixel edge. */
  x: number;
  /** Top pixel edge. */
  y: number;
  /** Frame width in pixels. */
  width: number;
  /** Frame height in pixels. */
  height: number;
  /** Frame label drawn at the top-left. */
  label: string;
  /** Category tone. */
  tone: CategoryRole;
}

/** A dashed, translucent seam frame with a corner label. */
export function SeamFrame({ x, y, width, height, label, tone }: SeamFrameProps): React.JSX.Element {
  return (
    <g data-testid="seam-frame" className="pointer-events-none">
      <rect
        x={x}
        y={y}
        width={width}
        height={height}
        rx={10}
        strokeWidth={1.5}
        strokeDasharray="5 4"
        className={`${categoryFillClassName(tone)} ${categoryStrokeClassName(tone)}`}
        fillOpacity={0.04}
      />
      {/* Label rides just above the frame's top edge, so it never lands on a stage block's text. */}
      <text
        x={x + 6}
        y={y - 4}
        className={`text-[10px] font-semibold uppercase tracking-wide ${categoryClassName(tone)}`}
        fill="currentColor"
      >
        {label}
      </text>
    </g>
  );
}
