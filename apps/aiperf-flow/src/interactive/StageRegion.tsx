/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! One labeled stage block positioned along the axis inside a lane. Clicking it drills into that
//! stage (the `ZoomStage` hook-up lives in the deck). The block `<rect>` uses `categoryFillClassName`
//! (fill) + `categoryStrokeClassName` (stroke) — the SVG-safe helpers — never `bg-*`/`border-*`.
//! When `active`, the fill brightens and the stroke thickens. See SKILL.md "SVG shapes need
//! fill-/stroke- classes".

import { categoryFillClassName, categoryStrokeClassName, inkClassName } from "../theme/tokens.js";
import type { CategoryRole } from "../theme/tokens.js";

export interface StageRegionProps {
  /** Left pixel edge of the block. */
  x: number;
  /** Top pixel edge of the block. */
  y: number;
  /** Block width in pixels. */
  width: number;
  /** Block height in pixels. */
  height: number;
  /** Block label. */
  label: string;
  /** Category tone. */
  tone: CategoryRole;
  /** Whether the play head is currently inside this region. */
  active?: boolean;
  /** Click handler (drills into the stage). */
  onClick?: () => void;
}

/** A clickable, labeled stage block on the timeline. The full stage name renders as one line, kept
 * intact (queryable + accessible); even columns keep blocks wide enough that names fit. */
export function StageRegion({
  x,
  y,
  width,
  height,
  label,
  tone,
  active = false,
  onClick,
}: StageRegionProps): React.JSX.Element {
  return (
    <g
      data-testid="stage-region"
      data-active={active ? "true" : "false"}
      role={onClick ? "button" : undefined}
      aria-label={onClick ? `Drill into ${label}` : undefined}
      onClick={onClick}
      className={onClick ? "cursor-zoom-in" : undefined}
    >
      <rect
        x={x}
        y={y}
        width={width}
        height={height}
        rx={8}
        strokeWidth={active ? 2 : 1}
        className={`${categoryFillClassName(tone)} ${categoryStrokeClassName(tone)}`}
        fillOpacity={active ? 0.34 : 0.14}
      />
      <text
        x={x + width / 2}
        y={y + height / 2}
        textAnchor="middle"
        dominantBaseline="middle"
        className={`pointer-events-none text-[12px] font-semibold ${inkClassName("primary")}`}
        fill="currentColor"
      >
        {label}
      </text>
    </g>
  );
}
