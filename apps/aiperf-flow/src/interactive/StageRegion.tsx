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

// Approx advance width (px) of one glyph at the 11px semibold label size — used to decide whether a
// label fits its block, and how many chars survive truncation. Deliberately conservative.
const CHAR_PX = 6.2;
const LABEL_PAD_X = 12;

/** A clickable, labeled stage block on the timeline. Labels truncate to fit; the full name is kept
 * in a `<title>` (hover tooltip) whenever it is clipped, and always in the `aria-label`. */
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
  const fits = label.length * CHAR_PX + LABEL_PAD_X <= width;
  const maxChars = Math.floor((width - LABEL_PAD_X) / CHAR_PX);
  // Only paint a label when at least a few characters survive; a sliver block stays unlabeled and
  // relies on its lane gutter + the hover title + the drill affordance.
  const showText = width >= 30 && maxChars >= 3;
  const display = fits ? label : `${label.slice(0, Math.max(1, maxChars - 1))}…`;
  return (
    <g
      data-testid="stage-region"
      data-active={active ? "true" : "false"}
      role={onClick ? "button" : undefined}
      aria-label={onClick ? `Drill into ${label}` : undefined}
      onClick={onClick}
      className={onClick ? "cursor-zoom-in" : undefined}
    >
      {/* Full name on hover, only when the visible text is clipped (so exactly one element ever
          carries the full label — keeps getByText unambiguous). */}
      {!fits && <title>{label}</title>}
      <rect
        x={x}
        y={y}
        width={width}
        height={height}
        rx={8}
        strokeWidth={active ? 2 : 1}
        className={`${categoryFillClassName(tone)} ${categoryStrokeClassName(tone)}`}
        fillOpacity={active ? 0.32 : 0.16}
      />
      {showText && (
        <text
          x={x + width / 2}
          y={y + height / 2}
          textAnchor="middle"
          dominantBaseline="middle"
          className={`pointer-events-none text-[11px] font-semibold ${inkClassName("primary")}`}
          fill="currentColor"
        >
          {display}
        </text>
      )}
    </g>
  );
}
