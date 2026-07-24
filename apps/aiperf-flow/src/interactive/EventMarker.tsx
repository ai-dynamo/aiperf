/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! One event dot on the request line. A filled `<circle>` (via `categoryFillClassName`) with an
//! optional label above it; when `active` (the play head is here) it grows and gains a ring —
//! animated only when motion is allowed. Colors via the SVG-safe `categoryFillClassName`/
//! `categoryStrokeClassName`. See SKILL.md "SVG shapes need fill-/stroke- classes".

import { motion, useReducedMotion } from "motion/react";
import { categoryFillClassName, categoryStrokeClassName, inkClassName } from "../theme/tokens.js";
import type { CategoryRole } from "../theme/tokens.js";

export interface EventMarkerProps {
  /** Center x in pixels. */
  x: number;
  /** Center y in pixels. */
  y: number;
  /** Category tone. */
  tone: CategoryRole;
  /** Whether the play head is currently on this event. */
  active?: boolean;
  /** Optional label drawn above the dot. */
  label?: string;
}

/** A single event dot (+ optional label), highlighted when active. */
export function EventMarker({ x, y, tone, active = false, label }: EventMarkerProps): React.JSX.Element {
  const prefersReduced = useReducedMotion() ?? false;
  return (
    <g data-testid="event-marker" data-active={active ? "true" : "false"}>
      {active && (
        <motion.circle
          cx={x}
          cy={y}
          r={9}
          fillOpacity={0.25}
          className={categoryFillClassName(tone)}
          animate={prefersReduced ? undefined : { scale: [1, 1.35, 1] }}
          transition={prefersReduced ? undefined : { duration: 1.1, repeat: Infinity, ease: "easeInOut" }}
          style={{ transformBox: "fill-box", transformOrigin: "center" }}
        />
      )}
      <circle
        cx={x}
        cy={y}
        r={active ? 5 : 3.5}
        strokeWidth={active ? 1.5 : 0}
        className={`${categoryFillClassName(tone)} ${categoryStrokeClassName(tone)}`}
      />
      {label !== undefined && (
        <text
          x={x}
          y={y - 12}
          textAnchor="middle"
          className={`text-[9px] font-medium ${active ? inkClassName("primary") : inkClassName("tertiary")}`}
          fill="currentColor"
        >
          {label}
        </text>
      )}
    </g>
  );
}
