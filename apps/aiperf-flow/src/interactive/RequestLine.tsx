/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! The single hero request line weaving through the timeline's events. A `<polyline>` through the
//! precomputed event points, stroked with `categoryStrokeClassName` (SVG-safe). Colored by the
//! active clock tone (e.g. green for RealClock, purple for SimClock) so the seam is legible at a
//! glance. Respects reduced motion: a subtle draw-in animation is applied only when motion is
//! allowed. See SKILL.md "SVG shapes need fill-/stroke- classes".

import { motion } from "motion/react";
import { categoryStrokeClassName } from "../theme/tokens.js";
import type { CategoryRole } from "../theme/tokens.js";

/** A pixel point on the request line. */
export interface LinePoint {
  x: number;
  y: number;
}

export interface RequestLineProps {
  /** Ordered pixel points the line threads through. */
  points: readonly LinePoint[];
  /** Category tone for the stroke. */
  tone: CategoryRole;
  /** When true, no draw-in animation is applied. */
  reducedMotion?: boolean;
}

/** The weaving request polyline. */
export function RequestLine({ points, tone, reducedMotion = false }: RequestLineProps): React.JSX.Element {
  const d = points.map((p) => `${p.x},${p.y}`).join(" ");
  return (
    <motion.polyline
      data-testid="request-line"
      points={d}
      fill="none"
      strokeWidth={2.5}
      strokeLinejoin="round"
      strokeLinecap="round"
      className={categoryStrokeClassName(tone)}
      initial={reducedMotion ? false : { pathLength: 0, opacity: 0.4 }}
      animate={{ pathLength: 1, opacity: 1 }}
      transition={{ duration: reducedMotion ? 0 : 0.8, ease: "easeInOut" }}
    />
  );
}
