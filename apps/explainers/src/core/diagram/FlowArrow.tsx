/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { SVGProps } from "react";
import { useHostTheme } from "../ui";

export type FlowArrowProps = Omit<
  SVGProps<SVGPathElement>,
  "color" | "d" | "markerEnd"
> & {
  d: string;
  markerId: string;
  color?: string;
  /** When false, omit marker-end (motion guides / undirected strokes). */
  showMarker?: boolean;
  /** When true, applies a dashed stroke (ignored while draw-reveal owns dasharray). */
  dashed?: boolean;
};

const DASHED_STROKE = "8 4";

/** Themed path with optional arrowhead marker and dashed stroke. */
export function FlowArrow({
  d,
  markerId,
  color,
  showMarker = true,
  dashed = false,
  fill = "none",
  strokeWidth = 2.2,
  strokeDasharray,
  ...pathProps
}: FlowArrowProps) {
  const theme = useHostTheme();

  return (
    <path
      {...pathProps}
      d={d}
      fill={fill}
      stroke={color ?? theme.category.green}
      strokeWidth={strokeWidth}
      strokeDasharray={
        strokeDasharray !== undefined
          ? strokeDasharray
          : dashed
            ? DASHED_STROKE
            : undefined
      }
      markerEnd={showMarker ? `url(#${markerId})` : undefined}
    />
  );
}
