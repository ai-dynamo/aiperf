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
};

export function FlowArrow({
  d,
  markerId,
  color,
  fill = "none",
  strokeWidth = 2.2,
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
      markerEnd={`url(#${markerId})`}
    />
  );
}
