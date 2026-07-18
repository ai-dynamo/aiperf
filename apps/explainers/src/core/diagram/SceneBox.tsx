/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { SVGProps } from "react";
import { useHostTheme, type Theme } from "../ui";

export type SceneBoxProps = Omit<SVGProps<SVGGElement>, "title"> & {
  x: number;
  y: number;
  width: number;
  height: number;
  title: string;
  detail: string;
  accent?: keyof Theme["category"];
};

export function SceneBox({
  x,
  y,
  width,
  height,
  title,
  detail,
  accent,
  ...groupProps
}: SceneBoxProps) {
  const theme = useHostTheme();
  const stroke = accent ? theme.category[accent] : theme.stroke.secondary;

  return (
    <g aria-label={`${title}: ${detail}`} {...groupProps}>
      <rect
        x={x}
        y={y}
        width={width}
        height={height}
        rx={10}
        fill={theme.bg.elevated}
        stroke={stroke}
        strokeWidth={accent ? 1.8 : 1.3}
      />
      <text
        x={x + width / 2}
        y={y + height / 2 - 8}
        textAnchor="middle"
        fill={theme.text.primary}
        fontSize={14}
        fontWeight={700}
      >
        {title}
      </text>
      <text
        x={x + width / 2}
        y={y + height / 2 + 16}
        textAnchor="middle"
        fill={theme.text.secondary}
        fontSize={11}
      >
        {detail}
      </text>
    </g>
  );
}
