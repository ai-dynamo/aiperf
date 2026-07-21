/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { CSSProperties, ReactNode } from "react";
import clsx from "clsx";

//! CSS Grid layout primitive; see `DESIGN.md` for the prop-to-Tailwind mapping, in particular why
//! `columns` cannot be a single Tailwind-only implementation.

const DEFAULT_GAP_PX = 16;

export type GridAlign = "start" | "center" | "end" | "stretch";

const ALIGN_CLASSES: Record<GridAlign, string> = {
  start: "items-start",
  center: "items-center",
  end: "items-end",
  stretch: "items-stretch",
};

// Tailwind's compiler only picks up classes it can see as literal strings in source, so a
// dynamically interpolated `grid-cols-${n}` would be purged. This table keeps every supported
// column count visible as a whole string.
const GRID_COLS_CLASSES: Record<number, string> = {
  1: "grid-cols-1",
  2: "grid-cols-2",
  3: "grid-cols-3",
  4: "grid-cols-4",
  5: "grid-cols-5",
  6: "grid-cols-6",
  7: "grid-cols-7",
  8: "grid-cols-8",
  9: "grid-cols-9",
  10: "grid-cols-10",
  11: "grid-cols-11",
  12: "grid-cols-12",
};

export type GridProps = {
  children?: ReactNode;
  columns: number | string;
  gap?: number;
  align?: GridAlign;
  className?: string;
};

export function Grid({
  children,
  columns,
  gap = DEFAULT_GAP_PX,
  align,
  className,
}: GridProps): React.JSX.Element {
  const isNumericColumns = typeof columns === "number";
  const style: CSSProperties = isNumericColumns
    ? { gap }
    : { gap, gridTemplateColumns: columns };

  return (
    <div
      className={clsx(
        "grid",
        isNumericColumns && GRID_COLS_CLASSES[columns],
        align !== undefined && ALIGN_CLASSES[align],
        className,
      )}
      style={style}
    >
      {children}
    </div>
  );
}
