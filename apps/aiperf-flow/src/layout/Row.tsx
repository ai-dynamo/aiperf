/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { ReactNode } from "react";
import clsx from "clsx";

//! Horizontal flex-row layout primitive; see `DESIGN.md` for the prop-to-Tailwind mapping.

const DEFAULT_GAP_PX = 16;

export type RowAlign = "start" | "center" | "end" | "stretch";
export type RowJustify = "start" | "center" | "end" | "space-between";

const ALIGN_CLASSES: Record<RowAlign, string> = {
  start: "items-start",
  center: "items-center",
  end: "items-end",
  stretch: "items-stretch",
};

const JUSTIFY_CLASSES: Record<RowJustify, string> = {
  start: "justify-start",
  center: "justify-center",
  end: "justify-end",
  "space-between": "justify-between",
};

export type RowProps = {
  children?: ReactNode;
  gap?: number;
  align?: RowAlign;
  justify?: RowJustify;
  wrap?: boolean;
  className?: string;
};

export function Row({
  children,
  gap = DEFAULT_GAP_PX,
  align,
  justify,
  wrap = false,
  className,
}: RowProps): React.JSX.Element {
  return (
    <div
      className={clsx(
        "flex flex-row",
        align !== undefined && ALIGN_CLASSES[align],
        justify !== undefined && JUSTIFY_CLASSES[justify],
        wrap && "flex-wrap",
        className,
      )}
      style={{ gap }}
    >
      {children}
    </div>
  );
}
