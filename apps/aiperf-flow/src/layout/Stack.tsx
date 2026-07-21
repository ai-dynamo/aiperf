/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { ReactNode } from "react";
import clsx from "clsx";

//! Vertical flex-column layout primitive; see `DESIGN.md` for the prop-to-Tailwind mapping.

const DEFAULT_GAP_PX = 16;

export type StackProps = {
  children?: ReactNode;
  gap?: number;
  className?: string;
};

export function Stack({ children, gap = DEFAULT_GAP_PX, className }: StackProps): React.JSX.Element {
  return (
    <div className={clsx("flex flex-col", className)} style={{ gap }}>
      {children}
    </div>
  );
}
