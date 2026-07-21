/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import clsx from "clsx";
import type { CategoryRole } from "../theme/tokens.js";

export type SwatchProps = {
  /** One of the shared category hues. Matches `categoryClassName` in `theme/tokens.ts`. */
  color: CategoryRole;
  className?: string;
};

/** Small filled square used as the color key for a `Legend` entry. */
export function Swatch({ color, className }: SwatchProps): React.JSX.Element {
  return (
    <span
      className={clsx("h-3 w-3 shrink-0 rounded-none", `bg-category-${color}`, className)}
      aria-hidden="true"
    />
  );
}
