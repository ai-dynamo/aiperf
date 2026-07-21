/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import clsx from "clsx";
import { inkClassName } from "../theme/tokens.js";
import type { CategoryRole } from "../theme/tokens.js";
import { Swatch } from "./Swatch.js";

export type LegendEntry = {
  /** Category hue for this entry's swatch. */
  color: CategoryRole;
  /** Text explaining what the color means. */
  label: string;
};

export type LegendProps = {
  /** Ordered set of color-to-meaning entries to display. */
  entries: LegendEntry[];
  className?: string;
};

/**
 * Horizontal, wrapping row of color-swatch-plus-label entries explaining what each
 * color means in a diagram (e.g. a green swatch next to "Healthy", a red swatch next
 * to "Failed").
 */
export function Legend({ entries, className }: LegendProps): React.JSX.Element {
  return (
    <div className={clsx("flex flex-wrap gap-4", className)}>
      {entries.map((entry) => (
        <div key={`${entry.color}-${entry.label}`} className="flex items-center gap-2.5">
          <Swatch color={entry.color} />
          <span className={clsx("text-sm font-medium tracking-tight", inkClassName("secondary"))}>
            {entry.label}
          </span>
        </div>
      ))}
    </div>
  );
}
