/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { ReactNode } from "react";
import clsx from "clsx";
import {
  surfaceClassName,
  strokeClassName,
  inkClassName,
  categoryClassName,
} from "../theme/tokens.js";
import type { CategoryRole } from "../theme/tokens.js";

export type CalloutTone = "info" | "warning" | "danger" | "success" | "neutral";

export type CalloutProps = {
  /** Semantic tone. Selects the accent border and title color. Defaults to `"info"`. */
  tone?: CalloutTone;
  /** Optional bold title line, shown above the body in the tone color. */
  title?: string;
  /** Body content. */
  children: ReactNode;
  className?: string;
};

const toneCategory: Record<CalloutTone, CategoryRole> = {
  info: "blue",
  warning: "yellow",
  danger: "red",
  success: "green",
  neutral: "gray",
};

// Tailwind's compiler only picks up classes it can see as literal strings in source, so a
// dynamically interpolated `border-l-category-${category}` would be purged for any role whose
// exact string doesn't happen to appear verbatim elsewhere (see the same note in theme/tokens.ts).
const CATEGORY_BORDER_L_CLASSES: Record<CategoryRole, string> = {
  green: "border-l-category-green",
  yellow: "border-l-category-yellow",
  purple: "border-l-category-purple",
  blue: "border-l-category-blue",
  red: "border-l-category-red",
  orange: "border-l-category-orange",
  cyan: "border-l-category-cyan",
  gray: "border-l-category-gray",
};

/** Tone-colored admonition box for calling out important notes in prose content. */
export function Callout({
  tone = "info",
  title,
  children,
  className,
}: CalloutProps): React.JSX.Element {
  const category = toneCategory[tone];
  return (
    <div
      className={clsx(
        "rounded-none border border-l-4 px-4 py-3",
        surfaceClassName("elevated"),
        strokeClassName("secondary"),
        CATEGORY_BORDER_L_CLASSES[category],
        className,
      )}
    >
      {title !== undefined && (
        <div className={clsx("mb-1 text-sm font-semibold", categoryClassName(category))}>
          {title}
        </div>
      )}
      <div className={clsx("text-sm", inkClassName("primary"))}>{children}</div>
    </div>
  );
}
