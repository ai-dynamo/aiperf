/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { ReactNode } from "react";
import clsx from "clsx";
import { surfaceClassName, strokeClassName, inkClassName } from "../theme/tokens.js";
import type { CategoryRole } from "../theme/tokens.js";

export type CalloutTone = "info" | "warning" | "danger" | "success";

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
        `border-l-category-${category}`,
        className,
      )}
    >
      {title !== undefined && (
        <div className={clsx("mb-1 text-sm font-semibold", `text-category-${category}`)}>
          {title}
        </div>
      )}
      <div className={clsx("text-sm", inkClassName("primary"))}>{children}</div>
    </div>
  );
}
