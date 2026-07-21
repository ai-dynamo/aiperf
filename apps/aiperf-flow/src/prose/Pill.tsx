/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { ReactNode } from "react";
import clsx from "clsx";
import {
  accentClassName,
  categoryBgTintClassName,
  categoryClassName,
  inkClassName,
  strokeClassName,
} from "../theme/tokens.js";
import type { CategoryRole } from "../theme/tokens.js";

export type PillProps = {
  children: ReactNode;
  /** Reads as the selected/current variant: accent border and ink. Ignored when `tone` is set. */
  active?: boolean;
  /** Category color instead of the neutral active/inactive styling (ports "TonePill"-style tags). */
  tone?: CategoryRole;
  /** Renders as a `<button>` and applies `aria-pressed={active}` when provided. */
  onClick?: () => void;
  className?: string;
};

/**
 * Compact tag/status chip shared by every deck. Consolidates the `Pill` / `Badge` / `TonePill`
 * one-offs that independent porting agents each built locally for the same shape: a
 * `rounded-none` bordered label, optionally clickable/toggleable, optionally colored by
 * `CategoryRole` instead of the neutral active/inactive palette.
 */
export function Pill({ children, active = false, tone, onClick, className }: PillProps): React.JSX.Element {
  const base = clsx(
    "inline-flex items-center rounded-none border px-2 py-0.5 text-xs font-semibold",
    tone
      ? clsx("border-transparent", categoryBgTintClassName(tone), categoryClassName(tone))
      : active
        ? clsx("border-accent-primary", accentClassName("primary"))
        : clsx(strokeClassName("secondary"), inkClassName("secondary")),
    onClick && "cursor-pointer",
    className,
  );

  if (onClick) {
    return (
      <button type="button" className={base} onClick={onClick} aria-pressed={active}>
        {children}
      </button>
    );
  }
  return <span className={base}>{children}</span>;
}
