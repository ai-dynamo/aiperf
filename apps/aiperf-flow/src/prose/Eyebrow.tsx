/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { ReactNode } from "react";
import clsx from "clsx";
import { categoryClassName, inkClassName } from "../theme/tokens.js";
import type { CategoryRole } from "../theme/tokens.js";

export type EyebrowProps = {
  children: ReactNode;
  /** Category color instead of the default tertiary ink (ports a status/kicker label's color). */
  tone?: CategoryRole;
  className?: string;
};

/**
 * Small uppercase section/status label — a "kicker" above a heading, a section eyebrow, or a
 * colored status word. Consolidates the `uppercase tracking-wide text-xs` span this app's decks
 * independently inlined 17+ times, and the local `Eyebrow`/`StatusPill`-named one-offs built for
 * the same shape. Use this instead of writing that span by hand.
 */
export function Eyebrow({ children, tone, className }: EyebrowProps): React.JSX.Element {
  return (
    <span
      className={clsx(
        "text-xs font-semibold uppercase tracking-wide",
        tone ? categoryClassName(tone) : inkClassName("tertiary"),
        className,
      )}
    >
      {children}
    </span>
  );
}
