/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Small local presentational helpers scoped to the cellular-algorithm-workbook deck. The source
//! Cursor canvas used the `cursor/canvas` host's `Pill` / status chips / `useHostTheme`; the app
//! vocabulary has no direct equivalent, so these one-off components re-express them flatly using
//! `theme/tokens` role classes (no raw hex, `rounded-none`).

import type { ReactNode } from "react";
import clsx from "clsx";
import {
  inkClassName,
  strokeClassName,
  surfaceClassName,
  accentClassName,
} from "../../theme/tokens.js";
import type { Status } from "./data.js";

/** A compact tag/label; `active` reads as the selected/current variant. */
export function Pill({
  children,
  active = false,
  onClick,
  className,
}: {
  children: ReactNode;
  active?: boolean;
  onClick?: () => void;
  className?: string;
}): React.JSX.Element {
  const base = clsx(
    "inline-flex items-center rounded-none border px-2 py-0.5 text-xs font-medium",
    active
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

const STATUS_LABELS: Readonly<Record<Status, string>> = {
  built: "Built",
  partial: "Partial",
  "feature-gated": "Feature gated",
  approximate: "Approximate",
  rejected: "Rejected",
};

/** Implementation-status chip; `rejected` is drawn in the danger category, others neutral. */
export function StatusLabel({ status }: { status: Status }): React.JSX.Element {
  const rejected = status === "rejected";
  return (
    <span
      aria-label={`Implementation status: ${STATUS_LABELS[status]}`}
      className={clsx(
        "inline-flex items-center rounded-none border px-2 py-0.5 text-xs font-medium",
        rejected
          ? "border-category-red text-category-red"
          : clsx(strokeClassName("secondary"), inkClassName("secondary")),
      )}
    >
      {STATUS_LABELS[status]}
    </span>
  );
}

/** Route admission chip: Admitted (neutral) vs Rejected (danger). */
export function AdmissionLabel({ valid }: { valid: boolean }): React.JSX.Element {
  return (
    <span
      aria-label={`Route admission: ${valid ? "Admitted" : "Rejected"}`}
      className={clsx(
        "inline-flex items-center rounded-none border px-2 py-0.5 text-xs font-medium",
        valid
          ? clsx(strokeClassName("secondary"), inkClassName("secondary"))
          : "border-category-red text-category-red",
      )}
    >
      {valid ? "Admitted" : "Rejected"}
    </span>
  );
}

/** An uppercase eyebrow / section label. */
export function Eyebrow({ children }: { children: ReactNode }): React.JSX.Element {
  return (
    <span className={clsx("text-xs font-semibold uppercase tracking-wide", inkClassName("tertiary"))}>
      {children}
    </span>
  );
}

/** A soft-bordered content panel used to group prose without a full Card. */
export function Framed({
  children,
  className,
  tone = "page",
}: {
  children: ReactNode;
  className?: string;
  tone?: "page" | "elevated" | "panel";
}): React.JSX.Element {
  return (
    <div
      className={clsx(
        "rounded-none border p-3",
        strokeClassName("tertiary"),
        surfaceClassName(tone),
        className,
      )}
    >
      {children}
    </div>
  );
}
