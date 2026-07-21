/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { ButtonHTMLAttributes } from "react";
import clsx from "clsx";

export type ButtonVariant = "primary" | "secondary" | "ghost";

const VARIANT_CLASSES: Record<ButtonVariant, string> = {
  primary:
    "bg-accent-primary text-white border-accent-primary hover:opacity-90 active:opacity-80",
  secondary:
    "bg-surface-elevated text-ink-primary border-stroke-primary hover:bg-surface-chrome hover:border-stroke-secondary active:bg-surface-panel",
  ghost:
    "bg-transparent text-ink-secondary border-transparent hover:bg-surface-chrome hover:border-stroke-secondary hover:text-ink-primary active:bg-surface-elevated",
};

export type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: ButtonVariant;
};

/** Flat, boxy action button — the shared control surface for every deck's interactive buttons. */
export function Button({
  variant = "secondary",
  className,
  type = "button",
  ...rest
}: ButtonProps): React.JSX.Element {
  return (
    <button
      type={type}
      className={clsx(
        "rounded-md border px-4 py-2 text-sm font-semibold tracking-wide shadow-sm transition-colors duration-150 hover:shadow-md",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-primary focus-visible:ring-offset-2 focus-visible:ring-offset-surface-page",
        "disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:bg-inherit disabled:active:bg-inherit",
        VARIANT_CLASSES[variant],
        className,
      )}
      {...rest}
    />
  );
}
