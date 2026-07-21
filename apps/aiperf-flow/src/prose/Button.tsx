/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { ButtonHTMLAttributes } from "react";
import clsx from "clsx";

export type ButtonVariant = "primary" | "secondary" | "ghost";

const VARIANT_CLASSES: Record<ButtonVariant, string> = {
  primary: "bg-accent-primary text-white border-accent-primary hover:opacity-90",
  secondary:
    "bg-surface-elevated text-ink-primary border-stroke-primary hover:bg-surface-chrome",
  ghost:
    "bg-transparent text-ink-secondary border-transparent hover:bg-surface-chrome hover:border-stroke-secondary",
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
        "rounded-none border px-4 py-2 text-sm font-semibold transition-colors",
        "disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:bg-inherit",
        VARIANT_CLASSES[variant],
        className,
      )}
      {...rest}
    />
  );
}
