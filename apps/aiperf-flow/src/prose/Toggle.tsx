/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import clsx from "clsx";

export type ToggleProps = {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label?: string;
  className?: string;
};

/** Flat, boxy boolean switch — a real button[role=switch] for consistent cross-browser styling. */
export function Toggle({ checked, onChange, label, className }: ToggleProps): React.JSX.Element {
  return (
    <span className="inline-flex items-center gap-2">
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        onClick={() => onChange(!checked)}
        className={clsx(
          "relative h-5 w-9 shrink-0 rounded-none border border-stroke-primary transition-colors duration-150",
          "hover:border-stroke-secondary",
          "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-primary focus-visible:ring-offset-2 focus-visible:ring-offset-surface-page",
          checked ? "bg-accent-primary hover:opacity-90" : "bg-surface-elevated hover:bg-surface-chrome",
          className,
        )}
      >
        <span
          className={clsx(
            "absolute top-0.5 h-3.5 w-3.5 bg-white transition-transform duration-150",
            checked ? "translate-x-4" : "translate-x-0.5",
          )}
        />
      </button>
      {label ? <span className="text-sm text-ink-secondary">{label}</span> : null}
    </span>
  );
}
