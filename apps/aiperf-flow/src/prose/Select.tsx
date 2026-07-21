/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { useId } from "react";
import clsx from "clsx";

export type SelectOption = {
  value: string;
  label: string;
};

export type SelectProps = {
  options: SelectOption[];
  value: string;
  onChange: (value: string) => void;
  label?: string;
  className?: string;
};

/** Flat, boxy native `<select>` — semantic HTML for reliable keyboard and screen-reader behavior. */
export function Select({ options, value, onChange, label, className }: SelectProps): React.JSX.Element {
  const generatedId = useId();
  // `appearance-none` strips the browser's own control chrome (which renders illegibly — a
  // light system dropdown arrow/panel — on this dark, custom-styled control in some browsers),
  // replaced by the themed chevron drawn below. The pr-8 pad keeps option text clear of it.
  const select = (
    <div className="relative">
      <select
        id={label ? generatedId : undefined}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className={clsx(
          "w-full appearance-none rounded-none border border-stroke-primary bg-surface-elevated px-3 py-2 pr-8 text-sm text-ink-primary transition-colors duration-150",
          "hover:border-stroke-secondary",
          "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-primary focus-visible:ring-offset-2 focus-visible:ring-offset-surface-page",
          className,
        )}
      >
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
      <svg
        viewBox="0 0 16 16"
        aria-hidden="true"
        className="pointer-events-none absolute right-2.5 top-1/2 h-3 w-3 -translate-y-1/2 text-ink-secondary"
      >
        <path d="M3 6 L8 11 L13 6" fill="none" stroke="currentColor" strokeWidth="1.5" />
      </svg>
    </div>
  );

  if (!label) {
    return select;
  }

  return (
    <label htmlFor={generatedId} className="flex flex-col gap-1 text-sm text-ink-secondary">
      {label}
      {select}
    </label>
  );
}
