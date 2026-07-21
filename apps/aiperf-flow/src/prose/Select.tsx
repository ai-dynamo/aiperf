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
  const select = (
    <select
      id={label ? generatedId : undefined}
      value={value}
      onChange={(event) => onChange(event.target.value)}
      className={clsx(
        "rounded-none border border-stroke-primary bg-surface-elevated px-3 py-2 text-sm text-ink-primary",
        className,
      )}
    >
      {options.map((option) => (
        <option key={option.value} value={option.value}>
          {option.label}
        </option>
      ))}
    </select>
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
