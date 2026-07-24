/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Reusable segmented control for swapping a diagram between named variants, built on the shared
//! `Pill` (no new chip/badge type). Generic over a variant string union `V`. The selected option
//! renders with `Pill`'s accent-active styling; unselected options may carry an optional category
//! tone. Used here for the Clock-mode and Transport selectors, reusable anywhere.

import { Pill } from "../prose/Pill.js";
import { Eyebrow } from "../prose/Eyebrow.js";
import { Row } from "../layout/Row.js";
import type { CategoryRole } from "../theme/tokens.js";

/** One selectable option in a {@link SeamToggle}. */
export interface SeamToggleOption<V extends string> {
  /** The variant value reported to `onChange` when this option is chosen. */
  value: V;
  /** Visible label. */
  label: string;
  /** Optional category tone for the option while it is *not* selected. */
  tone?: CategoryRole;
  /** Optional accessible label when `label` alone doesn't convey the meaning. */
  ariaLabel?: string;
}

export interface SeamToggleProps<V extends string> {
  /** Optional `Eyebrow` label rendered before the segments. */
  label?: string;
  /** The selectable options, in display order. */
  options: ReadonlyArray<SeamToggleOption<V>>;
  /** Currently selected value. */
  value: V;
  /** Called with the new value when a segment is clicked. */
  onChange: (value: V) => void;
  /** Accessible label for the group (defaults to `label`). */
  ariaLabel?: string;
  className?: string;
}

/**
 * Segmented single-select. Each segment is a `Pill`; the active one uses `Pill`'s `active` accent
 * styling (its `tone` is intentionally suppressed so the selection is unambiguous), and inactive
 * segments fall back to their optional `tone` tint or the neutral pill.
 */
export function SeamToggle<V extends string>({
  label,
  options,
  value,
  onChange,
  ariaLabel,
  className,
}: SeamToggleProps<V>): React.JSX.Element {
  return (
    <div role="group" aria-label={ariaLabel ?? label} className={className}>
      <Row gap={8} align="center" wrap>
        {label !== undefined && <Eyebrow>{label}</Eyebrow>}
        {options.map((option) => {
          const selected = option.value === value;
          return (
            <Pill
              key={option.value}
              active={selected}
              tone={selected ? undefined : option.tone}
              onClick={() => onChange(option.value)}
              ariaLabel={option.ariaLabel}
            >
              {option.label}
            </Pill>
          );
        })}
      </Row>
    </div>
  );
}
