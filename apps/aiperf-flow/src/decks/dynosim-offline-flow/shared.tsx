/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared, deck-local primitives for the Dynosim Offline explainer, ported from
//! `docs/canvases/dynosim-offline-flow.canvas.tsx`. These are one-off (not promoted to
//! `src/shell`/`src/prose`) because they're specific to this canvas's "detail level" concept,
//! which nothing else in the app currently uses.

import clsx from "clsx";
import { inkClassName, strokeClassName } from "../../theme/tokens.js";

/** Detail level selectable per the source canvas: executive / developer / maintainer. */
export type Level = "executive" | "developer" | "maintainer";

const LEVEL_RANK: Record<Level, number> = { executive: 0, developer: 1, maintainer: 2 };

/** True when `level` is at least as detailed as `min`. */
export function atLeast(level: Level, min: Level): boolean {
  return LEVEL_RANK[level] >= LEVEL_RANK[min];
}

/** Generic labeled option, e.g. for {@link SegControl}. */
export type SegOption<T extends string> = { id: T; label: string };

/**
 * Reusable segmented control (source canvas's `SegControl`), used by both `SeamsPage` (execution
 * mode) and `EnginePage` (topology / router_mode).
 */
export function SegControl<T extends string>({
  value,
  onChange,
  options,
}: {
  value: T;
  onChange: (value: T) => void;
  options: ReadonlyArray<SegOption<T>>;
}): React.JSX.Element {
  return (
    <div className="flex flex-wrap gap-1.5">
      {options.map((option) => {
        const active = option.id === value;
        return (
          <button
            key={option.id}
            type="button"
            aria-pressed={active}
            onClick={() => onChange(option.id)}
            className={clsx(
              "rounded-none border px-3 py-1 text-xs font-medium transition-colors",
              active
                ? "border-accent-primary bg-accent-primary text-white"
                : clsx(strokeClassName("secondary"), inkClassName("secondary")),
            )}
          >
            {option.label}
          </button>
        );
      })}
    </div>
  );
}

const LEVEL_OPTIONS: Level[] = ["executive", "developer", "maintainer"];

/** Segmented control switching the page's detail level (source canvas's `DetailToggle`). */
export function DetailToggle({
  level,
  onChange,
}: {
  level: Level;
  onChange: (level: Level) => void;
}): React.JSX.Element {
  return (
    <div className="flex gap-1">
      {LEVEL_OPTIONS.map((option) => {
        const active = option === level;
        return (
          <button
            key={option}
            type="button"
            aria-pressed={active}
            onClick={() => onChange(option)}
            className={clsx(
              "rounded-none border px-2.5 py-1 text-xs font-medium capitalize transition-colors",
              active
                ? "border-accent-primary bg-accent-primary text-white"
                : clsx(strokeClassName("secondary"), inkClassName("secondary")),
            )}
          >
            {option}
          </button>
        );
      })}
    </div>
  );
}
