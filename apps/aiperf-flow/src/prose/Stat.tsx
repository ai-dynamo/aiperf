/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! `Stat`: a big-number-plus-label KPI/metric summary tile, e.g. "1,284 req/s" with a
//! "Throughput" label and an optional trend indicator like "+8.2%".

import clsx from "clsx";
import { accentClassName, categoryClassName, inkClassName, surfaceClassName, strokeClassName } from "../theme/tokens.js";

export type StatTone = "neutral" | "positive" | "negative";

export type StatProps = {
  /** Label above the value, e.g. "Throughput". */
  label: string;
  /** The primary metric value, e.g. "1,284 req/s" or a bare number. */
  value: string | number;
  /** Optional delta indicator rendered beside the value, e.g. "+8.2%". */
  trend?: string;
  /** Semantic direction for the trend color. Defaults to `"neutral"`. */
  tone?: StatTone;
  /** Extra classes merged onto the component's own root classes, appended last. */
  className?: string;
};

const toneClassName: Record<StatTone, string> = {
  neutral: inkClassName("secondary"),
  positive: accentClassName("primary"),
  negative: categoryClassName("red"),
};

/**
 * Single metric display — a large value with a compact uppercase label and an
 * optional tone-colored trend chip.
 *
 * @example
 * ```tsx
 * <Stat label="Throughput" value="1,284 req/s" trend="+8.2%" tone="positive" />
 * ```
 */
export function Stat({ label, value, trend, tone = "neutral", className }: StatProps): React.JSX.Element {
  return (
    <div
      className={clsx(
        "rounded-none border px-4 py-3",
        surfaceClassName("elevated"),
        strokeClassName("secondary"),
        className,
      )}
    >
      <div className={`text-[11px] font-bold uppercase tracking-wide ${inkClassName("secondary")}`}>
        {label}
      </div>
      <div className="mt-1 flex items-baseline gap-2">
        <span className={`text-2xl font-bold ${inkClassName("primary")}`}>{value}</span>
        {trend !== undefined && (
          <span className={`text-xs font-semibold ${toneClassName[tone]}`}>{trend}</span>
        )}
      </div>
    </div>
  );
}
