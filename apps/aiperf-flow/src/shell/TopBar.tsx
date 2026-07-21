/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { ReactNode } from "react";
import { inkClassName, strokeClassName } from "../theme/tokens.js";

/** Small bar-chart brand mark, matching `apps/explainers`'s wordmark. */
function BrandMark(): React.JSX.Element {
  return (
    <span aria-hidden="true" className="flex items-end gap-[2px]">
      <span className="h-2.5 w-1 bg-accent-primary" />
      <span className="h-4 w-1 bg-accent-primary" />
      <span className="h-3 w-1 bg-accent-primary" />
    </span>
  );
}

/**
 * App-level chrome bar: brand mark, "AIPERF · <section>" breadcrumb, and an optional
 * right-aligned actions slot. Sits above any page-local navigation (e.g. `PageTabs`).
 */
export function TopBar({
  section,
  actions,
}: {
  section: string;
  actions?: ReactNode;
}): React.JSX.Element {
  return (
    <header
      className={`flex items-center justify-between border-b bg-surface-page px-8 py-3 ${strokeClassName("secondary")}`}
    >
      <div className="flex items-center gap-3">
        <BrandMark />
        <span className={`text-sm font-bold tracking-wide ${inkClassName("primary")}`}>
          AIPERF
        </span>
        <span className={`text-sm ${inkClassName("tertiary")}`}>·</span>
        <span className={`text-sm font-medium uppercase tracking-wide ${inkClassName("secondary")}`}>
          {section}
        </span>
      </div>
      {actions !== undefined && <div className="flex items-center gap-2">{actions}</div>}
    </header>
  );
}
