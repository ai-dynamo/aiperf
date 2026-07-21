/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { ReactNode } from "react";
import { accentClassName, inkClassName, strokeClassName } from "../theme/tokens.js";

/** Small bar-chart brand mark, matching `apps/explainers`'s wordmark. */
function BrandMark(): React.JSX.Element {
  return (
    <span aria-hidden="true" className="flex items-end gap-[3px]">
      <span className="h-3 w-[5px] rounded-t-sm bg-accent-primary" />
      <span className="h-5 w-[5px] rounded-t-sm bg-accent-primary" />
      <span className="h-4 w-[5px] rounded-t-sm bg-accent-primary" />
    </span>
  );
}

/**
 * App-level chrome bar: brand mark, "AIPERF · <section>" breadcrumb, and an optional
 * right-aligned actions slot. Sits above any page-local navigation (e.g. `PageTabs`).
 * Carries a thin accent-colored keyline along its bottom edge, deliberately restrained
 * (no gradient, no glow) to read as a product/keynote header rather than a generic tool bar.
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
      className={`relative flex items-center justify-between border-b bg-surface-page px-8 py-4 shadow-sm ${strokeClassName("secondary")}`}
    >
      <div className="flex items-center gap-4">
        <a href="/" className="flex items-center gap-3">
          <BrandMark />
          <span className={`text-base font-extrabold tracking-[0.18em] ${inkClassName("primary")}`}>
            AIPERF
          </span>
        </a>
        <span className={`text-sm ${accentClassName("primary")}`}>·</span>
        <span className={`text-sm font-semibold uppercase tracking-[0.1em] ${inkClassName("secondary")}`}>
          {section}
        </span>
      </div>
      {actions !== undefined && <div className="flex items-center gap-2">{actions}</div>}
      <span aria-hidden="true" className="absolute inset-x-0 bottom-0 h-[2px] bg-accent-primary/70" />
    </header>
  );
}
