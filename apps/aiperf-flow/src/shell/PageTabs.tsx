/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import clsx from "clsx";
import { inkClassName, strokeClassName } from "../theme/tokens.js";

/** One in-deck page tab: a stable id (used for selection) plus its display label. */
export interface PageTabDefinition<T extends string> {
  id: T;
  label: string;
}

/**
 * Tab-style pill row for switching between named "pages" within a single slide/deck, distinct
 * from `PresentationShell`'s slide-to-slide navigation. Ports the `Pill` tab pattern used by
 * Cursor canvases (e.g. `docs/canvases/segment-pools-and-body-plans.canvas.tsx`) into this app's
 * component library.
 */
export function PageTabs<T extends string>({
  pages,
  current,
  onChange,
  className,
}: {
  pages: ReadonlyArray<PageTabDefinition<T>>;
  current: T;
  onChange: (id: T) => void;
  className?: string;
}): React.JSX.Element {
  return (
    <div className={clsx("flex gap-2", className)}>
      {pages.map((page) => {
        const isCurrent = page.id === current;
        return (
          <button
            key={page.id}
            type="button"
            aria-pressed={isCurrent}
            onClick={() => onChange(page.id)}
            className={clsx(
              "rounded-full border px-3 py-1 text-xs font-medium transition-colors",
              isCurrent
                ? "border-accent-primary bg-accent-primary text-white"
                : clsx(strokeClassName("secondary"), inkClassName("secondary")),
            )}
          >
            {page.label}
          </button>
        );
      })}
    </div>
  );
}
