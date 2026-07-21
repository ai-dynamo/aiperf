/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState, type ReactNode } from "react";
import clsx from "clsx";
import { inkClassName, strokeClassName } from "../theme/tokens.js";

export type CollapsibleSectionProps = {
  /** Header text, always visible; click toggles the body below it. */
  title: string;
  /** Body content, shown only while the section is open. */
  children: ReactNode;
  /** Whether the section starts open. Defaults to `false`. */
  defaultOpen?: boolean;
  className?: string;
};

/** Borderless disclosure row: a clickable header that shows/hides its body content. */
export function CollapsibleSection({
  title,
  children,
  defaultOpen = false,
  className,
}: CollapsibleSectionProps): React.JSX.Element {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <div className={clsx("rounded-lg", className)}>
      <button
        type="button"
        aria-expanded={open}
        onClick={() => setOpen((prev) => !prev)}
        className={clsx(
          "flex w-full items-center gap-2 rounded-md px-1 py-1.5 text-left text-sm font-semibold transition-colors duration-150",
          "hover:bg-surface-chrome",
          "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-primary focus-visible:ring-offset-2 focus-visible:ring-offset-surface-page",
          inkClassName("primary"),
        )}
      >
        <svg
          viewBox="0 0 16 16"
          aria-hidden="true"
          className={clsx("h-3 w-3 shrink-0 transition-transform duration-150", open && "rotate-90")}
        >
          <path d="M5 2 L11 8 L5 14" fill="none" stroke="currentColor" strokeWidth="2" />
        </svg>
        {title}
      </button>
      {open && (
        <div className={clsx("border-t px-0 py-2 text-sm", strokeClassName("secondary"), inkClassName("secondary"))}>
          {children}
        </div>
      )}
    </div>
  );
}
