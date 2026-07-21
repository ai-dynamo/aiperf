/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import clsx from "clsx";
import { surfaceClassName, strokeClassName, inkClassName } from "../theme/tokens.js";

export type CodeProps = {
  /** Code text. Rendered verbatim — no syntax highlighting. */
  children: string;
  /** Renders a short monospace span with a subtle background tint instead of a block. */
  inline?: boolean;
  className?: string;
};

/**
 * Plain-monospace code primitive, formalizing the ad hoc `<pre>` pattern seen across
 * decks (e.g. `BodyPlanPage`). Two modes: `inline` for a short in-sentence span, and
 * block (default) for a `<pre><code>` region for longer content. No syntax highlighting
 * — flat monospace text is the established fidelity bar in this codebase.
 */
export function Code({ children, inline = false, className }: CodeProps): React.JSX.Element {
  if (inline) {
    return (
      <span
        className={clsx(
          "rounded-none border px-1.5 py-0.5 font-mono text-[0.9em]",
          surfaceClassName("panel"),
          strokeClassName("tertiary"),
          inkClassName("primary"),
          className,
        )}
      >
        {children}
      </span>
    );
  }

  return (
    <pre
      className={clsx(
        "overflow-x-auto whitespace-pre rounded-none border p-3 font-mono text-xs leading-[18px]",
        surfaceClassName("panel"),
        strokeClassName("secondary"),
        inkClassName("primary"),
        className,
      )}
    >
      <code>{children}</code>
    </pre>
  );
}
