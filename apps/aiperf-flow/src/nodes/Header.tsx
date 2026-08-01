/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { NodeProps, Node } from "@xyflow/react";
import clsx from "clsx";
import { inkClassName } from "../theme/tokens.js";
import type { HeaderNodeData } from "./types.js";

export type HeaderNodeType = Node<HeaderNodeData, "header">;

/**
 * Band/lane label — the caption over a row of cards, not a node in its own right.
 *
 * Deliberately unboxed: an accent tick, the label, and a hairline rule that fades out
 * across the band. The filled, shadowed card it used to be competed with the real cards
 * for attention and, being opaque, painted over any edge routed beneath it — React Flow
 * draws nodes above edges, so a band label sitting between two rows hid the connector
 * running between them.
 */
export function HeaderNode({ data }: NodeProps<HeaderNodeType>): React.JSX.Element {
  return (
    <div
      className={clsx(
        // Never intercepts a click meant for the canvas beneath it.
        "min-w-[280px] max-w-[420px] select-none pointer-events-none",
        data.className,
      )}
    >
      <div className="flex items-center gap-2.5">
        <span
          aria-hidden="true"
          className="h-3.5 w-[3px] shrink-0 rounded-full bg-[color:var(--accent,var(--color-accent-primary))]"
        />
        <span
          className={clsx(
            "shrink-0 text-[11px] font-bold uppercase tracking-[0.18em]",
            inkClassName("secondary"),
          )}
        >
          {data.title}
        </span>
        <span
          aria-hidden="true"
          className={clsx(
            "h-px flex-1 bg-gradient-to-r to-transparent",
            "from-[color:var(--color-stroke-secondary)]",
          )}
        />
      </div>
      {data.caption !== undefined && (
        <div
          className={clsx("mt-1 ml-[13px] text-[11px] tracking-wide break-words", inkClassName("tertiary"))}
        >
          {data.caption}
        </div>
      )}
    </div>
  );
}
