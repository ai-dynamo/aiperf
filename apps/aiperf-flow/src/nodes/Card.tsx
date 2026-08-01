/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { NodeProps, Node } from "@xyflow/react";
import clsx from "clsx";
import { surfaceClassName, inkClassName } from "../theme/tokens.js";
import { NodeAnchorHandles } from "./anchors.js";
import type { CardNodeData } from "./types.js";

export type CardNodeType = Node<CardNodeData, "card">;

// The role/accent color, resolved from the `--accent` CSS var a role sets (see `roleClassName`),
// falling back to the primary cyan chalk accent for un-roled cards.
const ACCENT = "var(--accent, var(--color-accent-primary))";

/**
 * Systems Chalk card: a neutral, softly-shadowed dark card whose color is carried as an ACCENT — a
 * small badge dot beside the title, an accent-colored subtitle, and an accent hover border — not a
 * fill tint or a left strip. A role sets `--accent` via `data.className` (`roleClassName`).
 */
export function CardNode({ data }: NodeProps<CardNodeType>): React.JSX.Element {
  return (
    <div
      className={clsx(
        "min-w-[190px] max-w-[264px] rounded-[13px] border border-white/10 px-4 py-3.5",
        "shadow-[0_12px_28px_rgba(0,0,0,0.28)] transition-[transform,border-color]",
        "hover:-translate-y-0.5 hover:border-[color:var(--accent,var(--color-accent-primary))]",
        surfaceClassName("elevated"),
        data.className,
      )}
    >
      <div className="flex items-center gap-2.5">
        <span
          className="h-2.5 w-2.5 shrink-0 rounded-[3px]"
          style={{ backgroundColor: ACCENT }}
          aria-hidden="true"
        />
        <div className={`text-sm font-semibold tracking-tight break-words ${inkClassName("primary")}`}>
          {data.title}
        </div>
      </div>
      {data.subtitle !== undefined && (
        <div
          className="mt-1.5 ml-[22px] text-[11px] font-semibold tracking-wide uppercase break-words"
          style={{ color: ACCENT }}
        >
          {data.subtitle}
        </div>
      )}
      {data.diagram !== undefined && data.diagram}
      {data.detail !== undefined && (
        <div className={`mt-1.5 text-xs break-words ${inkClassName("secondary")}`}>{data.detail}</div>
      )}
      <NodeAnchorHandles />
    </div>
  );
}
