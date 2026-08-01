/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { NodeProps, Node } from "@xyflow/react";
import clsx from "clsx";
import { surfaceClassName, inkClassName } from "../theme/tokens.js";
import { NodeAnchorHandles } from "./anchors.js";
import type { PanelNodeData } from "./types.js";

export type PanelNodeType = Node<PanelNodeData, "panel">;

const ACCENT = "var(--accent, var(--color-accent-primary))";

/** Systems Chalk panel: neutral softly-shadowed card, color carried by an accent badge dot + hover
 * border (see `CardNode`). A role sets `--accent` via `data.className`. */
export function PanelNode({ data }: NodeProps<PanelNodeType>): React.JSX.Element {
  return (
    <div
      className={clsx(
        "min-w-[160px] max-w-[264px] rounded-[13px] border border-white/10 px-4 py-3.5",
        "shadow-[0_12px_28px_rgba(0,0,0,0.28)] transition-[transform,border-color]",
        "hover:-translate-y-0.5 hover:border-[color:var(--accent,var(--color-accent-primary))]",
        surfaceClassName(data.surfaceRole ?? "elevated"),
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
      {data.diagram !== undefined && data.diagram}
      {data.detail !== undefined && (
        <div className={`mt-1.5 text-xs break-words ${inkClassName("secondary")}`}>{data.detail}</div>
      )}
      <NodeAnchorHandles />
    </div>
  );
}
