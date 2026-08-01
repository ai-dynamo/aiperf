/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { NodeProps, Node } from "@xyflow/react";
import clsx from "clsx";
import { surfaceClassName, inkClassName, categoryBgClassName } from "../theme/tokens.js";
import { NodeAnchorHandles } from "./anchors.js";
import { CELL_H, CELL_W, CELL_GAP } from "./blocksLayout.js";
import type { BlocksNodeData } from "./types.js";

export type BlocksNodeType = Node<BlocksNodeData, "blocks">;

/**
 * Stacked strips of per-block tags — the shape of a shared prefix and where two paths diverge.
 *
 * A prefix is only reusable if both paths agree on every block's tag, so the claim being made is
 * a cell-by-cell comparison between two strips. Cells away from the highlight are dimmed, and the
 * highlighted index is outlined on every strip so the eye lands on the same column in each.
 */
export function BlocksNode({ data }: NodeProps<BlocksNodeType>): React.JSX.Element {
  const { strips, title, highlight } = data;

  return (
    <div
      className={clsx(
        "rounded-[13px] border border-white/10 px-4 py-3.5",
        "shadow-[0_12px_28px_rgba(0,0,0,0.28)]",
        surfaceClassName(data.surfaceRole ?? "elevated"),
        data.className,
      )}
    >
      {title !== undefined && (
        <div className={`mb-2 text-sm font-semibold tracking-tight ${inkClassName("primary")}`}>
          {title}
        </div>
      )}
      <div className="flex flex-col gap-2.5">
        {strips.map((strip) => (
          <div key={strip.label} className="flex flex-col gap-1">
            <span className={`text-xs ${inkClassName("tertiary")}`}>{strip.label}</span>
            <div className="flex" style={{ gap: CELL_GAP }}>
              {strip.cells.map((role, i) => (
                <span
                  key={i}
                  className={clsx("inline-block", categoryBgClassName(role))}
                  style={{
                    width: CELL_W,
                    height: CELL_H,
                    opacity: i === highlight ? 1 : 0.55,
                    outline:
                      i === highlight ? "2px solid var(--color-category-orange)" : undefined,
                  }}
                  // The strips are the content; a per-cell label would be read out 23 times.
                  aria-hidden="true"
                />
              ))}
            </div>
          </div>
        ))}
      </div>
      {data.detail !== undefined && (
        <div className={`mt-2 max-w-[420px] text-xs ${inkClassName("secondary")}`}>
          {data.detail}
        </div>
      )}
      <NodeAnchorHandles />
    </div>
  );
}
