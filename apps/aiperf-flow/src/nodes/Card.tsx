/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { NodeProps, Node } from "@xyflow/react";
import { Handle, Position } from "@xyflow/react";
import { surfaceClassName, strokeClassName, inkClassName } from "../theme/tokens.js";
import type { CardNodeData } from "./types.js";

export type CardNodeType = Node<CardNodeData, "card">;

export function CardNode({ data }: NodeProps<CardNodeType>): React.JSX.Element {
  return (
    <div
      className={`min-w-[180px] rounded-none border px-4 py-3 ${surfaceClassName("elevated")} ${strokeClassName(data.strokeRole ?? "primary")}`}
    >
      <Handle type="target" position={Position.Left} />
      <div className={`text-sm font-semibold ${inkClassName("primary")}`}>{data.title}</div>
      {data.subtitle !== undefined && (
        <div className={`mt-1 text-[11px] ${inkClassName("tertiary")}`}>{data.subtitle}</div>
      )}
      {data.detail !== undefined && (
        <div className={`mt-1 text-xs ${inkClassName("secondary")}`}>{data.detail}</div>
      )}
      <Handle type="source" position={Position.Right} />
    </div>
  );
}
