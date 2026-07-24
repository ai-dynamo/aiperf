/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { NodeProps, Node } from "@xyflow/react";
import { Handle, Position } from "@xyflow/react";
import clsx from "clsx";
import { surfaceClassName, strokeClassName, inkClassName } from "../theme/tokens.js";
import type { CardNodeData } from "./types.js";

export type CardNodeType = Node<CardNodeData, "card">;

const HANDLE_CLASS_NAME = "!rounded-full !border-stroke-primary !bg-accent-primary !h-2 !w-2";

export function CardNode({ data }: NodeProps<CardNodeType>): React.JSX.Element {
  return (
    <div
      className={clsx(
        // Soft, floating 3D card: rounded, full (non-strip) border, and a soft elevation shadow.
        // A role's `roleClassName` (in data.className) upgrades the border + shadow to its color.
        "min-w-[180px] max-w-[260px] rounded-xl border px-4 py-3 shadow-md transition-shadow",
        surfaceClassName("elevated"),
        strokeClassName(data.strokeRole ?? "primary"),
        data.className,
      )}
    >
      <Handle type="target" position={Position.Left} className={HANDLE_CLASS_NAME} />
      <div className={`text-sm font-semibold tracking-tight break-words ${inkClassName("primary")}`}>
        {data.title}
      </div>
      {data.subtitle !== undefined && (
        <div
          className={`mt-1 text-[11px] font-medium tracking-wide uppercase break-words ${inkClassName("tertiary")}`}
        >
          {data.subtitle}
        </div>
      )}
      {data.detail !== undefined && (
        <div className={`mt-1.5 text-xs break-words ${inkClassName("secondary")}`}>{data.detail}</div>
      )}
      <Handle type="source" position={Position.Right} className={HANDLE_CLASS_NAME} />
    </div>
  );
}
