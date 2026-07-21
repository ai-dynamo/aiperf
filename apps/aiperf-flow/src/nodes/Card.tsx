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

const HANDLE_CLASS_NAME = "!rounded-none !border-stroke-primary !bg-accent-primary !h-2 !w-2";

export function CardNode({ data }: NodeProps<CardNodeType>): React.JSX.Element {
  return (
    <div
      className={clsx(
        "min-w-[180px] rounded-none border border-l-2 px-4 py-3",
        surfaceClassName("elevated"),
        strokeClassName(data.strokeRole ?? "primary"),
        "border-l-accent-primary",
        data.className,
      )}
    >
      <Handle type="target" position={Position.Left} className={HANDLE_CLASS_NAME} />
      <div className={`text-sm font-semibold tracking-tight ${inkClassName("primary")}`}>
        {data.title}
      </div>
      {data.subtitle !== undefined && (
        <div
          className={`mt-1 text-[11px] font-medium tracking-wide uppercase ${inkClassName("tertiary")}`}
        >
          {data.subtitle}
        </div>
      )}
      {data.detail !== undefined && (
        <div className={`mt-1.5 text-xs ${inkClassName("secondary")}`}>{data.detail}</div>
      )}
      <Handle type="source" position={Position.Right} className={HANDLE_CLASS_NAME} />
    </div>
  );
}
