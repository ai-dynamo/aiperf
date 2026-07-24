/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { NodeProps, Node } from "@xyflow/react";
import { Handle, Position } from "@xyflow/react";
import clsx from "clsx";
import { surfaceClassName, strokeClassName, inkClassName } from "../theme/tokens.js";
import type { ChipNodeData } from "./types.js";

export type ChipNodeType = Node<ChipNodeData, "chip">;

// Invisible, zero-footprint handles: present in the DOM so a chip can be an edge endpoint (a
// junction/decision node like "is_virtual()?") without React Flow's "couldn't create edge for
// handle id null" (#008) error, but with no visible connector dot so pure-label chips look unchanged.
const HIDDEN_HANDLE_CLASS_NAME = "!h-0 !w-0 !min-h-0 !min-w-0 !border-0 !bg-transparent opacity-0";

export function ChipNode({ data }: NodeProps<ChipNodeType>): React.JSX.Element {
  return (
    <div
      className={clsx(
        "max-w-[220px] rounded-md border px-3 py-1.5 text-xs font-semibold tracking-wide break-words shadow-sm",
        surfaceClassName("panel"),
        strokeClassName(data.strokeRole ?? "secondary"),
        inkClassName("secondary"),
        data.className,
      )}
    >
      <Handle type="target" position={Position.Left} className={HIDDEN_HANDLE_CLASS_NAME} />
      {data.label}
      <Handle type="source" position={Position.Right} className={HIDDEN_HANDLE_CLASS_NAME} />
    </div>
  );
}
