/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { NodeProps, Node } from "@xyflow/react";
import { surfaceClassName, strokeClassName, inkClassName } from "../theme/tokens.js";
import type { ChipNodeData } from "./types.js";

export type ChipNodeType = Node<ChipNodeData, "chip">;

export function ChipNode({ data }: NodeProps<ChipNodeType>): React.JSX.Element {
  return (
    <div
      className={`rounded-none border px-3 py-1 text-xs font-medium ${surfaceClassName("panel")} ${strokeClassName(data.strokeRole ?? "secondary")} ${inkClassName("secondary")}`}
    >
      {data.label}
    </div>
  );
}
