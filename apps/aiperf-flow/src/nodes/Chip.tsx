/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { NodeProps, Node } from "@xyflow/react";
import clsx from "clsx";
import { surfaceClassName, strokeClassName, inkClassName } from "../theme/tokens.js";
import type { ChipNodeData } from "./types.js";

export type ChipNodeType = Node<ChipNodeData, "chip">;

export function ChipNode({ data }: NodeProps<ChipNodeType>): React.JSX.Element {
  return (
    <div
      className={clsx(
        "rounded-md border px-3 py-1.5 text-xs font-semibold tracking-wide shadow-sm",
        surfaceClassName("panel"),
        strokeClassName(data.strokeRole ?? "secondary"),
        inkClassName("secondary"),
        data.className,
      )}
    >
      {data.label}
    </div>
  );
}
