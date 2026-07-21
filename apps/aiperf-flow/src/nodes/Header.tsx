/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { NodeProps, Node } from "@xyflow/react";
import clsx from "clsx";
import { surfaceClassName, strokeClassName, inkClassName } from "../theme/tokens.js";
import type { HeaderNodeData } from "./types.js";

export type HeaderNodeType = Node<HeaderNodeData, "header">;

export function HeaderNode({ data }: NodeProps<HeaderNodeType>): React.JSX.Element {
  return (
    <div
      className={clsx(
        "min-w-[280px] rounded-none border-b px-4 py-3",
        surfaceClassName(data.surfaceRole ?? "chrome"),
        strokeClassName("tertiary"),
        data.className,
      )}
    >
      <div className={`text-xs font-bold tracking-widest ${inkClassName("secondary")}`}>
        {data.title}
      </div>
      {data.caption !== undefined && (
        <div className={`mt-1 text-[11px] tracking-wide ${inkClassName("tertiary")}`}>
          {data.caption}
        </div>
      )}
    </div>
  );
}
