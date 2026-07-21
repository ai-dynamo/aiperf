/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { ReactFlow, ReactFlowProvider } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { nodeTypes } from "../nodes/nodeTypes.js";
import { useReveal } from "../reveal/useReveal.js";
import { inkClassName } from "../theme/tokens.js";
import type { SlideDefinition } from "./types.js";

export function Slide({ slide }: { slide: SlideDefinition }): React.JSX.Element {
  const revealOrder = slide.revealOrder ?? slide.nodes.map((node) => node.id);
  const revealed = useReveal(revealOrder);

  const nodes = slide.nodes.map((node) => ({
    ...node,
    hidden: !revealed.has(node.id),
  }));
  const edges = slide.edges.map((edge) => ({
    ...edge,
    hidden: !revealed.has(edge.source) || !revealed.has(edge.target),
  }));

  return (
    <div className="flex h-full flex-col">
      <div className="px-6 pt-6">
        <div className={`text-xs font-bold tracking-wide ${inkClassName("link")}`}>
          {slide.eyebrow}
        </div>
        <h1 className={`mt-1 text-3xl font-extrabold ${inkClassName("primary")}`}>
          {slide.title}
        </h1>
        <p className={`mt-2 max-w-2xl text-sm ${inkClassName("secondary")}`}>{slide.lede}</p>
      </div>
      <div className="min-h-0 flex-1">
        <ReactFlowProvider>
          <ReactFlow nodeTypes={nodeTypes} nodes={nodes} edges={edges} fitView />
        </ReactFlowProvider>
      </div>
    </div>
  );
}
