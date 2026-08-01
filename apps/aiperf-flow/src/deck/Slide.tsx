/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { useEffect } from "react";
import { ReactFlow, ReactFlowProvider, useReactFlow, type Edge, type Node } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { nodeTypes } from "../nodes/nodeTypes.js";
import { useReveal } from "../reveal/useReveal.js";
import { inkClassName } from "../theme/tokens.js";
import type { SlideDefinition } from "./types.js";

/**
 * The canvas, inside the provider so it can reach `useReactFlow`.
 *
 * `fitView` alone only fits on mount — when the reveal cascade is still running
 * that means fitting to whichever node appeared first, leaving the rest of the
 * diagram off-screen for good. Re-fitting as the visible count grows keeps the
 * whole diagram framed while it assembles.
 */
function SlideCanvas({ nodes, edges }: { nodes: Node[]; edges: Edge[] }): React.JSX.Element {
  const { fitView } = useReactFlow();
  const visibleCount = nodes.filter((node) => node.hidden !== true).length;

  useEffect(() => {
    void fitView({ duration: 220, padding: 0.15 });
  }, [visibleCount, fitView]);

  return <ReactFlow nodeTypes={nodeTypes} nodes={nodes} edges={edges} fitView />;
}

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
          <SlideCanvas nodes={nodes} edges={edges} />
        </ReactFlowProvider>
      </div>
    </div>
  );
}
