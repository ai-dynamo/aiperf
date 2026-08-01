/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { useEffect } from "react";
import {
  Background,
  BackgroundVariant,
  ReactFlow,
  ReactFlowProvider,
  useReactFlow,
  type Edge,
  type Node,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { nodeTypes } from "../nodes/nodeTypes.js";
import { edgeTypes } from "../edges/edgeTypes.js";
import { autoRouteEdges } from "../nodes/anchors.js";
import { useReveal } from "../reveal/useReveal.js";
import clsx from "clsx";
import { Eyebrow } from "../prose/Eyebrow.js";
import { inkClassName, surfaceClassName } from "../theme/tokens.js";
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
    // maxZoom 1 keeps text at its natural size. Without it, the first reveal tick has
    // exactly one visible node and React Flow zooms it to the default maxZoom of 2 —
    // one card blown up to fill the slide, then punching back out as the rest appear.
    void fitView({ duration: 220, padding: 0.16, maxZoom: 1 });
  }, [visibleCount, fitView]);

  return (
    <ReactFlow
      nodeTypes={nodeTypes}
      edgeTypes={edgeTypes}
      nodes={nodes}
      edges={edges}
      fitView
      fitViewOptions={{ padding: 0.16, maxZoom: 1 }}
      minZoom={0.5}
      // A narrated deck is a presentation, not an editor: a stray drag would both
      // reorder the diagram and fight the re-fit effect above.
      nodesDraggable={false}
      nodesConnectable={false}
      proOptions={{ hideAttribution: true }}
    >
      <Background
        variant={BackgroundVariant.Dots}
        gap={20}
        size={1}
        color="var(--color-stroke-secondary)"
      />
    </ReactFlow>
  );
}

export function Slide({ slide }: { slide: SlideDefinition }): React.JSX.Element {
  const revealOrder = slide.revealOrder ?? slide.nodes.map((node) => node.id);
  const revealed = useReveal(revealOrder);

  const nodes = slide.nodes.map((node) => ({
    ...node,
    hidden: !revealed.has(node.id),
  }));
  // Routed from the authored positions, so an edge leaves and enters by whichever of the
  // eight anchors the geometry calls for rather than always right-to-left.
  const edges = autoRouteEdges(
    slide.nodes,
    slide.edges.map((edge) => ({
      ...edge,
      hidden: !revealed.has(edge.source) || !revealed.has(edge.target),
    })),
  );

  return (
    <div className="flex h-full flex-col">
      <div className="px-6 pt-6">
        <Eyebrow className={inkClassName("link")}>{slide.eyebrow}</Eyebrow>
        <h1 className={`mt-1 text-3xl font-extrabold ${inkClassName("primary")}`}>
          {slide.title}
        </h1>
        <p className={`mt-2 max-w-2xl text-sm ${inkClassName("secondary")}`}>{slide.lede}</p>
      </div>
      <div className={clsx("min-h-0 flex-1", surfaceClassName("page"))}>
        <ReactFlowProvider>
          <SlideCanvas nodes={nodes} edges={edges} />
        </ReactFlowProvider>
      </div>
    </div>
  );
}
