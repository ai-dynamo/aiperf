/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Reusable React Flow canvas wrapper. Renders one level's nodes/edges and owns its OWN
//! `ReactFlowProvider` per `<ReactFlow>` instance — sibling `<ReactFlow>`s sharing a single
//! ancestor provider silently collide onto one internal store (only the last-mounted one renders).
//! Mirrors `decks/rust-aiperf-architecture/shared.tsx`'s `DeckDiagram`, adding an `onNodeClick`
//! id callback so a click can drive drill-down in a `ZoomStage`.

import type { Edge, Node } from "@xyflow/react";
import { Background, BackgroundVariant, ReactFlow, ReactFlowProvider } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { nodeTypes } from "../nodes/nodeTypes.js";
import { edgeTypes } from "../edges/edgeTypes.js";

export interface PipelineCanvasProps {
  /** Nodes for the level being rendered. */
  nodes: Node[];
  /** Edges for the level being rendered. */
  edges: Edge[];
  /** Fixed canvas height in px (React Flow needs a sized container). Defaults to 420. */
  height?: number;
  /** Called with a node's id when it is clicked — wire this to a `ZoomStage`'s `drill`. */
  onNodeClick?: (nodeId: string) => void;
  /** `fitView` padding fraction. Defaults to 0.2. */
  fitViewPadding?: number;
  /** Merged onto the sizing wrapper `<div>`. */
  className?: string;
}

/**
 * One React Flow level, each instance self-contained in its own `ReactFlowProvider`. Nodes are
 * static position hints (React Flow reflows via `fitView`); nothing is draggable. Pass
 * `onNodeClick` to turn a node click into a drill-down.
 */
export function PipelineCanvas({
  nodes,
  edges,
  height = 420,
  onNodeClick,
  fitViewPadding = 0.2,
  className,
}: PipelineCanvasProps): React.JSX.Element {
  return (
    <div className={className} style={{ height }}>
      <ReactFlowProvider>
        <ReactFlow
          nodeTypes={nodeTypes}
          edgeTypes={edgeTypes}
          nodes={nodes}
          edges={edges}
          fitView
          fitViewOptions={{ padding: fitViewPadding }}
          nodesDraggable={false}
          onNodeClick={(_event, node) => onNodeClick?.(node.id)}
          proOptions={{ hideAttribution: true }}
        >
          <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="var(--color-stroke-secondary)" />
        </ReactFlow>
      </ReactFlowProvider>
    </div>
  );
}
