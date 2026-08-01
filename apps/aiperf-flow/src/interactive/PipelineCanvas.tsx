/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Reusable React Flow canvas wrapper. Renders one level's nodes/edges and owns its OWN
//! `ReactFlowProvider` per `<ReactFlow>` instance — sibling `<ReactFlow>`s sharing a single
//! ancestor provider silently collide onto one internal store (only the last-mounted one renders).
//! Mirrors `decks/rust-aiperf-architecture/shared.tsx`'s `DeckDiagram`, adding an `onNodeClick`
//! id callback so a click can drive drill-down in a `ZoomStage`.
//!
//! Pass `layout` to compute node positions with the shared ELK engine instead of honoring the
//! nodes' authored `position` hints (the app-wide fix for smooshed boxes / doubled-back edges).
//! Omit it (`"off"`) to keep the legacy manual-position behavior unchanged.

import { useMemo } from "react";
import type { Edge, Node } from "@xyflow/react";
import { Background, BackgroundVariant, ReactFlow, ReactFlowProvider } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { nodeTypes } from "../nodes/nodeTypes.js";
import { autoRouteEdges } from "../nodes/anchors.js";
import { edgeTypes } from "../edges/edgeTypes.js";
import { useElkLayout } from "../layout/graph/index.js";
import type { ElkOptions } from "../layout/graph/index.js";

export interface PipelineCanvasProps {
  /** Nodes for the level being rendered. */
  nodes: Node[];
  /** Edges for the level being rendered. */
  edges: Edge[];
  /** Fixed canvas height in px (React Flow needs a sized container). Defaults to 420. */
  height?: number;
  /**
   * Tailwind height class(es) for a responsive/viewport-relative canvas (e.g. `"h-[72vh]"`).
   * Takes precedence over `height` when set — use it so the diagram grows with the screen.
   */
  heightClass?: string;
  /** Called with a node's id when it is clicked — wire this to a `ZoomStage`'s `drill`. */
  onNodeClick?: (nodeId: string) => void;
  /** `fitView` padding fraction. Defaults to 0.2. */
  fitViewPadding?: number;
  /**
   * ELK auto-layout options. When provided, node positions are computed from graph structure and
   * measured node sizes; the nodes' authored `position` fields are ignored. Default `"off"` keeps
   * the legacy manual-position behavior.
   */
  layout?: ElkOptions | "off";
  /** Merged onto the sizing wrapper `<div>`. */
  className?: string;
}

/** Inner canvas: runs the ELK layout hook (inside the provider) and renders the auto-laid-out nodes. */
function AutoLaidCanvas({
  nodes: inputNodes,
  edges,
  layout,
  onNodeClick,
  fitViewPadding,
}: {
  nodes: Node[];
  edges: Edge[];
  layout: ElkOptions;
  onNodeClick?: (nodeId: string) => void;
  fitViewPadding: number;
}): React.JSX.Element {
  const { nodes, laidOut } = useElkLayout(inputNodes, edges, layout);
  // See `AutoLayoutFlow`: anchors are chosen from post-layout coordinates.
  const routedEdges = useMemo(() => autoRouteEdges(nodes, edges), [nodes, edges]);
  return (
    <ReactFlow
      nodeTypes={nodeTypes}
      edgeTypes={edgeTypes}
      nodes={nodes}
      edges={routedEdges}
      fitView
      fitViewOptions={{ padding: fitViewPadding, maxZoom: 1 }}
      minZoom={0.5}
      nodesDraggable={false}
      onNodeClick={(_event, node) => onNodeClick?.(node.id)}
      proOptions={{ hideAttribution: true }}
      // Hide the pre-layout frame so nodes never flash at placeholder coordinates.
      style={{ opacity: laidOut ? 1 : 0, transition: "opacity 150ms ease" }}
    >
      <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="var(--color-stroke-secondary)" />
    </ReactFlow>
  );
}

/**
 * One React Flow level, each instance self-contained in its own `ReactFlowProvider`. With `layout`
 * off, nodes are static position hints (React Flow reflows via `fitView`); with `layout` set, the
 * ELK engine positions them. Nothing is draggable. Pass `onNodeClick` to turn a click into a drill.
 */
export function PipelineCanvas({
  nodes,
  edges,
  height = 420,
  heightClass,
  onNodeClick,
  fitViewPadding = 0.2,
  layout = "off",
  className,
}: PipelineCanvasProps): React.JSX.Element {
  return (
    <div
      className={heightClass ? `${heightClass} ${className ?? ""}` : className}
      style={heightClass ? undefined : { height }}
    >
      <ReactFlowProvider>
        {layout === "off" ? (
          <ReactFlow
            nodeTypes={nodeTypes}
            edgeTypes={edgeTypes}
            nodes={nodes}
            edges={autoRouteEdges(nodes, edges)}
            fitView
            fitViewOptions={{ padding: fitViewPadding }}
            nodesDraggable={false}
            onNodeClick={(_event, node) => onNodeClick?.(node.id)}
            proOptions={{ hideAttribution: true }}
          >
            <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="var(--color-stroke-secondary)" />
          </ReactFlow>
        ) : (
          <AutoLaidCanvas
            nodes={nodes}
            edges={edges}
            layout={layout}
            onNodeClick={onNodeClick}
            fitViewPadding={fitViewPadding}
          />
        )}
      </ReactFlowProvider>
    </div>
  );
}
