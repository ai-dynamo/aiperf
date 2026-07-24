/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Drop-in auto-laid-out `<ReactFlow>` for the decks that embed a raw `<ReactFlow>` rather than
//! going through `PipelineCanvas`. Owns its own `ReactFlowProvider` (per the one-provider-per-
//! instance trap) and runs the ELK measure→layout→apply cycle internally, so a migrating deck
//! deletes its hand-picked node positions and just declares nodes + edges. Extra chrome
//! (Background/Controls/MiniMap) can be passed as `children`; a dotted background is the default.

import { Background, BackgroundVariant, ReactFlow, ReactFlowProvider } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import type { Edge, Node } from "@xyflow/react";
import { nodeTypes } from "../../nodes/nodeTypes.js";
import { edgeTypes } from "../../edges/edgeTypes.js";
import { useElkLayout } from "./useElkLayout.js";
import type { ElkOptions } from "./elkEngine.js";

export interface AutoLayoutFlowProps {
  nodes: Node[];
  edges: Edge[];
  /** ELK layout options; direction defaults to `RIGHT`. */
  layout?: ElkOptions;
  /** Fixed canvas height in px (React Flow needs a sized container). Defaults to 420. */
  height?: number;
  onNodeClick?: (nodeId: string) => void;
  /** Extra React Flow children (Background/Controls/MiniMap). A dotted Background is used if omitted. */
  children?: React.ReactNode;
  className?: string;
}

function AutoLayoutInner({
  nodes: inputNodes,
  edges,
  layout,
  onNodeClick,
  children,
}: Pick<AutoLayoutFlowProps, "nodes" | "edges" | "layout" | "onNodeClick" | "children">): React.JSX.Element {
  const { nodes, laidOut } = useElkLayout(inputNodes, edges, layout ?? {});
  return (
    <ReactFlow
      nodeTypes={nodeTypes}
      edgeTypes={edgeTypes}
      nodes={nodes}
      edges={edges}
      fitView
      fitViewOptions={{ padding: 0.16, maxZoom: 1 }}
      minZoom={0.5}
      nodesDraggable={false}
      onNodeClick={(_event, node) => onNodeClick?.(node.id)}
      proOptions={{ hideAttribution: true }}
      // Hide the pre-layout frame so nodes never flash at placeholder coordinates.
      style={{ opacity: laidOut ? 1 : 0, transition: "opacity 150ms ease" }}
    >
      {children ?? (
        <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="var(--color-stroke-secondary)" />
      )}
    </ReactFlow>
  );
}

/** Auto-laid-out React Flow canvas, self-contained in its own `ReactFlowProvider`. */
export function AutoLayoutFlow({
  height = 420,
  className,
  ...inner
}: AutoLayoutFlowProps): React.JSX.Element {
  return (
    <div className={className} style={{ height }}>
      <ReactFlowProvider>
        <AutoLayoutInner {...inner} />
      </ReactFlowProvider>
    </div>
  );
}
