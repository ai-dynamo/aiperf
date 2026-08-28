/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { Edge, Node } from "@xyflow/react";
import { ReactFlow, ReactFlowProvider, Background, BackgroundVariant } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { nodeTypes } from "../../nodes/nodeTypes.js";
import { edgeTypes } from "../../edges/edgeTypes.js";
import { useElkLayout } from "../../layout/graph/index.js";
import type { ElkOptions } from "../../layout/graph/index.js";
import { Row } from "../../layout/Row.js";
import { Eyebrow } from "../../prose/Eyebrow.js";
import { inkClassName, strokeClassName } from "../../theme/tokens.js";

//! Shared layout helpers for the streaming-dynamo-shadow-replay deck pages.

/** Standard React Flow canvas with its own ReactFlowProvider (prevents multi-diagram store collisions). */
export function DeckDiagram({
  nodes,
  edges,
  height,
  layout,
}: {
  nodes: Node[];
  edges: Edge[];
  height: number;
  layout?: ElkOptions;
}): React.JSX.Element {
  return (
    <div style={{ height }}>
      <ReactFlowProvider>
        {layout !== undefined ? (
          <DeckDiagramAutoLaid nodes={nodes} edges={edges} layout={layout} />
        ) : (
          <ReactFlow
            nodeTypes={nodeTypes}
            edgeTypes={edgeTypes}
            nodes={nodes}
            edges={edges}
            fitView
            fitViewOptions={{ padding: 0.15 }}
            nodesDraggable={false}
            proOptions={{ hideAttribution: true }}
          >
            <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="var(--color-stroke-secondary)" />
          </ReactFlow>
        )}
      </ReactFlowProvider>
    </div>
  );
}

function DeckDiagramAutoLaid({
  nodes: inputNodes,
  edges,
  layout,
}: {
  nodes: Node[];
  edges: Edge[];
  layout: ElkOptions;
}): React.JSX.Element {
  const { nodes, laidOut } = useElkLayout(inputNodes, edges, layout);
  return (
    <ReactFlow
      nodeTypes={nodeTypes}
      edgeTypes={edgeTypes}
      nodes={nodes}
      edges={edges}
      fitView
      fitViewOptions={{ padding: 0.15 }}
      nodesDraggable={false}
      proOptions={{ hideAttribution: true }}
      style={{ opacity: laidOut ? 1 : 0, transition: "opacity 150ms ease" }}
    >
      <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="var(--color-stroke-secondary)" />
    </ReactFlow>
  );
}

export interface EvidenceItem {
  label: string;
  path: string;
}

export function EvidenceRow({ items }: { items: ReadonlyArray<EvidenceItem> }): React.JSX.Element {
  return (
    <div>
      <Eyebrow className="mb-2">Source anchors</Eyebrow>
      <Row gap={8} wrap>
        {items.map((item) => (
          <span
            key={item.path + item.label}
            className={`inline-flex items-center gap-2 rounded-md border px-3 py-1 text-xs shadow-sm ${strokeClassName("secondary")}`}
          >
            <span className={`font-medium ${inkClassName("secondary")}`}>{item.label}</span>
            <code className={`${inkClassName("tertiary")}`}>{item.path}</code>
          </span>
        ))}
      </Row>
    </div>
  );
}

export function PageIntro({ title, children }: { title: string; children: React.ReactNode }): React.JSX.Element {
  return (
    <div>
      <h2 className="text-lg font-semibold">{title}</h2>
      <p className={`mt-1 max-w-3xl text-sm ${inkClassName("secondary")}`}>{children}</p>
    </div>
  );
}
