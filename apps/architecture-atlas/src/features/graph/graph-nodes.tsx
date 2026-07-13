// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Handle, Position, type Node, type NodeProps } from "@xyflow/react";

import type { AudienceLevel, GraphNode } from "../../domain/architecture";
import type {
  GraphFlavorClass,
  GraphNodePortView,
  GraphPathState,
} from "./types";

const flavorColors: Record<GraphFlavorClass, string> = {
  "compare-only": "#A78BFA",
  "primary-only": "#45C7F4",
  shared: "#76B900",
};

export interface RuntimeGraphNodeData extends Record<string, unknown> {
  audience: AudienceLevel;
  flavorClass: GraphFlavorClass;
  node: GraphNode;
  onSelect(nodeId: string): void;
  pathState: GraphPathState;
  ports: GraphNodePortView[];
}

export function RuntimeGraphNode({
  data,
}: NodeProps<Node<RuntimeGraphNodeData>>) {
  return (
    <article
      data-owner={data.node.owner}
      data-flavor-class={data.flavorClass}
      data-implementation-state={data.node.status.state}
      data-path-state={data.pathState}
      data-testid={`graph-node-${data.node.id}`}
      style={{
        border: `${data.pathState === "focused" ? 3 : 2}px ${
          data.node.status.state === "planned" ? "dashed" : "solid"
        } ${data.node.status.state === "planned" ? "#FF7A7A" : flavorColors[data.flavorClass]}`,
        opacity: data.pathState === "default" ? 0.82 : 1,
      }}
    >
      <Handle position={Position.Left} type="target" />
      <button
        aria-label={data.node.title[data.audience]}
        data-graph-entity-id={data.node.id}
        data-graph-entity-trigger="true"
        onClick={() => data.onSelect(data.node.id)}
        type="button"
      >
        <strong>{data.node.title[data.audience]}</strong>
        <p>{data.node.summary[data.audience]}</p>
        <p>
          owner: {data.node.owner} | tier: {data.node.tier} | status:{" "}
          {data.node.status.state}
        </p>
        <p>
          flavor: {data.flavorClass} | path: {data.pathState}
        </p>
      </button>
      <ul aria-label={`${data.node.title[data.audience]} seam ports`}>
        {data.ports.map((port) => (
          <li key={port.id}>
            {port.name} - {port.channel} - {port.direction}
          </li>
        ))}
      </ul>
      <Handle position={Position.Right} type="source" />
    </article>
  );
}
