// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Handle, Position, type Node, type NodeProps } from "@xyflow/react";

import type { AudienceLevel, GraphNode } from "../../domain/architecture";
import type { GraphNodePortView, GraphPathState } from "./types";

export interface RuntimeGraphNodeData extends Record<string, unknown> {
  audience: AudienceLevel;
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
      data-path-state={data.pathState}
      data-testid={`graph-node-${data.node.id}`}
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
