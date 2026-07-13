// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { Handle, Position, type Node, type NodeProps } from "@xyflow/react";

import type { AudienceLevel, GraphNode } from "../../domain/architecture";
import type {
  GraphFlavorClass,
  GraphNodePortView,
  GraphPathState,
  GraphPulseState,
  GraphRelayoutState,
  GraphTraceMode,
} from "./types";

export interface RuntimeGraphNodeData extends Record<string, unknown> {
  audience: AudienceLevel;
  expanded: boolean;
  flavorClass: GraphFlavorClass;
  node: GraphNode;
  onCollapse?(nodeId: string): void;
  onExpand?(nodeId: string): void;
  onSelect(nodeId: string): void;
  onTraceModeChange?(nodeId: string, mode: GraphTraceMode): void;
  pathState: GraphPathState;
  ports: GraphNodePortView[];
  pulseState: GraphPulseState;
  relayoutState: GraphRelayoutState;
  traceMode: GraphTraceMode;
}

export function RuntimeGraphNode({
  data,
}: NodeProps<Node<RuntimeGraphNodeData>>) {
  const classes = [
    "graph-node",
    `graph-node-tier-${data.node.tier}`,
    `graph-node-flavor-${data.flavorClass}`,
    `graph-node-path-${data.pathState}`,
    `graph-node-${data.node.status.state}`,
    `graph-node-pulse-${data.pulseState}`,
    `graph-node-relayout-${data.relayoutState}`,
  ].join(" ");
  const title = data.node.title[data.audience];
  const detailsVisible = data.expanded || data.audience === "maintainer";
  const badgeFlavor = data.flavorClass.replace("-", " ");

  return (
    <article
      className={classes}
      data-audience={data.audience}
      data-details-visible={detailsVisible || undefined}
      data-expanded={data.expanded || undefined}
      data-owner={data.node.owner}
      data-flavor-class={data.flavorClass}
      data-implementation-state={data.node.status.state}
      data-path-state={data.pathState}
      data-pulse-state={data.pulseState}
      data-relayout-state={data.relayoutState}
      data-tier={data.node.tier}
      data-trace-mode={data.traceMode}
      data-testid={`graph-node-${data.node.id}`}
      style={{
        opacity:
          data.traceMode !== "none" && data.pathState === "default" ? 0.35 : 1,
      }}
    >
      <Handle position={Position.Left} type="target" />
      <span
        aria-hidden="true"
        className="graph-node-drag-handle"
        data-testid={`graph-node-drag-handle-${data.node.id}`}
      >
        drag
      </span>
      <button
        aria-label={title}
        className="graph-node-trigger"
        data-graph-entity-id={data.node.id}
        data-graph-entity-trigger="true"
        onClick={() => data.onSelect(data.node.id)}
        type="button"
      >
        <span className="graph-node-title-row">
          <strong>{title}</strong>
          {data.node.status.state === "built" ? null : (
            <span className="graph-node-status-chip">{data.node.status.state}</span>
          )}
        </span>
        <span className="graph-node-summary">{data.node.summary[data.audience]}</span>
        <span className="graph-node-meta-badges" aria-label={`${title} metadata badges`}>
          <span>{data.node.owner}</span>
          <span>tier {data.node.tier}</span>
          <span>{badgeFlavor}</span>
        </span>
      </button>
      <div aria-label={`${title} graph controls`} className="graph-node-actions nodrag">
        {data.node.childIds.length > 0 ? (
          <button
            aria-expanded={data.expanded}
            aria-label={`${data.expanded ? "Collapse" : "Expand"} ${title}`}
            onClick={() =>
              data.expanded
                ? data.onCollapse?.(data.node.id)
                : data.onExpand?.(data.node.id)
            }
            type="button"
          >
            {data.expanded ? "Collapse" : "Expand"}
          </button>
        ) : null}
        {(["upstream", "downstream", "isolate"] as const).map((mode) => (
          <button
            className="graph-node-action-trace"
            aria-label={`Trace ${mode} ${mode === "isolate" ? "" : "from "}${title}`.replace(
              "  ",
              " ",
            )}
            aria-pressed={data.traceMode === mode}
            key={mode}
            onClick={() => data.onTraceModeChange?.(data.node.id, mode)}
            type="button"
          >
            {mode}
          </button>
        ))}
      </div>
      <div className="graph-node-details">
        <ul aria-label={`${data.node.title[data.audience]} seam ports`}>
          {data.ports.map((port) => (
            <li key={port.id}>
              {port.name} - {port.channel} - {port.direction}
            </li>
          ))}
        </ul>
      </div>
      <Handle position={Position.Right} type="source" />
    </article>
  );
}
