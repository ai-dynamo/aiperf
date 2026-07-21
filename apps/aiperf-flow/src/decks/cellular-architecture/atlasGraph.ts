/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Turns the cellular atlas (`NODES`/`EDGES` in `data.ts`) into real React Flow `Node[]`/`Edge[]`.
//! The source canvas hand-drew SVG `<rect>`/`<path>` coordinates; here the same box positions are
//! expressed only as React Flow `position` hints, and highlight/status become static class strings
//! (never template-interpolated — see the aiperf-flow-diagrams skill's Tailwind JIT trap note).

import type { Edge, Node } from "@xyflow/react";
import { NODES, EDGES, type Lane, type Status } from "./data.js";

// Every possible class combination is a literal string so Tailwind's scanner keeps them.
const LANE_ACCENT: Record<Lane, string> = {
  control: "border-l-4 border-l-category-yellow",
  data: "border-l-4 border-l-category-blue",
  execution: "border-l-4 border-l-category-green",
  results: "border-l-4 border-l-category-purple",
};

const STATUS_BORDER: Record<Status, string> = {
  built: "",
  partial: "border-dashed",
  planned: "border-dashed",
  rejected: "border-dashed",
};

export type GraphOptions = {
  /** Node ids to render at all. */
  visibleNodeIds: Set<string>;
  /** Edge ids to render at all (endpoints must also be visible). */
  visibleEdgeIds: Set<string>;
  /** Node ids on the active route / newly introduced — rendered at full emphasis. */
  activeNodeIds: Set<string>;
  /** Edge ids on the active route / newly introduced — animated `flow` edges. */
  activeEdgeIds: Set<string>;
  /** The currently inspected node/edge id. */
  selectedId: string;
};

function nodeClassName(lane: Lane, status: Status, active: boolean, selected: boolean): string {
  const emphasis = selected
    ? "opacity-100 ring-2 ring-accent-primary bg-surface-panel"
    : active
      ? "opacity-100 ring-1 ring-accent-primary"
      : "opacity-40";
  return `${LANE_ACCENT[lane]} ${STATUS_BORDER[status]} ${emphasis}`;
}

/** Builds the React Flow node/edge model for one atlas view. */
export function buildAtlasGraph(options: GraphOptions): { nodes: Node[]; edges: Edge[] } {
  const { visibleNodeIds, visibleEdgeIds, activeNodeIds, activeEdgeIds, selectedId } = options;

  const nodes: Node[] = NODES.filter((node) => visibleNodeIds.has(node.id)).map((node) => ({
    id: node.id,
    type: "panel",
    position: { x: node.x, y: node.y },
    data: {
      title: node.label,
      detail: node.detail,
      className: nodeClassName(node.lane, node.status, activeNodeIds.has(node.id), selectedId === node.id),
    },
  }));

  const edges: Edge[] = EDGES.filter(
    (edge) => visibleEdgeIds.has(edge.id) && visibleNodeIds.has(edge.from) && visibleNodeIds.has(edge.to),
  ).map((edge) => {
    const active = activeEdgeIds.has(edge.id);
    return {
      id: edge.id,
      source: edge.from,
      target: edge.to,
      type: active ? "flow" : undefined,
      style: active ? undefined : { opacity: 0.25 },
    } satisfies Edge;
  });

  return { nodes, edges };
}
