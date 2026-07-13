// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// <reference lib="webworker" />

import ELK from "elkjs/lib/elk.bundled.js";
import elkWorkerUrl from "elkjs/lib/elk-worker.min.js?url";

import {
  LAYOUT_PROTOCOL_VERSION,
  LayoutWorkerRequestSchema,
  composeBandLayouts,
  type LayoutRequest,
  type RelativeBandLayout,
} from "./layout-protocol";

const elk = new ELK({
  workerFactory: (url?: string) => {
    if (!url) {
      throw new Error("ELK worker URL is required");
    }
    return new Worker(url);
  },
  workerUrl: elkWorkerUrl,
});
const NODE_HEIGHT = 156;
const NODE_WIDTH = 320;

interface ElkLayoutNode {
  children?: ElkLayoutNode[];
  height?: number;
  id: string;
  width?: number;
  x?: number;
  y?: number;
}

function bandHierarchyNodes(
  request: LayoutRequest,
  bandId: string,
): ElkLayoutNode[] {
  const nodes = request.nodes.filter((node) => node.bandId === bandId);
  const byId = new Map<string, ElkLayoutNode>(
    nodes.map(({ id }) => [
      id,
      { children: [], height: NODE_HEIGHT, id, width: NODE_WIDTH },
    ]),
  );
  const roots: ElkLayoutNode[] = [];
  for (const { id, parentId } of nodes) {
    const current = byId.get(id);
    if (!current) {
      continue;
    }
    const parent = parentId ? byId.get(parentId) : undefined;
    if (!parent || parent === current) {
      roots.push(current);
      continue;
    }
    parent.children = [...(parent.children ?? []), current];
  }
  return roots;
}

function flattenElkNodes(
  nodes: readonly ElkLayoutNode[],
  offsetX = 0,
  offsetY = 0,
): Array<{ id: string; x: number; y: number }> {
  const positions: Array<{ id: string; x: number; y: number }> = [];
  for (const node of nodes) {
    const x = offsetX + (node.x ?? 0);
    const y = offsetY + (node.y ?? 0);
    positions.push({ id: node.id, x, y });
    if (node.children && node.children.length > 0) {
      positions.push(...flattenElkNodes(node.children, x, y));
    }
  }
  return positions;
}

async function layoutBand(
  request: LayoutRequest,
  bandId: string,
): Promise<RelativeBandLayout> {
  const nodes = request.nodes.filter((node) => node.bandId === bandId);
  const nodeIds = new Set(nodes.map(({ id }) => id));
  const graph = await elk.layout({
    id: bandId,
    layoutOptions: {
      "elk.algorithm": "layered",
      "elk.direction": request.perspective === "ownership" ? "RIGHT" : "DOWN",
      "elk.edgeRouting": "ORTHOGONAL",
      "elk.layered.considerModelOrder.strategy": "NODES_AND_EDGES",
      "elk.spacing.nodeNode": "44",
      "elk.layered.spacing.nodeNodeBetweenLayers": "76",
    },
    children: bandHierarchyNodes(request, bandId),
    edges: request.edges
      .filter(({ from, to }) => nodeIds.has(from) && nodeIds.has(to))
      .map(({ from, id, to }) => ({
        id,
        sources: [from],
        targets: [to],
      })),
  });
  const positions = flattenElkNodes(
    (graph.children as ElkLayoutNode[] | undefined) ?? [],
  );
  return {
    bandId,
    height:
      Math.max(0, ...positions.map(({ y }) => y + NODE_HEIGHT)),
    positions,
    width:
      Math.max(0, ...positions.map(({ x }) => x + NODE_WIDTH)),
  };
}

self.addEventListener(
  "message",
  async (event: MessageEvent<unknown>) => {
    const candidate =
      typeof event.data === "object" && event.data !== null
        ? (event.data as { requestId?: unknown })
        : undefined;
    const fallbackRequestId =
      typeof candidate?.requestId === "number" &&
      Number.isInteger(candidate.requestId) &&
      candidate.requestId >= 0
        ? candidate.requestId
        : 0;
    try {
      const { request, requestId } = LayoutWorkerRequestSchema.parse(event.data);
      const layouts = await Promise.all(
        request.bands.map(({ id }) => layoutBand(request, id)),
      );
      self.postMessage({
        requestId,
        result: composeBandLayouts(request, layouts),
        version: LAYOUT_PROTOCOL_VERSION,
      });
    } catch (error) {
      self.postMessage({
        error: error instanceof Error ? error.message : String(error),
        requestId: fallbackRequestId,
        version: LAYOUT_PROTOCOL_VERSION,
      });
    }
  },
);
