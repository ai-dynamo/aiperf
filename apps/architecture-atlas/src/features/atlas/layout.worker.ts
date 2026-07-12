// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// <reference lib="webworker" />

import ELK from "elkjs/lib/elk.bundled.js";

import {
  composeBandLayouts,
  type LayoutRequest,
  type RelativeBandLayout,
} from "./layout-protocol";

const elk = new ELK();

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
      "elk.spacing.nodeNode": "36",
      "elk.layered.spacing.nodeNodeBetweenLayers": "56",
    },
    children: nodes.map(({ id }) => ({ height: 112, id, width: 248 })),
    edges: request.edges
      .filter(({ from, to }) => nodeIds.has(from) && nodeIds.has(to))
      .map(({ from, id, to }) => ({
        id,
        sources: [from],
        targets: [to],
      })),
  });
  const positions = (graph.children ?? []).map((node) => ({
    id: node.id,
    x: node.x ?? 0,
    y: node.y ?? 0,
  }));
  return {
    bandId,
    height:
      Math.max(0, ...positions.map(({ y }) => y + 112)),
    positions,
    width:
      Math.max(0, ...positions.map(({ x }) => x + 248)),
  };
}

self.addEventListener(
  "message",
  async (
    event: MessageEvent<{ request: LayoutRequest; requestId: number }>,
  ) => {
    const { request, requestId } = event.data;
    try {
      const layouts = await Promise.all(
        request.bands.map(({ id }) => layoutBand(request, id)),
      );
      self.postMessage({
        requestId,
        result: composeBandLayouts(request, layouts),
      });
    } catch (error) {
      self.postMessage({
        error: error instanceof Error ? error.message : String(error),
        requestId,
      });
    }
  },
);
