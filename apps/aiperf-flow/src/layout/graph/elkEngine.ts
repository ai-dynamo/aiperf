/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Pure ELK (`elkjs`) layout wrapper for React Flow diagrams. `layoutGraph` maps React Flow
//! nodes/edges into an ELK layered graph — using each node's *measured* size so boxes never
//! overlap — runs `elk.layout()`, and maps the returned coordinates back onto the nodes. A
//! deterministic `fallbackLayout` (longest-path layering) is used if ELK throws (e.g. a
//! restricted test environment), so callers always get non-overlapping positions.
//!
//! No React imports live here: this module is unit-testable in isolation. The measure→apply
//! cycle that feeds real sizes in lives in `useElkLayout`.

import ELK from "elkjs/lib/elk.bundled.js";
import type { Edge, Node } from "@xyflow/react";

/** Fallback footprint for a node React Flow has not measured yet (jsdom reports every size as 0). */
export const DEFAULT_NODE_WIDTH = 220;
export const DEFAULT_NODE_HEIGHT = 90;

/** Layout knobs an adopting diagram passes through `PipelineCanvas`/`AutoLayoutFlow`. */
export interface ElkOptions {
  /** Flow direction. `RIGHT` = left→right (default, matches the request-lifecycle decks). */
  direction?: "RIGHT" | "DOWN";
  /** Spacing between sibling nodes within one layer (px). */
  nodeSpacing?: number;
  /** Spacing between adjacent layers (px). */
  layerSpacing?: number;
  /**
   * Opt-in swimlane partitioning: nodes sharing a `laneOf` key are kept in the same ELK
   * partition (rendered as an aligned band). Omit for a plain layered layout.
   */
  laneOf?: (node: Node) => string;
}

const elk = new ELK();

/** A node's real measured footprint, falling back to the default when React Flow hasn't measured it. */
function nodeSize(node: Node): { width: number; height: number } {
  const width = node.measured?.width ?? node.width ?? DEFAULT_NODE_WIDTH;
  const height = node.measured?.height ?? node.height ?? DEFAULT_NODE_HEIGHT;
  // A measured-but-zero size (jsdom) is treated as unmeasured so the fallback footprint applies.
  return {
    width: width > 0 ? width : DEFAULT_NODE_WIDTH,
    height: height > 0 ? height : DEFAULT_NODE_HEIGHT,
  };
}

/** Stable 0-based partition index per distinct `laneOf` key, in first-seen order. */
function partitionIndices(nodes: Node[], laneOf: (node: Node) => string): Map<string, number> {
  const indices = new Map<string, number>();
  for (const node of nodes) {
    const key = laneOf(node);
    if (!indices.has(key)) {
      indices.set(key, indices.size);
    }
  }
  return indices;
}

/**
 * Lays out `nodes`/`edges` with ELK's layered algorithm and returns new node objects with computed
 * `position`. Node identity, data, and all other fields are preserved. Falls back to a deterministic
 * synchronous layout if ELK is unavailable or throws.
 */
export async function layoutGraph(
  nodes: Node[],
  edges: Edge[],
  opts: ElkOptions = {},
): Promise<Node[]> {
  if (nodes.length === 0) {
    return nodes;
  }
  const direction = opts.direction ?? "RIGHT";
  const nodeSpacing = opts.nodeSpacing ?? 40;
  const layerSpacing = opts.layerSpacing ?? 90;
  const lanes = opts.laneOf ? partitionIndices(nodes, opts.laneOf) : undefined;

  const rootOptions: Record<string, string> = {
    "elk.algorithm": "layered",
    "elk.direction": direction,
    "elk.spacing.nodeNode": String(nodeSpacing),
    "elk.layered.spacing.nodeNodeBetweenLayers": String(layerSpacing),
  };
  if (lanes) {
    rootOptions["elk.partitioning.activate"] = "true";
  }

  const graph = {
    id: "root",
    layoutOptions: rootOptions,
    children: nodes.map((node) => {
      const { width, height } = nodeSize(node);
      const layoutOptions: Record<string, string> = {};
      if (opts.laneOf && lanes) {
        layoutOptions["elk.partitioning.partition"] = String(lanes.get(opts.laneOf(node)));
      }
      return { id: node.id, width, height, layoutOptions };
    }),
    edges: edges.map((edge) => ({ id: edge.id, sources: [edge.source], targets: [edge.target] })),
  };

  try {
    const laid = await elk.layout(graph);
    const positions = new Map<string, { x: number; y: number }>();
    for (const child of laid.children ?? []) {
      positions.set(child.id, { x: child.x ?? 0, y: child.y ?? 0 });
    }
    return nodes.map((node) => ({ ...node, position: positions.get(node.id) ?? node.position }));
  } catch {
    return fallbackLayout(nodes, edges, opts);
  }
}

/**
 * Deterministic longest-path layered layout used when ELK is unavailable. Assigns each node to a
 * layer (its longest incoming-edge depth), then places layers along the flow axis and stacks nodes
 * within a layer along the cross axis, using measured sizes so boxes do not overlap. Not as pretty
 * as ELK, but stable and non-overlapping — enough for tests and as a hard fallback.
 */
export function fallbackLayout(nodes: Node[], edges: Edge[], opts: ElkOptions = {}): Node[] {
  const direction = opts.direction ?? "RIGHT";
  const nodeSpacing = opts.nodeSpacing ?? 40;
  const layerSpacing = opts.layerSpacing ?? 90;

  const incoming = new Map<string, string[]>();
  const ids = new Set(nodes.map((n) => n.id));
  for (const node of nodes) incoming.set(node.id, []);
  for (const edge of edges) {
    if (ids.has(edge.source) && ids.has(edge.target)) {
      incoming.get(edge.target)!.push(edge.source);
    }
  }

  // Longest-path depth via memoized DFS (cycles guarded by a visiting set → depth 0 on back-edge).
  const depthCache = new Map<string, number>();
  const visiting = new Set<string>();
  const depthOf = (id: string): number => {
    if (depthCache.has(id)) return depthCache.get(id)!;
    if (visiting.has(id)) return 0;
    visiting.add(id);
    const preds = incoming.get(id) ?? [];
    const depth = preds.length === 0 ? 0 : 1 + Math.max(...preds.map(depthOf));
    visiting.delete(id);
    depthCache.set(id, depth);
    return depth;
  };

  const layers = new Map<number, Node[]>();
  for (const node of nodes) {
    const d = depthOf(node.id);
    if (!layers.has(d)) layers.set(d, []);
    layers.get(d)!.push(node);
  }

  const positions = new Map<string, { x: number; y: number }>();
  let mainAxis = 0; // position along the flow direction (accumulates per layer)
  const sortedLayers = [...layers.keys()].sort((a, b) => a - b);
  for (const layer of sortedLayers) {
    const members = layers.get(layer)!;
    let crossAxis = 0;
    let maxMain = 0;
    for (const node of members) {
      const { width, height } = nodeSize(node);
      const main = direction === "RIGHT" ? width : height;
      const cross = direction === "RIGHT" ? height : width;
      const pos =
        direction === "RIGHT" ? { x: mainAxis, y: crossAxis } : { x: crossAxis, y: mainAxis };
      positions.set(node.id, pos);
      crossAxis += cross + nodeSpacing;
      maxMain = Math.max(maxMain, main);
    }
    mainAxis += maxMain + layerSpacing;
  }

  return nodes.map((node) => ({ ...node, position: positions.get(node.id) ?? node.position }));
}
