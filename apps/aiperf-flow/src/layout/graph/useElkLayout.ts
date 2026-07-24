/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! The measure→layout→apply seam. `useElkLayout` waits for React Flow to measure every node,
//! runs `layoutGraph` with those real sizes, and returns laid-out nodes plus a `laidOut` flag a
//! canvas uses to fade in only once positions are real (avoiding a flash at the authored
//! placeholder coordinates). Must be called inside a `ReactFlowProvider`.

import { useEffect, useRef, useState } from "react";
import { useNodesInitialized, useReactFlow } from "@xyflow/react";
import type { Edge, Node } from "@xyflow/react";
import { layoutGraph } from "./elkEngine.js";
import type { ElkOptions } from "./elkEngine.js";

export interface UseElkLayoutResult {
  /** Nodes with ELK-computed positions (or the input nodes until the first layout resolves). */
  nodes: Node[];
  /** True once a layout pass has completed and positions are real. */
  laidOut: boolean;
}

/**
 * Runs an ELK layout pass whenever the node/edge set or options change, gated on React Flow having
 * measured the nodes. Fits the view after each pass. The `opts` object should be stable
 * (memoize it at the call site) to avoid redundant relayouts.
 */
export function useElkLayout(inputNodes: Node[], edges: Edge[], opts: ElkOptions = {}): UseElkLayoutResult {
  const nodesInitialized = useNodesInitialized();
  const { getNodes, fitView } = useReactFlow();
  const [nodes, setNodes] = useState<Node[]>(inputNodes);
  const [laidOut, setLaidOut] = useState(false);

  // Re-seed and re-lay-out when the actual graph identity changes (new level/subgraph).
  const graphKey = `${inputNodes.map((n) => n.id).join(",")}|${edges.map((e) => e.id).join(",")}`;
  const lastKeyRef = useRef<string>("");
  if (lastKeyRef.current !== graphKey) {
    lastKeyRef.current = graphKey;
    // Synchronous reset so a stale laid-out set from the previous graph never flashes.
    if (nodes !== inputNodes) setNodes(inputNodes);
    if (laidOut) setLaidOut(false);
  }

  useEffect(() => {
    if (!nodesInitialized) return;
    let cancelled = false;
    // Prefer React Flow's own node objects — they carry `.measured` sizes the input nodes lack.
    const measured = getNodes();
    const byId = new Map(measured.map((n) => [n.id, n]));
    const sized = inputNodes.map((n) => ({ ...n, measured: byId.get(n.id)?.measured }));
    void layoutGraph(sized, edges, opts).then((laid) => {
      if (cancelled) return;
      setNodes(laid);
      setLaidOut(true);
      // Fit after positions apply; rAF lets React Flow ingest the new coordinates first.
      requestAnimationFrame(() => {
        if (!cancelled) fitView({ padding: 0.2 });
      });
    });
    return () => {
      cancelled = true;
    };
    // graphKey captures inputNodes/edges identity; opts is expected stable per call site.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nodesInitialized, graphKey, opts]);

  return { nodes, laidOut };
}
