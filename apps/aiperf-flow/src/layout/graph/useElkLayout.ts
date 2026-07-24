/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! The measure→layout→apply seam. `useElkLayout` waits for React Flow to measure every node, then
//! computes positions with ELK and overlays them onto the *live* input nodes — so a diagram whose
//! node `data` changes under stable ids (a step simulator recoloring nodes, a selection restyle)
//! stays current without relayout. Returns a `laidOut` flag a canvas uses to fade in only once
//! positions are real. Must be called inside a `ReactFlowProvider`.
//!
//! Two robustness guarantees:
//!  - A synchronous deterministic `fallbackLayout` is applied the instant nodes are measured, so
//!    the canvas is NEVER blank waiting on ELK's async pass (elkjs runs in a Web Worker in the
//!    browser, which can fail to resolve); ELK then refines the seed when/if it resolves.
//!  - Positions are kept as a separate `id → {x,y}` map, decoupled from node `data`, so live data
//!    updates never get clobbered by (or clobber) the computed layout.

import { useEffect, useMemo, useRef, useState } from "react";
import { useNodesInitialized, useReactFlow } from "@xyflow/react";
import type { Edge, Node, XYPosition } from "@xyflow/react";
import { layoutGraph, fallbackLayout } from "./elkEngine.js";
import type { ElkOptions } from "./elkEngine.js";

export interface UseElkLayoutResult {
  /** Live input nodes with the latest computed positions overlaid. */
  nodes: Node[];
  /** True once a layout pass has been applied and positions are real. */
  laidOut: boolean;
}

function positionMap(nodes: Node[]): Map<string, XYPosition> {
  return new Map(nodes.map((n) => [n.id, n.position]));
}

/**
 * Lays out `inputNodes`/`edges` with ELK whenever the graph identity (node ids + edge ids) or
 * `opts` change, gated on React Flow having measured the nodes, and fits the view after each pass.
 * Node `data` changes that keep the same ids do NOT trigger relayout — the existing positions are
 * reused and the fresh data is rendered. The `opts` object should be stable (memoize at the call
 * site) to avoid redundant relayouts.
 */
export function useElkLayout(inputNodes: Node[], edges: Edge[], opts: ElkOptions = {}): UseElkLayoutResult {
  const nodesInitialized = useNodesInitialized();
  const { getNodes, fitView } = useReactFlow();
  const [positions, setPositions] = useState<Map<string, XYPosition>>(new Map());
  const [laidOut, setLaidOut] = useState(false);

  // Relayout only when the graph's structural identity changes (new level/subgraph), not on data.
  const graphKey = `${inputNodes.map((n) => n.id).join(",")}|${edges.map((e) => e.id).join(",")}`;
  // Serialized options key so an inline `opts={}` at the call site does NOT re-run the effect every
  // render (object identity churn) — that would be an infinite relayout loop. Only real option
  // changes matter; `laneOf`'s identity is intentionally ignored (its presence is enough).
  const optsKey = `${opts.direction ?? "RIGHT"}|${opts.nodeSpacing ?? ""}|${opts.layerSpacing ?? ""}|${opts.laneOf ? "lane" : ""}`;
  const lastKeyRef = useRef<string>("");
  if (lastKeyRef.current !== graphKey) {
    lastKeyRef.current = graphKey;
    // Drop stale positions and hide until the new graph is seeded (no flash of the old layout).
    setPositions(new Map());
    setLaidOut(false);
  }

  // NOT gated on `nodesInitialized`: in practice it can stay false (nested/animated canvases), which
  // would leave the diagram unlaid-out and hidden forever. Instead we lay out immediately with
  // whatever sizes React Flow has measured so far (falling back to default box sizes when it has
  // measured none) — that still yields a real ELK layout, never a blank canvas — and re-run when
  // `nodesInitialized` flips true to refine with exact measured sizes.
  useEffect(() => {
    let cancelled = false;

    const run = (): void => {
      // Prefer React Flow's own node objects — they carry `.measured` sizes the input nodes lack.
      const measured = getNodes();
      const byId = new Map(measured.map((n) => [n.id, n]));
      const sized = inputNodes.map((n) => ({ ...n, measured: byId.get(n.id)?.measured }));

      // Seed synchronously so the canvas is visible immediately, independent of ELK's async pass.
      setPositions(positionMap(fallbackLayout(sized, edges, opts)));
      setLaidOut(true);

      // Refine with ELK's layered/orthogonal routing; overwrite the seed when it resolves.
      void layoutGraph(sized, edges, opts).then((laid) => {
        if (cancelled) return;
        setPositions(positionMap(laid));
        requestAnimationFrame(() => {
          if (!cancelled) fitView({ padding: 0.2 });
        });
      });
    };

    run();
    requestAnimationFrame(() => {
      if (!cancelled) fitView({ padding: 0.2 });
    });
    return () => {
      cancelled = true;
    };
    // graphKey/optsKey are stable string identities; nodesInitialized re-triggers a refine pass
    // once real measurements exist. Using optsKey (not the opts object) tolerates inline opts.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nodesInitialized, graphKey, optsKey]);

  // Overlay computed positions onto the live input nodes so data updates always show through.
  const nodes = useMemo(
    () => inputNodes.map((n) => (positions.has(n.id) ? { ...n, position: positions.get(n.id)! } : n)),
    [inputNodes, positions],
  );

  return { nodes, laidOut };
}
