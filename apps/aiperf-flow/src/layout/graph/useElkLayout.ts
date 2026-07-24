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
import { useReactFlow } from "@xyflow/react";
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

/** Escape a node id for use in a CSS attribute selector (falls back to a manual escape). */
function cssEscape(id: string): string {
  if (typeof CSS !== "undefined" && typeof CSS.escape === "function") {
    return CSS.escape(id);
  }
  return id.replace(/["\\\]]/g, "\\$&");
}

/**
 * Lays out `inputNodes`/`edges` with ELK whenever the graph identity (node ids + edge ids) or
 * `opts` change, gated on React Flow having measured the nodes, and fits the view after each pass.
 * Node `data` changes that keep the same ids do NOT trigger relayout — the existing positions are
 * reused and the fresh data is rendered. The `opts` object should be stable (memoize at the call
 * site) to avoid redundant relayouts.
 */
export function useElkLayout(inputNodes: Node[], edges: Edge[], opts: ElkOptions = {}): UseElkLayoutResult {
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

  // Lay out immediately, then re-run at two fixed delays. NOT gated on `useNodesInitialized` (which
  // can stay false for nested/animated canvases → blank forever) and NOT subscribed to the store
  // (a size-signature subscription blanked the animated drill-down canvas in the real browser).
  // The first pass uses whatever sizes exist (default box sizes when unmeasured — still a real ELK
  // layout, never blank); the delayed passes catch React Flow's real measured (text-wrapped) sizes
  // so boxes stop overlapping, and re-fit after the drill-in animation has settled.
  useEffect(() => {
    let cancelled = false;
    let seeded = false;

    const run = (): void => {
      // Measure each node's true rendered footprint straight from the DOM. React Flow's own
      // `getNodes().measured` is unreliable on a canvas that mounts inside an animation (it can stay
      // unmeasured, so ELK would reserve default box sizes and tall text-wrapped cards overlap).
      // The rendered element's offsetWidth/offsetHeight is the real layout size (CSS transforms do
      // not affect it), so ELK reserves each box's actual space and boxes stop overlapping.
      const measured = getNodes();
      const rfById = new Map(measured.map((n) => [n.id, n]));
      const sized = inputNodes.map((n) => {
        const el =
          typeof document !== "undefined"
            ? (document.querySelector(`.react-flow__node[data-id="${cssEscape(n.id)}"]`) as HTMLElement | null)
            : null;
        const dom =
          el && el.offsetWidth > 0 ? { width: el.offsetWidth, height: el.offsetHeight } : undefined;
        return { ...n, measured: dom ?? rfById.get(n.id)?.measured };
      });

      // Seed synchronously on the first pass so the canvas is visible immediately.
      if (!seeded) {
        setPositions(positionMap(fallbackLayout(sized, edges, opts)));
        setLaidOut(true);
        seeded = true;
      }

      // Refine with ELK's layered/orthogonal routing; overwrite when it resolves, then fit.
      void layoutGraph(sized, edges, opts).then((laid) => {
        if (cancelled) return;
        setPositions(positionMap(laid));
        setLaidOut(true);
        requestAnimationFrame(() => {
          // maxZoom 1 keeps text at its natural size when the graph fits; larger graphs settle at
          // the flow's minZoom and stay pannable rather than shrinking to unreadable.
          if (!cancelled) fitView({ padding: 0.16, maxZoom: 1 });
        });
      });
    };

    run();
    // Re-run once measurements have likely landed, and again after the drill-in animation settles.
    const t1 = setTimeout(run, 250);
    const t2 = setTimeout(run, 700);
    return () => {
      cancelled = true;
      clearTimeout(t1);
      clearTimeout(t2);
    };
    // graphKey/optsKey are stable string identities; opts identity is intentionally not a dep.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [graphKey, optsKey]);

  // Overlay computed positions onto the live input nodes so data updates always show through.
  const nodes = useMemo(
    () => inputNodes.map((n) => (positions.has(n.id) ? { ...n, position: positions.get(n.id)! } : n)),
    [inputNodes, positions],
  );

  return { nodes, laidOut };
}
