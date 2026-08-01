/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Four connection anchors per node, one per edge midpoint, so an edge can leave and
//! enter whichever side the geometry actually wants instead of always exiting right and
//! entering left.
//!
//! Corner anchors were tried and removed: a connector leaving a corner reads as though it
//! is escaping the box rather than departing a face, and the diagonal it produces is
//! harder to follow than the same hop resolved onto its dominant axis.

import { Handle, Position, type Edge, type Node } from "@xyflow/react";
import { Fragment, type CSSProperties } from "react";

/** Compass anchor ids. Used as React Flow `sourceHandle` / `targetHandle` values. */
export type NodeAnchor = "n" | "e" | "s" | "w";

/**
 * Where each anchor sits on the node box, and which way an edge leaves it.
 *
 * `position` drives React Flow's bezier control point, not placement — placement is the
 * explicit `style`.
 */
const ANCHOR_LAYOUT: Record<NodeAnchor, { position: Position; style: CSSProperties }> = {
  n: { position: Position.Top, style: { left: "50%", top: 0 } },
  e: { position: Position.Right, style: { left: "100%", top: "50%" } },
  s: { position: Position.Bottom, style: { left: "50%", top: "100%" } },
  w: { position: Position.Left, style: { left: 0, top: "50%" } },
};

export const NODE_ANCHORS = Object.keys(ANCHOR_LAYOUT) as readonly NodeAnchor[];

// Anchors are attachment geometry, not decoration: a node showing sixteen dots would be
// unreadable. They stay invisible and non-interactive; the node's own visible handle
// remains the only dot a viewer sees.
const ANCHOR_CLASS_NAME =
  "!h-0 !w-0 !min-h-0 !min-w-0 !border-0 !bg-transparent opacity-0 pointer-events-none";

/**
 * The eight anchors, each as both a source and a target handle.
 *
 * Render this *after* a node's existing id-less `Handle`s. React Flow binds an edge that
 * omits `sourceHandle`/`targetHandle` to the handle whose id is null, so keeping the
 * original handles first and id-less leaves every existing deck's edges untouched.
 */
export function NodeAnchorHandles(): React.JSX.Element {
  return (
    <>
      {NODE_ANCHORS.map((anchor) => {
        const { position, style } = ANCHOR_LAYOUT[anchor];
        const placement = { ...style, transform: "translate(-50%, -50%)" };
        return (
          <Fragment key={anchor}>
            <Handle
              type="target"
              id={anchor}
              position={position}
              className={ANCHOR_CLASS_NAME}
              style={placement}
              isConnectable={false}
            />
            <Handle
              type="source"
              id={anchor}
              position={position}
              className={ANCHOR_CLASS_NAME}
              style={placement}
              isConnectable={false}
            />
          </Fragment>
        );
      })}
    </>
  );
}

/** Fallback node box, for laying out before React Flow has measured anything. */
const DEFAULT_NODE_SIZE = { width: 240, height: 76 };

/** A node's centre, preferring measured dimensions when React Flow has them. */
function centerOf(node: Node): { x: number; y: number } {
  const width = node.measured?.width ?? node.width ?? DEFAULT_NODE_SIZE.width;
  const height = node.measured?.height ?? node.height ?? DEFAULT_NODE_SIZE.height;
  return { x: node.position.x + width / 2, y: node.position.y + height / 2 };
}

/**
 * Pick the facing pair of anchors for the direction between two nodes.
 *
 * Resolved onto whichever axis dominates: a wider-than-tall run leaves east and enters
 * west, a taller-than-wide run leaves south and enters north. A perfect diagonal ties to
 * the horizontal, matching the left-to-right reading order these decks are built around.
 */
export function chooseAnchors(
  from: { x: number; y: number },
  to: { x: number; y: number },
): { source: NodeAnchor; target: NodeAnchor } {
  const dx = to.x - from.x;
  const dy = to.y - from.y;

  if (Math.abs(dx) >= Math.abs(dy)) {
    return dx >= 0 ? { source: "e", target: "w" } : { source: "w", target: "e" };
  }
  return dy >= 0 ? { source: "s", target: "n" } : { source: "n", target: "s" };
}

/**
 * Assign each edge the anchor pair its endpoints' geometry calls for.
 *
 * Opt-in per deck rather than applied inside `Slide`, so decks that hand-tuned their
 * layout around the default right-to-left routing keep rendering exactly as authored.
 * An edge that already names a handle, or whose endpoints are missing, is left alone.
 */
export function autoRouteEdges(nodes: readonly Node[], edges: readonly Edge[]): Edge[] {
  const byId = new Map(nodes.map((node) => [node.id, node]));

  return edges.map((edge) => {
    if (edge.sourceHandle != null || edge.targetHandle != null) return edge;
    const source = byId.get(edge.source);
    const target = byId.get(edge.target);
    if (source === undefined || target === undefined) return edge;

    const { source: sourceHandle, target: targetHandle } = chooseAnchors(
      centerOf(source),
      centerOf(target),
    );
    return { ...edge, sourceHandle, targetHandle };
  });
}
