/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { Edge, Node } from "@xyflow/react";
import { ReactFlow, ReactFlowProvider, Background, BackgroundVariant } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { nodeTypes } from "../../nodes/nodeTypes.js";
import { edgeTypes } from "../../edges/edgeTypes.js";
import { Row } from "../../layout/Row.js";
import { Eyebrow } from "../../prose/Eyebrow.js";
import { inkClassName, strokeClassName } from "../../theme/tokens.js";
import type { StrokeRole } from "../../theme/tokens.js";

//! Shared helpers for the Rust AIPerf architecture deck. All eleven pages compose the same
//! four-part shape: an intro paragraph, one real React Flow node/edge diagram, a Grid of
//! `Callout` cards, and an evidence row of {label, path} source anchors. These builders keep
//! each page's node/edge tables terse while staying inside the app's token vocabulary.

/** Horizontal spacing between diagram columns. */
export const COL = 280;
/** Vertical spacing between stacked bands. */
export const BAND = 200;

/** A band label rendered as a `header` node (the "band" grouping heading from the source canvas). */
export function bandHeader(id: string, title: string, x: number, y: number, caption?: string): Node {
  return { id, type: "header", position: { x, y }, data: { title, caption } };
}

/** A `panel` node: one process/step box with a title and optional detail line. */
export function panel(
  id: string,
  title: string,
  detail: string | undefined,
  x: number,
  y: number,
  strokeRole: StrokeRole = "secondary",
): Node {
  return { id, type: "panel", position: { x, y }, data: { title, detail, strokeRole } };
}

/** A `card` node: an emphasized box with an optional subtitle and detail. */
export function card(
  id: string,
  title: string,
  subtitle: string | undefined,
  detail: string | undefined,
  x: number,
  y: number,
  strokeRole: StrokeRole = "primary",
): Node {
  return { id, type: "card", position: { x, y }, data: { title, subtitle, detail, strokeRole } };
}

/** A `chip` node: a compact label-only tag (catalog rows, families). */
export function chip(id: string, label: string, x: number, y: number): Node {
  return { id, type: "chip", position: { x, y }, data: { label } };
}

/** A solid, animated primary-path edge. */
export function flow(source: string, target: string, label?: string): Edge {
  return { id: `e-${source}-${target}`, source, target, type: "flow", label };
}

/** A dashed, muted optional/delegated/feature-gated edge (rendered slow to read as secondary). */
export function dashed(source: string, target: string, label?: string): Edge {
  return {
    id: `e-${source}-${target}`,
    source,
    target,
    type: "flow",
    label,
    data: { speed: "slow", color: "var(--color-stroke-tertiary)" },
  };
}

/**
 * Standard React Flow canvas frame used by every diagram page. Wraps its own
 * `ReactFlowProvider` — several pages in this deck render more than one `DeckDiagram`
 * side by side, and sibling `<ReactFlow>` instances sharing one ancestor provider
 * silently collide onto the same internal store (only the last-mounted one's nodes
 * actually render). Each instance owning its own provider makes every diagram
 * independent regardless of how many appear on one page.
 */
export function DeckDiagram({
  nodes,
  edges,
  height,
}: {
  nodes: Node[];
  edges: Edge[];
  height: number;
}): React.JSX.Element {
  return (
    <div style={{ height }}>
      <ReactFlowProvider>
        <ReactFlow
          nodeTypes={nodeTypes}
          edgeTypes={edgeTypes}
          nodes={nodes}
          edges={edges}
          fitView
          fitViewOptions={{ padding: 0.15 }}
          nodesDraggable={false}
          proOptions={{ hideAttribution: true }}
        >
          <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="var(--color-stroke-secondary)" />
        </ReactFlow>
      </ReactFlowProvider>
    </div>
  );
}

/** One source-file anchor: a human label plus the repo-relative implementation path. */
export interface EvidenceItem {
  label: string;
  path: string;
}

/**
 * Non-interactive row of source anchors. The source canvas rendered these as file-open buttons;
 * this app has no file-open action, so each becomes a bordered tag showing the label and its
 * implementation path as inline monospace code.
 */
export function EvidenceRow({ items }: { items: ReadonlyArray<EvidenceItem> }): React.JSX.Element {
  return (
    <div>
      <Eyebrow className="mb-2">Source anchors</Eyebrow>
      <Row gap={8} wrap>
        {items.map((item) => (
          <span
            key={item.path + item.label}
            className={`inline-flex items-center gap-2 rounded-md border px-3 py-1 text-xs shadow-sm ${strokeClassName("secondary")}`}
          >
            <span className={`font-medium ${inkClassName("secondary")}`}>{item.label}</span>
            <code className={`${inkClassName("tertiary")}`}>{item.path}</code>
          </span>
        ))}
      </Row>
    </div>
  );
}

/** Intro heading + framing paragraph shared by every page. */
export function PageIntro({ title, children }: { title: string; children: React.ReactNode }): React.JSX.Element {
  return (
    <div>
      <h2 className="text-lg font-semibold">{title}</h2>
      <p className={`mt-1 max-w-3xl text-sm ${inkClassName("secondary")}`}>{children}</p>
    </div>
  );
}
